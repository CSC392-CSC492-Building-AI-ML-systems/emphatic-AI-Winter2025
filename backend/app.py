import os
import json
from flask import Flask, request, jsonify, Response, session
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from flask.json import JSONEncoder
from ambiguitydetector import AmbiguityDetector
import json

load_dotenv()

class ResponseGenerator:
    def __init__(self):
        """Initialize the response generator."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.conversation_history = {}
        # Track if we're in a clarification loop
        self.clarification_state = {}

    def generate_response(self, analysis_results, session_id=None):
        """
        Generate either a clarifying question or a direct answer based on analysis results.
        Maintains conversation context if session_id is provided.
        """
        input_text = analysis_results.get("input_text", "")
        ambiguity_info = analysis_results.get("ambiguity", {})
        intent_info = analysis_results.get("intent", {})
        sentiment_info = analysis_results.get("sentiment", {})

        def clean_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            else:
                return obj

        ambiguity_info = clean_for_json(ambiguity_info)
        intent_info = clean_for_json(intent_info)
        sentiment_info = clean_for_json(sentiment_info)

        # Check clarification state for this session
        is_response_to_clarification = False
        if session_id:
            if session_id not in self.clarification_state:
                self.clarification_state[session_id] = False
            else:
                # If the previous response was a clarification question
                is_response_to_clarification = self.clarification_state[session_id]
        
        # Override ambiguity if this is a response to a clarification
        if is_response_to_clarification:
            # We're in a clarification flow, so treat this as a direct answer even if it seems ambiguous
            ambiguity_info["is_ambiguous"] = False
            ambiguity_info["original_ambiguity"] = ambiguity_info.get("is_ambiguous", False)
            # Add context about previous clarification to the analysis
            ambiguity_info["context"] = "Response to previous clarification question"

        if session_id and session_id not in self.conversation_history:
            self.conversation_history[session_id] = [
                {"role": "system", "content": "You are an assistant that either asks a clarifying question or provides a direct answer based on analysis of user input. If the input is ambiguous, ask for clarification. If it's clear, provide a helpful response. Maintain context from previous messages in the conversation."}
            ]
        
        prompt = self._create_prompt(input_text, ambiguity_info, intent_info, sentiment_info)
        current_message = {"role": "user", "content": prompt}
        
        try:
            # Use conversation history if available
            if session_id and session_id in self.conversation_history:
                messages = self.conversation_history[session_id] + [current_message]
            else:
                messages = [
                    {"role": "system", "content": "You are an assistant that either asks a clarifying question or provides a direct answer based on analysis of user input. If the input is ambiguous, ask for clarification. If it's clear, provide a helpful response."},
                    current_message
                ]
            
            completion = self.client.chat.completions.create(
                model="gpt-4-turbo",  
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )

            response_text = completion.choices[0].message.content
            
            if session_id:
                # Add the user message and assistant response to history
                if session_id not in self.conversation_history:
                    self.conversation_history[session_id] = messages
                else:
                    # Add just the new messages
                    self.conversation_history[session_id].append(current_message)
                    self.conversation_history[session_id].append({"role": "assistant", "content": response_text})
                
                # Limit history size to prevent token limits (optional)
                if len(self.conversation_history[session_id]) > 20:  # Arbitrary limit
                    # Keep system message and last N messages
                    system_message = self.conversation_history[session_id][0]
                    self.conversation_history[session_id] = [system_message] + self.conversation_history[session_id][-19:]

            # Determine if the response is a clarification and update the state
            is_clarification = self._is_clarification_question(response_text, ambiguity_info)
            if session_id:
                self.clarification_state[session_id] = is_clarification

            # Ensure all values are properly serializable
            result = {
                "response_text": str(response_text),
                "is_clarification": bool(is_clarification),
                "was_response_to_clarification": bool(is_response_to_clarification),
                "analysis": {
                    "ambiguity": ambiguity_info,
                    "intent": intent_info,
                    "sentiment": sentiment_info
                },
                "session_id": str(session_id) if session_id else None
            }
            
            return result

        except Exception as e:
            return {"error": f"Error generating response: {str(e)}"}
    
    def _create_prompt(self, input_text, ambiguity_info, intent_info, sentiment_info):
        is_ambiguous = ambiguity_info.get("is_ambiguous", False)
        ambiguity_analysis = ambiguity_info.get("analysis", {})
        intent = intent_info.get("intent", "unknown")
        sentiment = sentiment_info.get("sentiment", "neutral")
        
        # Check if this is a response to a clarification
        context_note = ""
        if "context" in ambiguity_info:
            context_note = f"\nContext: {ambiguity_info['context']}"
        
        prompt = f"""
User input: "{input_text}"{context_note}

Analysis:
- Ambiguity: {'Ambiguous' if is_ambiguous else 'Not ambiguous'}
- Intent: {intent}
- Sentiment: {sentiment}

Detailed ambiguity analysis: {json.dumps(ambiguity_analysis)}

Based on this analysis, {'ask a clarifying question to resolve the ambiguity' if is_ambiguous else 'provide a direct and helpful answer'}.
If the input is ambiguous, focus your clarification question on the specific part that needs clarification.
If the input is clear, provide a helpful and friendly response that addresses the user's intent.

Your response:
"""
        return prompt
    
    def _is_clarification_question(self, response_text, ambiguity_info):
        """
        Determine if the generated response is a clarification question.
        This is a simple heuristic - you might want to improve it.
        """
        if ambiguity_info.get("is_ambiguous", False):
            return True
        
        if response_text.strip().endswith("?"):
            return True
            
        clarification_prefixes = [
            "could you clarify",
            "can you please specify",
            "would you mind explaining",
            "i'm not sure if you mean",
            "do you mean",
            "did you mean",
            "could you please explain",
            "i need more information about",
            "please provide more details"
        ]
        
        response_lower = response_text.lower()
        for prefix in clarification_prefixes:
            if prefix in response_lower[:100]:  
                return True
                
        return False

    def clear_session(self, session_id):
        """Clear conversation history for a session while preserving the system message."""
        if session_id in self.conversation_history:
            system_message = self.conversation_history[session_id][0]
            self.conversation_history[session_id] = [system_message]
            # Reset clarification state
            self.clarification_state[session_id] = False
            return True
        return False

class ModelPipeline:
    def __init__(self, ambiguity_model_path, intent_model_path, intent_tfidf_path, sentiment_model_path=None):
        """
        Initialize the pipeline with paths to all required models.
        
        Args:
            ambiguity_model_path (str): Path to the ambiguity detection model.
            intent_model_path (str): Path to the intent classifier file.
            intent_tfidf_path (str): Path to the TFIDF vectorizer file.
            sentiment_model_path (str, optional): Path to the sentiment classifier directory.
        """
        self.ambiguity_model_path = ambiguity_model_path
        self.intent_model_path = intent_model_path
        self.intent_tfidf_path = intent_tfidf_path
        self.sentiment_model_path = sentiment_model_path

        self.load_models()

    def load_models(self):
        """Load all models from their respective paths."""
        # Load ambiguity model using AmbiguityDetector
        try:
            print(f"Loading ambiguity model from {self.ambiguity_model_path}")
            self.ambiguity_detector = AmbiguityDetector.load_model(self.ambiguity_model_path)
        except Exception as e:
            print(f"Error loading ambiguity model: {e}")
            self.ambiguity_detector = None

        try:
            print(f"Loading intent classifier from {self.intent_model_path}")
            self.intent_model = joblib.load(self.intent_model_path)
            print(f"Loading TFIDF vectorizer from {self.intent_tfidf_path}")
            self.intent_tfidf = joblib.load(self.intent_tfidf_path)
        except Exception as e:
            print(f"Error loading intent classifier or TFIDF: {e}")
            self.intent_model = None
            self.intent_tfidf = None

        if self.sentiment_model_path:
            try:
                print(f"Loading sentiment classifier from {self.sentiment_model_path}")
                self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_path)
                self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_path)
                self.sentiment_label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
            except Exception as e:
                print(f"Error loading sentiment model: {e}")
                self.sentiment_model = None
                self.sentiment_tokenizer = None
                self.sentiment_label_names = None
        else:
            print("No sentiment model path provided. Sentiment prediction will be skipped.")
            self.sentiment_model = None
            self.sentiment_tokenizer = None
            self.sentiment_label_names = None

    def predict_ambiguity(self, text):
        """Predict ambiguity score and analysis for the input text.""" 
        if self.ambiguity_detector is None:
            return {"error": "Ambiguity model not loaded"}
        try:
            prediction, probabilities = self.ambiguity_detector.predict(text)
            is_ambiguous = prediction[0] == 1
            confidence = float(np.max(probabilities[0])) if probabilities[0].size > 0 else None

            analysis = self.ambiguity_detector.analyze_ambiguity(text)

            return {
                "is_ambiguous": is_ambiguous,
                "confidence": confidence,
                "analysis": analysis
            }
        except Exception as e:
            return {"error": f"Error in ambiguity prediction: {str(e)}"}

    def predict_intent(self, text):
        """Predict intent for the input text."""
        if self.intent_model is None or self.intent_tfidf is None:
            return {"error": "Intent model or TFIDF vectorizer not loaded"}
        try:
            text_transformed = self.intent_tfidf.transform([text])

            predicted_intent = self.intent_model.predict(text_transformed)[0]
            return {"intent": predicted_intent}
        except Exception as e:
            return {"error": f"Error in intent prediction: {str(e)}"}

    def predict_sentiment(self, text):
        """Predict sentiment for the input text using the Transformers model."""
        if self.sentiment_model is None or self.sentiment_tokenizer is None:
            return {"error": "Sentiment model not loaded"}
        try:
            # Tokenize the input text
            inputs = self.sentiment_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=64
            )
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
            logits = outputs.logits
            predicted_class_id = int(torch.argmax(logits, dim=-1).item())
            predicted_label = self.sentiment_label_names[predicted_class_id]
            confidence = float(torch.softmax(logits, dim=-1)[0][predicted_class_id].item())

            return {
                "sentiment": predicted_label,
                "confidence": confidence,
                "logits": logits.tolist()  
            }
        except Exception as e:
            return {"error": f"Error in sentiment prediction: {str(e)}"}

    def process_text(self, text):
        """
        Process the input text through all models and return results.
        
        Args:
            text (str): The input text to process.
            
        Returns:
            dict: A dictionary containing predictions from ambiguity, intent, and sentiment models.
        """
        results = {
            "input_text": text,
            "ambiguity": self.predict_ambiguity(text),
            "intent": self.predict_intent(text),
            "sentiment": self.predict_sentiment(text)
        }
        return results


app = Flask(__name__)
CORS(app) 
class CustomJSONEncoder(JSONEncoder):
    """Custom JSON encoder to handle various types that aren't natively JSON serializable."""
    def default(self, obj):
        try:
            if hasattr(obj, 'item'):
                return obj.item()
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            return JSONEncoder.default(self, obj)
        except:
            return str(obj)

app.json_encoder = CustomJSONEncoder

ambiguity_model_path = "ambiguity_model_files/model.pkl"
intent_model_path = "intent_model_files/intent_classifier.pkl"
intent_tfidf_path = "intent_model_files/tfidf_vectorizer.pkl"
sentiment_model_path = "final_model_directory"


pipeline = ModelPipeline(
    ambiguity_model_path,
    intent_model_path,
    intent_tfidf_path,
    sentiment_model_path
)


response_generator = ResponseGenerator()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the server is running."""
    return jsonify({"status": "ok", "message": "Server is running"})

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400

    text = data['text']
    
    # Get session ID if provided
    session_id = data.get('session_id')

    analysis_results = pipeline.process_text(text)
    
    # Pass session ID to response generator
    llm_response = response_generator.generate_response(analysis_results, session_id=session_id)
    
    # Use the jsonify function which will use our custom encoder
    return jsonify(llm_response)


@app.route('/predict/ambiguity', methods=['POST'])
def predict_ambiguity():
    """Endpoint to get only ambiguity predictions."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400
    
    text = data['text']
    results = pipeline.predict_ambiguity(text)
    
    return jsonify(results)

@app.route('/predict/intent', methods=['POST'])
def predict_intent():
    """Endpoint to get only intent predictions."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400
    
    text = data['text']
    results = pipeline.predict_intent(text)
    
    return jsonify(results)

@app.route('/predict/sentiment', methods=['POST'])
def predict_sentiment():
    """Endpoint to get only sentiment predictions."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400
    
    text = data['text']
    results = pipeline.predict_sentiment(text)
    
    return jsonify(results)

@app.route('/generate_response', methods=['POST'])
def generate_response():
    """Endpoint to analyze text and generate an appropriate response."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400
    
    text = data['text']
    
    session_id = data.get('session_id')
    
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())

    try:
        analysis_results = pipeline.process_text(text)
        response = response_generator.generate_response(analysis_results, session_id=session_id)
        
        return jsonify(response)
    except Exception as e:
        error_response = {"error": f"Error processing request: {str(e)}"}
        return jsonify(error_response), 500

@app.route('/clear_session', methods=['POST'])
def clear_session():
    """Endpoint to clear the conversation history for a session."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    if 'session_id' not in data:
        return jsonify({"error": "Missing 'session_id' field in request"}), 400
    
    session_id = data['session_id']
    
    if session_id in response_generator.conversation_history:
        system_message = response_generator.conversation_history[session_id][0]
        response_generator.conversation_history[session_id] = [system_message]
        return jsonify({"status": "success", "message": "Conversation history cleared"})
    else:
        return jsonify({"status": "warning", "message": "Session not found"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4010, debug=False)