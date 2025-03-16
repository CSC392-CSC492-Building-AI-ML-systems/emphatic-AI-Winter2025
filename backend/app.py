import os
import json
from flask import Flask, request, jsonify, Response
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask_cors import CORS
import openai
from dotenv import load_dotenv

from ambiguitydetector import AmbiguityDetector

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

class ResponseGenerator:
    def __init__(self):
        """Initialize the response generator."""
        pass

    def generate_response(self, analysis_results):
        """
        Generate either a clarifying question or a direct answer based on analysis results.
        """

        input_text = analysis_results.get("input_text", "")
        ambiguity_info = analysis_results.get("ambiguity", {})
        intent_info = analysis_results.get("intent", {})
        sentiment_info = analysis_results.get("sentiment", {})

        prompt = self._create_prompt(input_text, ambiguity_info, intent_info, sentiment_info)

        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4-turbo",  
                messages=[
                    {"role": "system", "content": "You are an assistant that either asks a clarifying question or provides a direct answer based on analysis of user input. If the input is ambiguous, ask for clarification. If it's clear, provide a helpful response."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            response_text = completion.choices[0].message.content

            is_clarification = self._is_clarification_question(response_text, ambiguity_info)

            return {
                "response_text": response_text,
                "is_clarification": is_clarification,
                "analysis": {
                    "ambiguity": ambiguity_info,
                    "intent": intent_info,
                    "sentiment": sentiment_info
                }
            }
            
        except Exception as e:
            return {"error": f"Error generating response: {str(e)}"}
    
    def _create_prompt(self, input_text, ambiguity_info, intent_info, sentiment_info):
        """Create a prompt for the language model based on analysis results."""
        is_ambiguous = ambiguity_info.get("is_ambiguous", False)
        ambiguity_analysis = ambiguity_info.get("analysis", {})
        intent = intent_info.get("intent", "unknown")
        sentiment = sentiment_info.get("sentiment", "neutral")
        
        prompt = f"""
User input: "{input_text}"

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

        # Load intent model and TFIDF vectorizer using joblib
        try:
            print(f"Loading intent classifier from {self.intent_model_path}")
            self.intent_model = joblib.load(self.intent_model_path)
            print(f"Loading TFIDF vectorizer from {self.intent_tfidf_path}")
            self.intent_tfidf = joblib.load(self.intent_tfidf_path)
        except Exception as e:
            print(f"Error loading intent classifier or TFIDF: {e}")
            self.intent_model = None
            self.intent_tfidf = None

        # Load sentiment model using Transformers (if path provided)
        if self.sentiment_model_path:
            try:
                print(f"Loading sentiment classifier from {self.sentiment_model_path}")
                self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_path)
                self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_path)
                # Define label names as provided in your sentiment code
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
            # Get prediction and probabilities from the ambiguity detector
            prediction, probabilities = self.ambiguity_detector.predict(text)
            # Assuming prediction[0] == 1 indicates ambiguous
            is_ambiguous = prediction[0] == 1
            confidence = float(np.max(probabilities[0])) if probabilities[0].size > 0 else None

            # Get detailed analysis/explanation
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

ambiguity_model_path = "ambiguity_model_files/model.pkl"
intent_model_path = "intent_model_files/intent_classifier.pkl"
intent_tfidf_path = "intent_model_files/tfidf_vectorizer.pkl"
sentiment_model_path = "final_model_directory"  # Directory for the sentiment model


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
    

    analysis_results = pipeline.process_text(text)
    

    llm_response = response_generator.generate_response(analysis_results)
    print(llm_response)
    # Convert any non-serializable objects to strings and return the JSON response
    response_json = json.dumps(llm_response, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
    return Response(response=response_json, mimetype='application/json')


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
    

    analysis_results = pipeline.process_text(text)

    response = response_generator.generate_response(analysis_results)
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4010, debug=False)