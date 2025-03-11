import pickle
import os
import numpy as np
from pathlib import Path

class ModelPipeline:
    def __init__(self, ambiguity_model_path, intent_model_path, intent_tfidf_path, sentiment_model_path=None):
        """
        Initialize the pipeline with paths to all required models
        
        Args:
            ambiguity_model_path: Path to the ambiguity detection model
            intent_model_path: Path to the intent classifier pickle file
            intent_tfidf_path: Path to the TFIDF vectorizer pickle file
            sentiment_model_path: Path to the sentiment classifier (optional)
        """
        self.ambiguity_model_path = ambiguity_model_path
        self.intent_model_path = intent_model_path
        self.intent_tfidf_path = intent_tfidf_path
        self.sentiment_model_path = sentiment_model_path
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load all models from their respective paths"""
        # Load ambiguity model - assuming it's a language model that needs to be loaded differently
        try:
            """
            this is a temporary place holder -- Mehtab and Dev update it for their model and how it needs to be loaded i.e if it's a full model or a 
            state dictionary or a pkl file, etc
            """
            print(f"Loading ambiguity model from {self.ambiguity_model_path}")
            self.ambiguity_model = None  # Replace with actual loading code
        except Exception as e:
            print(f"Error loading ambiguity model: {e}")
            self.ambiguity_model = None
        

        try:
            print(f"Loading intent classifier from {self.intent_model_path}")
            with open(self.intent_model_path, 'rb') as f:
                self.intent_model = pickle.load(f)
            
            print(f"Loading TFIDF vectorizer from {self.intent_tfidf_path}")
            with open(self.intent_tfidf_path, 'rb') as f:
                self.intent_tfidf = pickle.load(f)
        except Exception as e:
            print(f"Error loading intent classifier or TFIDF: {e}")
            self.intent_model = None
            self.intent_tfidf = None
        

        if self.sentiment_model_path:
            try:
                print(f"Loading sentiment classifier from {self.sentiment_model_path}")
                with open(self.sentiment_model_path, 'rb') as f:
                    self.sentiment_model = pickle.load(f)
            except Exception as e:
                print(f"Error loading sentiment model: {e}")
                self.sentiment_model = None
        else:
            print("No sentiment model path provided. Will look for model during first prediction.")
            self.sentiment_model = None
    
    def find_sentiment_model(self):
        """Try to find sentiment model in the current directory or common locations"""
        common_names = ['sentiment_model.pkl', 'sentiment_classifier.pkl', 'sentiment.pkl']
        common_dirs = ['.', './models', '../models']
        
        for directory in common_dirs:
            for name in common_names:
                path = os.path.join(directory, name)
                if os.path.exists(path):
                    print(f"Found sentiment model at {path}")
                    with open(path, 'rb') as f:
                        self.sentiment_model = pickle.load(f)
                    return True
        
        print("Sentiment model not found. Sentiment classification will be skipped.")
        return False
    
    def predict_ambiguity(self, text):
        """Predict ambiguity score for the input text"""
        if self.ambiguity_model is None:
            return {"error": "Ambiguity model not loaded"}
        
        try:
            # This is a placeholder. Replace with actual prediction code for your model
            # For example, if it's a Hugging Face model:
            # inputs = self.ambiguity_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            # outputs = self.ambiguity_model(**inputs)
            # prediction = outputs.logits.softmax(dim=1).tolist()[0]
            # return {"is_ambiguous": bool(np.argmax(prediction)), "confidence": max(prediction)}
            
            # Placeholder return value
            return {"is_ambiguous": False, "confidence": 0.8}
        except Exception as e:
            return {"error": f"Error in ambiguity prediction: {str(e)}"}
    
    def predict_intent(self, text):
        """Predict intent for the input text"""
        if self.intent_model is None or self.intent_tfidf is None:
            return {"error": "Intent model or TFIDF vectorizer not loaded"}
        
        try:
            # Transform text using the TFIDF vectorizer
            text_transformed = self.intent_tfidf.transform([text])
            
            # Predict intent
            intent_prediction = self.intent_model.predict(text_transformed)[0]
            intent_proba = self.intent_model.predict_proba(text_transformed)[0]
            
            # Get the highest probability and its class
            max_prob_idx = np.argmax(intent_proba)
            confidence = intent_proba[max_prob_idx]
            
            return {
                "intent": intent_prediction,
                "confidence": float(confidence),
                "all_intents": {
                    intent_class: float(prob) 
                    for intent_class, prob in zip(self.intent_model.classes_, intent_proba)
                }
            }
        except Exception as e:
            return {"error": f"Error in intent prediction: {str(e)}"}
    
    def predict_sentiment(self, text):
        """Predict sentiment for the input text"""
        if self.sentiment_model is None:
            # Try to find the sentiment model if it wasn't loaded initially
            if not self.find_sentiment_model():
                return {"error": "Sentiment model not found"}
        
        try:
            # This assumes the sentiment model has a similar API to the intent model
            # Modify as needed for your specific sentiment model
            
            # Check if the model has a vectorizer or if it needs the same TFIDF as intent
            if hasattr(self.sentiment_model, 'predict') and hasattr(self.sentiment_model, 'predict_proba'):
                # Try using the intent TFIDF first
                try:
                    text_transformed = self.intent_tfidf.transform([text])
                    sentiment_prediction = self.sentiment_model.predict(text_transformed)[0]
                    sentiment_proba = self.sentiment_model.predict_proba(text_transformed)[0]
                except:
                    # If that fails, try direct prediction (model might have its own vectorizer)
                    sentiment_prediction = self.sentiment_model.predict([text])[0]
                    sentiment_proba = self.sentiment_model.predict_proba([text])[0]
                
                max_prob_idx = np.argmax(sentiment_proba)
                confidence = sentiment_proba[max_prob_idx]
                
                return {
                    "sentiment": sentiment_prediction,
                    "confidence": float(confidence),
                    "all_sentiments": {
                        sentiment_class: float(prob)
                        for sentiment_class, prob in zip(self.sentiment_model.classes_, sentiment_proba)
                    }
                }
            else:
                # For other types of models that might have different APIs
                return {"error": "Sentiment model has an unsupported interface"}
        except Exception as e:
            return {"error": f"Error in sentiment prediction: {str(e)}"}
    
    def process_text(self, text):
        """
        Process the input text through all models and return results in a dictionary
        
        Args:
            text: The input text to process
            
        Returns:
            A dictionary containing the predictions from all models
        """
        results = {
            "input_text": text,
            "ambiguity": self.predict_ambiguity(text),
            "intent": self.predict_intent(text),
            "sentiment": self.predict_sentiment(text)
        }
        return results


# Example usage
if __name__ == "__main__":
    # Example paths - replace with your actual file paths
    ambiguity_model_path = "path/to/ambiguity_model"
    intent_model_path = "path/to/intent_classifier.pkl"
    intent_tfidf_path = "path/to/intent_tfidf.pkl"
    sentiment_model_path = "path/to/sentiment_model.pkl" 
    
    # Initialize the pipeline
    pipeline = ModelPipeline(
        ambiguity_model_path,
        intent_model_path,
        intent_tfidf_path,
        sentiment_model_path
    )
    
    # Process a sample text
    sample_text = "I'm not sure if I want to cancel my subscription or just pause it for now."
    results = pipeline.process_text(sample_text)
    
    # Print results
    import json
    print(json.dumps(results, indent=2))