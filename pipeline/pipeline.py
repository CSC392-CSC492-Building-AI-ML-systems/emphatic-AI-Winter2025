import os
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ambiguitydetector import AmbiguityDetector


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



if __name__ == "__main__":

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

    # Process a sample text
    sample_text = "I'm not sure if I want to cancel my subscription or just pause it for now."
    results = pipeline.process_text(sample_text)

    import json
    print(json.dumps(results, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else x))

