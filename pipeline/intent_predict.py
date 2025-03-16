import joblib
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

print(pd.__version__)


vectorizer = joblib.load("intent_model_files/tfidf_vectorizer.pkl")

# Load the model
model = joblib.load("intent_model_files/intent_classifier.pkl")

# Example usage
sample_query = ["How do I reset my password?"]
sample_query_tfidf = vectorizer.transform(sample_query)
predicted_intent = model.predict(sample_query_tfidf)

print("Predicted Intent:", predicted_intent[0])