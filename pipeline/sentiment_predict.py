import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


model_dir = "final_model_directory"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)


label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

def predict(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    

    with torch.no_grad():
        outputs = model(**inputs)
    

    logits = outputs.logits
    predicted_class_id = logits.argmax(dim=-1).item()
    predicted_label = label_names[predicted_class_id]
    return predicted_label

if __name__ == "__main__":

    text = "I am really happy with the service I received!"
    prediction = predict(text)
    print("Input text:", text)
    print("Predicted emotion:", prediction)
