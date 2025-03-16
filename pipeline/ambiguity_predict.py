from ambiguitydetector import *


query = 'I need help'

detector = AmbiguityDetector.load_model("ambiguity_model_files/model.pkl")

prediction, probabilities = detector.predict(query)
ambiguity_analysis = detector.analyze_ambiguity(query)

print(f"Prediction: {'Ambiguous' if prediction[0] == 1 else 'Clear'}")