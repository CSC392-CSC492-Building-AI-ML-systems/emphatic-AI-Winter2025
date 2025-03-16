import numpy as np
import pandas as pd
import spacy
import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import datetime
import joblib
import os

nlp = spacy.load("en_core_web_sm")


class AmbiguityDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.feature_names = None

    def extract_features(self, texts):
        """Extract features related to ambiguity from text."""
        features = []
        
        for text in texts:

            doc = nlp(text)
            

            text_features = {}
            
            # 1. Lexical ambiguity 
            polysemy_count = 0
            content_words = 0
            
            for token in doc:
                if token.is_alpha and not token.is_stop and token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                    content_words += 1
                    synsets = wordnet.synsets(token.text)
                    if len(synsets) > 1:
                        polysemy_count += 1
            
            text_features['polysemy_ratio'] = polysemy_count / max(1, content_words)
            
            # 2. Referential ambiguity 
            pronouns = [token for token in doc if token.pos_ == 'PRON']
            nouns = [token for token in doc if token.pos_ in ['NOUN', 'PROPN']]
            text_features['pronoun_count'] = len(pronouns)
            text_features['pronoun_noun_ratio'] = len(pronouns) / max(1, len(nouns))
            
            # 3. Vague quantifiers
            vague_quantifiers = ['some', 'many', 'few', 'several', 'various', 'numerous', 'lot', 'lots']
            vague_count = sum(1 for token in doc if token.text.lower() in vague_quantifiers)
            text_features['vague_quantifier_count'] = vague_count
            
            # 4. Question words without specific context
            question_words = ['who', 'what', 'where', 'when', 'why', 'how']
            question_word_count = sum(1 for token in doc if token.text.lower() in question_words)
            text_features['question_word_count'] = question_word_count
            
            # 5. Sentence complexity measures
            text_features['avg_token_length'] = np.mean([len(token.text) for token in doc if token.is_alpha])
            text_features['sentence_count'] = len(list(doc.sents))
            text_features['avg_sentence_length'] = len(doc) / max(1, text_features['sentence_count'])
            
            # 6. Context specificity (measure of named entities)
            named_entities = len(doc.ents)
            text_features['named_entity_count'] = named_entities
            
            # 7. Conjunction count (potential for ambiguous scope)
            conjunctions = sum(1 for token in doc if token.pos_ == 'CCONJ')
            text_features['conjunction_count'] = conjunctions

            text_features['syntatic_ambiguity'] = self.detect_syntactic_ambiguity(text)
            text_features['modal_verb_count'] = self.detect_modal_verbs(text)
            
            features.append(text_features)
            
        return pd.DataFrame(features)


    def detect_syntactic_ambiguity(self, text):
        doc = nlp(text)
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ('prep', 'pobj') and token.head.pos_ == 'VERB':
                    return True  # Potential syntactic ambiguity
        return False
        
    def detect_modal_verbs(self, text):
        modal_verbs = {'can', 'could', 'may', 'might', 'shall', 'should', 'will', 'would', 'must'}
        doc = nlp(text)
        return sum(1 for token in doc if token.text.lower() in modal_verbs)


    
    def train(self, texts, labels):
        """Train the ambiguity detection model."""

        X_tfidf = self.vectorizer.fit_transform(texts)
        

        X_custom = self.extract_features(texts)
        

        X_tfidf_array = X_tfidf.toarray()
        X_combined = np.hstack([X_tfidf_array, X_custom.values])
        

        X_train, X_val, y_train, y_val = train_test_split(
            X_combined, labels, test_size=0.2, random_state=42
        )
        

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        

        y_pred = self.model.predict(X_val)
        print(classification_report(y_val, y_pred))
        
        return self
    
    def predict(self, text):
        """Predict if a text is ambiguous."""
        if isinstance(text, str):
            text = [text]
            

        X_tfidf = self.vectorizer.transform(text)
        X_custom = self.extract_features(text)
        

        X_tfidf_array = X_tfidf.toarray()
        X_combined = np.hstack([X_tfidf_array, X_custom.values])
        

        return self.model.predict(X_combined), self.model.predict_proba(X_combined)
    
    def analyze_ambiguity(self, text):
        """Analyze why a text might be ambiguous and return explanation."""
        doc = nlp(text)
        
        ambiguity_types = []
        explanations = []
        
        # Check for lexical ambiguity
        polysemous_words = []
        for token in doc:
            if token.is_alpha and not token.is_stop and token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                synsets = wordnet.synsets(token.text)
                
                #---------------- added synonym checking

                synonyms = set()
                for syn in synsets:
                    synonyms.update(syn.lemma_names())

                #-------------------------
                
                
                if len(synsets) > 2 and len(synonyms) > 4:  # Only consider highly ambiguous words
                    polysemous_words.append(token.text)
        
        if polysemous_words:
            ambiguity_types.append("Lexical Ambiguity")
            explanations.append(f"Words with multiple meanings: {', '.join(polysemous_words[:3])}")
        
        # Check for referential ambiguity
        pronouns = [token.text for token in doc if token.pos_ == 'PRON']
        if len(pronouns) > 2 and len([t for t in doc if t.pos_ in ['NOUN', 'PROPN']]) > 2:
            ambiguity_types.append("Referential Ambiguity")
            explanations.append(f"Multiple pronouns with potential unclear references: {', '.join(pronouns[:3])}")
        
        # Check for vague quantifiers
        vague_terms = ['some', 'many', 'few', 'several', 'various', 'numerous', 'lot', 'lots', 'tons', 'a bunch', 'plenty'
                       'a couple', 'a handful', 'loads', 'countless', 'a great deal', 'a good amount']
        vague_found = [token.text for token in doc if token.text.lower() in vague_terms]
        if vague_found:
            ambiguity_types.append("Vague Quantifiers")
            explanations.append(f"Vague quantity terms: {', '.join(vague_found)}")
        
        # Check for missing information
        question_words = ['who', 'what', 'where', 'when', 'why', 'how', 'which', "whose", "whom", "whenever", 
                          "wherever", "whichever", "whatever", "however"]
        question_word_found = [token.text for token in doc if token.text.lower() in question_words]
        if question_word_found and len(doc) < 10:
            ambiguity_types.append("Incomplete Information")
            explanations.append("Query appears too short and may lack sufficient context")
        
        return {
            "ambiguity_types": ambiguity_types,
            "explanations": explanations,
            "overall_assessment": "Ambiguous" if ambiguity_types else "Clear"
        }
    
    def save_model(self, filepath=None):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str, optional): Path to save the model. If None, generates a timestamped filename.
        
        Returns:
            str: Path where the model was saved
        """
        if not self.model:
            raise ValueError("No trained model to save. Train the model first.")

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"ambiguity_detector_{timestamp}.pkl"
        

        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        

        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'feature_names': self.feature_names
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
        
        return filepath
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model file
            
        Returns:
            AmbiguityDetector: Loaded model instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        

        try:
            model_data = joblib.load(filepath)

            instance = cls()

            instance.model = model_data['model']
            instance.vectorizer = model_data['vectorizer']
            instance.feature_names = model_data['feature_names']
            
            print(f"Model loaded from {filepath}")
            return instance
            
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")


query = 'I need help'

detector = AmbiguityDetector.load_model("ambiguity_model_files/model.pkl")

prediction, probabilities = detector.predict(query)
ambiguity_analysis = detector.analyze_ambiguity(query)

print(f"Prediction: {'Ambiguous' if prediction[0] == 1 else 'Clear'}")