"""
Machine learning model training script for document classification.
This script can be used to train a more sophisticated classifier using scikit-learn.
"""

import csv
import pickle
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_training_data(csv_file: str = 'training_data.csv'):
    """
    Load training data from CSV file.
    
    Args:
        csv_file: Path to the CSV file containing training data
        
    Returns:
        Tuple of (texts, labels)
    """
    texts, labels = [], []
    
    if not os.path.exists(csv_file):
        logging.error(f"Training data file {csv_file} not found")
        return texts, labels
    
    try:
        with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if 'text' in row and 'label' in row:
                    texts.append(row['text'])
                    labels.append(row['label'])
                else:
                    logging.warning("CSV file must have 'text' and 'label' columns")
                    break
        
        logging.info(f"Loaded {len(texts)} training samples")
        return texts, labels
    
    except Exception as e:
        logging.error(f"Error loading training data: {str(e)}")
        return [], []

def train_model(texts, labels, model_file: str = 'model.pkl'):
    """
    Train a machine learning model for document classification.
    
    Args:
        texts: List of text samples
        labels: List of corresponding labels
        model_file: Path to save the trained model
    """
    if len(texts) == 0:
        logging.error("No training data available")
        return
    
    try:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        # Fit vectorizer and transform training data
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Train logistic regression model
        classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        classifier.fit(X_train_tfidf, y_train)
        
        # Evaluate model
        y_pred = classifier.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"Model accuracy: {accuracy:.3f}")
        logging.info("Classification Report:")
        logging.info(classification_report(y_test, y_pred))
        
        # Save the trained model and vectorizer
        with open(model_file, 'wb') as f:
            pickle.dump((vectorizer, classifier), f)
        
        logging.info(f"Model saved to {model_file}")
        
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")

def load_model(model_file: str = 'model.pkl'):
    """
    Load a trained model from file.
    
    Args:
        model_file: Path to the model file
        
    Returns:
        Tuple of (vectorizer, classifier) or (None, None) if loading fails
    """
    try:
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                vectorizer, classifier = pickle.load(f)
            logging.info(f"Model loaded from {model_file}")
            return vectorizer, classifier
        else:
            logging.warning(f"Model file {model_file} not found")
            return None, None
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None, None

def predict_with_model(text: str, vectorizer, classifier):
    """
    Make a prediction using the trained model.
    
    Args:
        text: Text to classify
        vectorizer: Fitted TF-IDF vectorizer
        classifier: Trained classifier
        
    Returns:
        Prediction label
    """
    try:
        text_tfidf = vectorizer.transform([text])
        prediction = classifier.predict(text_tfidf)[0]
        return prediction
    except Exception as e:
        logging.error(f"Error making prediction: {str(e)}")
        return 'Unknown'

if __name__ == '__main__':
    # Load training data and train model
    texts, labels = load_training_data()
    
    if texts and labels:
        train_model(texts, labels)
        
        # Test the trained model
        vectorizer, classifier = load_model()
        if vectorizer and classifier:
            test_text = "Your electricity bill for this month shows usage of 450 kWh"
            prediction = predict_with_model(test_text, vectorizer, classifier)
            logging.info(f"Test prediction: {prediction}")
    else:
        logging.info("No training data available. Create training_data.csv with 'text' and 'label' columns to train the model.")
