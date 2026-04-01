"""
Email Spam Detection - Full-Stack Backend
==========================================
HOW TO TRAIN WITH YOUR OWN DATA
---------------------------------
1. Place your CSV file at:  backend/data/spam.csv
   Required columns:
       Category  →  "spam" or "ham"
       Message   →  the email/SMS text

2. Run the training script FIRST:
       cd backend
       python train_model.py

3. Then start the API:
       python app.py

The training script saves the model to backend/model/.
app.py loads that saved model on startup automatically.

Enhanced with MLC-grade improvements:
- TF-IDF instead of basic CountVectorizer
- Multiple model comparison (Naive Bayes, Logistic Regression, SVM)
- Text preprocessing pipeline
- REST API with Flask
- Model persistence with joblib
- Logging, error handling, health checks
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import re
import string
import logging
import os
import json
from datetime import datetime

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.calibration import CalibratedClassifierCV
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -- Paths ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(SCRIPT_DIR, "model")
# PUT YOUR CSV FILE IN:  backend/data/spam.csv
DATA_PATH  = os.path.join(SCRIPT_DIR, "data", "spam.csv")

app = Flask(__name__)
CORS(app)

# ============================
# TEXT PREPROCESSING (UPGRADE 1)
# ============================
def preprocess_text(text):
    """
    Enhanced text preprocessing pipeline.
    MLC Upgrade: Proper NLP preprocessing increases accuracy significantly.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', 'URL', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', 'EMAIL', text)
    
    # Remove phone numbers
    text = re.sub(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', 'PHONE', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters but keep meaningful ones
    text = re.sub(r'[^\w\s!?$£€%]', '', text)
    
    return text


# ============================
# MODEL TRAINING
# ============================
class SpamDetector:
    def __init__(self):
        self.models = {}
        self.best_model_name = None
        self.best_pipeline = None
        self.metrics = {}
        self.is_trained = False
        self.training_stats = {}
    
    def build_pipelines(self):
        """
        MLC Upgrade 2: TF-IDF with n-grams instead of basic CountVectorizer.
        MLC Upgrade 3: Multiple model comparison.
        """
        tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),      # Unigrams + bigrams
            sublinear_tf=True,        # Apply log normalization
            min_df=2,                 # Remove rare terms
            strip_accents='unicode',
            analyzer='word',
            stop_words='english'
        )
        
        pipelines = {
            'Naive Bayes': Pipeline([
                ('tfidf', tfidf),
                ('clf', MultinomialNB(alpha=0.1))
            ]),
            'Logistic Regression': Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                    strip_accents='unicode',
                    analyzer='word',
                    stop_words='english'
                )),
                ('clf', LogisticRegression(
                    C=5.0,
                    max_iter=1000,
                    class_weight='balanced',  # Handle imbalance
                    solver='liblinear'
                ))
            ]),
            'SVM': Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                    strip_accents='unicode',
                    analyzer='word',
                    stop_words='english'
                )),
                ('clf', CalibratedClassifierCV(
                    LinearSVC(C=1.0, class_weight='balanced', max_iter=2000)
                ))
            ])
        }
        return pipelines
    
    def train(self, data_path=None):
        """
        Train models on SMS spam dataset.

        data_path priority:
          1. Argument passed to this method
          2. DATA_PATH constant (backend/data/spam.csv)
          3. Built-in sample data (fallback demo only)
        """
        logger.info("Starting model training...")

        # Column aliases: handle common variants automatically
        COLUMN_ALIASES = {
            'v1': 'Category', 'v2': 'Message',
            'label': 'Category', 'class': 'Category', 'target': 'Category',
            'text': 'Message', 'sms': 'Message', 'content': 'Message',
            'email': 'Message', 'body': 'Message',
        }

        # Resolve data path
        resolved_path = data_path or DATA_PATH

        # Load dataset
        if resolved_path and os.path.exists(resolved_path):
            logger.info(f"Loading CSV: {resolved_path}")
            df = pd.read_csv(resolved_path, encoding='latin-1')
            df = df.rename(columns={c: COLUMN_ALIASES[c]
                                     for c in df.columns if c in COLUMN_ALIASES})
        else:
            # Use sample data for demo
            logger.warning(
                f"CSV not found at '{resolved_path}'. "
                "Using built-in sample data.  "
                "To use your own data: place spam.csv in backend/data/ "
                "then run: python train_model.py"
            )
            df = self._get_sample_data()
        
        # Clean data
        df = df[['Category', 'Message']].dropna()
        df = df.drop_duplicates()
        df['Category'] = df['Category'].str.strip().str.lower()
        df = df[df['Category'].isin(['ham', 'spam'])]
        
        # Preprocess text
        df['ProcessedMessage'] = df['Message'].apply(preprocess_text)
        df['Spam'] = (df['Category'] == 'spam').astype(int)
        
        # Record stats
        spam_count = df['Spam'].sum()
        ham_count = len(df) - spam_count
        self.training_stats = {
            'total_samples': len(df),
            'spam_count': int(spam_count),
            'ham_count': int(ham_count),
            'imbalance_ratio': round(ham_count / spam_count, 2),
            'trained_at': datetime.now().isoformat()
        }
        logger.info(f"Dataset: {len(df)} samples, {spam_count} spam, {ham_count} ham")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['ProcessedMessage'], df['Spam'],
            test_size=0.25, random_state=42, stratify=df['Spam']
        )
        
        # Train and evaluate all models
        pipelines = self.build_pipelines()
        best_f1 = 0
        
        for name, pipeline in pipelines.items():
            logger.info(f"Training {name}...")
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': round(accuracy_score(y_test, y_pred), 4),
                'precision': round(precision_score(y_test, y_pred), 4),
                'recall': round(recall_score(y_test, y_pred), 4),
                'f1_score': round(f1_score(y_test, y_pred), 4),
                'roc_auc': round(roc_auc_score(y_test, y_prob), 4),
            }
            
            self.models[name] = pipeline
            self.metrics[name] = metrics
            
            logger.info(f"{name} - F1: {metrics['f1_score']}, AUC: {metrics['roc_auc']}")
            
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                self.best_model_name = name
                self.best_pipeline = pipeline
        
        self.is_trained = True
        logger.info(f"Best model: {self.best_model_name} (F1={best_f1:.4f})")
        
        # Save model
        self._save_model()
        
        return self.metrics
    
    def predict(self, text, model_name=None):
        """
        Predict if text is spam.
        MLC Upgrade 4: Return confidence scores, not just binary predictions.
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        processed = preprocess_text(text)
        
        pipeline = self.models.get(model_name, self.best_pipeline)
        if pipeline is None:
            raise ValueError(f"Model '{model_name}' not found")
        
        prediction = pipeline.predict([processed])[0]
        probability = pipeline.predict_proba([processed])[0]
        
        confidence = float(probability[prediction])
        spam_prob = float(probability[1])
        
        # Detect spam indicators for explainability
        indicators = self._extract_spam_indicators(text)
        
        return {
            'is_spam': bool(prediction),
            'label': 'SPAM' if prediction else 'HAM',
            'confidence': round(confidence * 100, 1),
            'spam_probability': round(spam_prob * 100, 1),
            'ham_probability': round((1 - spam_prob) * 100, 1),
            'indicators': indicators,
            'model_used': model_name or self.best_model_name
        }
    
    def _extract_spam_indicators(self, text):
        """
        MLC Upgrade 5: Feature explainability - explain WHY it's spam.
        """
        indicators = []
        text_lower = text.lower()
        
        spam_patterns = {
            'Urgency words': ['urgent', 'immediately', 'act now', 'limited time', 'expires'],
            'Free offers': ['free', 'prize', 'winner', 'won', 'claim', 'reward'],
            'Financial lures': ['cash', 'money', 'earn', '£', '$', '€', '%', 'credit'],
            'Call to action': ['click', 'call', 'reply', 'respond', 'subscribe'],
            'Contact info': ['http', 'www', '@', 'tel:', 'call us'],
        }
        
        for category, patterns in spam_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    indicators.append({'type': category, 'match': pattern})
                    break
        
        # Text stats
        if len(text) > 0:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.3:
                indicators.append({'type': 'Formatting', 'match': 'Excessive capitals'})
            
            punct_count = sum(1 for c in text if c in '!?')
            if punct_count > 2:
                indicators.append({'type': 'Formatting', 'match': f'{punct_count} exclamation/question marks'})
        
        return indicators
    
    def _get_sample_data(self):
        """Fallback sample data for demo."""
        data = {
            'Category': ['ham'] * 50 + ['spam'] * 20,
            'Message': [
                'Hey, can we meet tomorrow?', 'What time is the meeting?',
                'I love you too!', 'See you at the party', 'Thanks for your help',
                'Call me when you get this', 'Happy birthday!', 'Good morning!',
                'Did you watch the game last night?', 'Are you coming to dinner?',
                'I am not feeling well today', 'Can you pick me up at 7?',
                'The weather is beautiful today', 'Just got home from work',
                'What do you want for lunch?', 'I miss you so much',
                'How was your day?', 'Running a bit late', 'Save my number',
                'Congrats on the promotion!', 'Netflix is down again',
                'Can you check my email?', 'Mom says dinner is ready',
                'I have a headache', "Let's go hiking this weekend",
                'Did you finish the report?', 'Great job on the presentation',
                'My flight is delayed', 'The kids are driving me crazy',
                'Just wanted to say hi', 'Are you free this Saturday?',
                'I will call you tonight', 'Got your package today',
                'Did you see the news?', 'Our team won!',
                'I am stuck in traffic', 'Working from home today',
                'Happy anniversary!', "Can't wait to see you",
                'Just woke up', 'Going to the gym',
                'Have you eaten yet?', 'The meeting is at 3pm',
                'Hope you feel better soon', 'Thanks for the birthday wishes',
                'I need help with my project', 'Great seeing you today',
                'Taking a day off tomorrow', 'Running out of milk',
                'Watch out for the rain',
                # Spam examples
                'FREE! You have won a £2000 prize! Click here to claim NOW!',
                'Urgent: Your account will be closed unless you verify immediately',
                'Win a free iPhone! Text WIN to 12345 now!',
                'You have been selected for a $1000 cash reward! Call 800-555-1234',
                'CONGRATULATIONS! You are our lucky winner! Claim your prize today',
                'Hot singles in your area! Visit now: www.dating.com/free',
                'Limited offer! 80% off all items! Buy now before it expires!',
                'Your loan has been APPROVED! Get cash today, no credit check needed',
                'ACT NOW! Lose 30lbs in 30 days with our secret formula!',
                'You have been randomly selected to receive a FREE vacation!',
                'FINAL NOTICE: Your warranty is expiring. Call immediately to renew',
                'Get rich quick! Work from home and earn $5000/week guaranteed!',
                'FREE entry into our £250 weekly competition. Text FREE to 84080',
                'Important: Your bank account needs verification. Visit our website',
                'Urgent! Reply to claim your free gift worth £500 before midnight',
                'Make money online! No experience needed. Start earning today!',
                'Your Netflix subscription is expiring. Click here to update payment',
                'You won a brand new car! Collect within 24 hours or forfeit prize',
                'Buy cheap medications online! No prescription needed. Best prices!',
                'Dating site alert: You have 5 messages waiting for you!'
            ]
        }
        return pd.DataFrame(data)
    
    def _save_model(self):
        """Save trained model and metadata to MODEL_DIR."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.best_pipeline,
                    os.path.join(MODEL_DIR, 'best_pipeline.pkl'))
        # Also save all pipelines so the API can serve every model
        joblib.dump(self.models,
                    os.path.join(MODEL_DIR, 'all_pipelines.pkl'))
        meta_path = os.path.join(MODEL_DIR, 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'best_model': self.best_model_name,
                'metrics': self.metrics,
                'training_stats': self.training_stats
            }, f, indent=2)
        logger.info(f"Model saved to {MODEL_DIR}/")
    
    def load_model(self):
        """
        Load the pre-trained model from MODEL_DIR.

        Loads:
          - model/best_pipeline.pkl   (always)
          - model/all_pipelines.pkl   (if present, enables per-model selection)
          - model/metadata.json       (metrics + training stats)

        If no saved model exists, falls back to training from DATA_PATH
        or from built-in sample data.
        """
        best_pkl  = os.path.join(MODEL_DIR, 'best_pipeline.pkl')
        all_pkl   = os.path.join(MODEL_DIR, 'all_pipelines.pkl')
        meta_path = os.path.join(MODEL_DIR, 'metadata.json')

        if not os.path.isfile(best_pkl):
            logger.warning(
                f"No saved model found at {best_pkl}. "
                "Training a new model now...  "
                "Tip: run 'python train_model.py' first for better results."
            )
            self.train()
            return

        self.best_pipeline = joblib.load(best_pkl)
        logger.info(f"Loaded best_pipeline from {best_pkl}")

        # Load all pipelines if available (saved by train_model.py or _save_model)
        if os.path.isfile(all_pkl):
            self.models = joblib.load(all_pkl)
            logger.info(f"Loaded all_pipelines: {list(self.models.keys())}")
        else:
            # Fallback: only best is available
            self.models = {}

        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        self.best_model_name = metadata['best_model']
        self.metrics = metadata['metrics']
        self.training_stats = metadata.get('training_stats', {})

        # Make sure the best pipeline is always in the dict
        self.models[self.best_model_name] = self.best_pipeline
        self.is_trained = True
        logger.info(f"Active model: {self.best_model_name}")


# Initialize detector
detector = SpamDetector()

# load_model handles the fallback to training internally
detector.load_model()


# ============================
# API ENDPOINTS
# ============================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_trained': detector.is_trained,
        'best_model': detector.best_model_name,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    
    Request body:
    {
        "text": "Email content here",
        "model": "Logistic Regression"  (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field'}), 400
        
        text = data.get('text', '').strip()
        
        if len(text) < 1:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        if len(text) > 5000:
            return jsonify({'error': 'Text too long (max 5000 chars)'}), 400
        
        model_name = data.get('model', None)
        result = detector.predict(text, model_name=model_name)
        
        return jsonify({
            'success': True,
            'result': result,
            'text_length': len(text),
            'timestamp': datetime.now().isoformat()
        })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple emails.
    MLC Upgrade 6: Batch inference endpoint for production use cases.
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing "texts" field'}), 400
        
        texts = data.get('texts', [])
        
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({'error': 'texts must be a non-empty list'}), 400
        
        if len(texts) > 100:
            return jsonify({'error': 'Max 100 texts per batch'}), 400
        
        results = []
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                results.append({'index': i, 'error': 'Invalid text'})
                continue
            result = detector.predict(text)
            result['index'] = i
            results.append(result)
        
        spam_count = sum(1 for r in results if r.get('is_spam'))
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total': len(texts),
                'spam': spam_count,
                'ham': len(texts) - spam_count,
                'spam_rate': round(spam_count / len(texts) * 100, 1)
            }
        })
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get all trained models and their metrics."""
    return jsonify({
        'best_model': detector.best_model_name,
        'models': detector.metrics,
        'training_stats': detector.training_stats
    })


@app.route('/api/retrain', methods=['POST'])
def retrain():
    """Trigger model retraining."""
    try:
        metrics = detector.train()
        return jsonify({
            'success': True,
            'message': 'Models retrained successfully',
            'best_model': detector.best_model_name,
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Method not allowed'}), 405


if __name__ == '__main__':
    app.run(debug=True, port=5000)