"""
Phishing Email Detection Model Training Script
Author: Rananjay Singh Chauhan, Granth Satsangi
Description: Comparative analysis of machine learning models for phishing URL detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                            recall_score, f1_score, roc_auc_score, roc_curve, 
                            classification_report)
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Load the dataset with proper path formatting
try:
    df = pd.read_csv(r"C:\Users\ranan\OneDrive\Documents\Phishing app\Datasets\URL\spamemail.csv", encoding='ISO-8859-1')
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
    exit()

# Clean and prepare data
def clean_data(df):
    """Clean and preprocess the dataset"""
    # Remove unnamed columns if they exist
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Rename columns for consistency
    df = df.rename(columns={'v1': 'Type', 'v2': 'Message'})
    
    # Create binary spam column
    df['Spam'] = df['Type'].map({'ham': 0, 'spam': 1})
    return df

df = clean_data(df)

# Visualization functions
def plot_distribution(df):
    """Plot class distribution"""
    plt.figure(figsize=(6,6))
    counts = df['Type'].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', 
            colors=['#66b3ff', '#ff9999'], startangle=90)
    plt.title('Email Type Distribution')
    plt.show()

def plot_wordcloud(df, text_column, title):
    """Generate word cloud visualization"""
    text = ' '.join(df[text_column].astype(str))
    
    wordcloud = WordCloud(width=1000, height=500,
                        background_color='white',
                        max_words=200,
                        colormap='Reds').generate(text)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Data visualization
plot_distribution(df)
plot_wordcloud(df[df['Spam'] == 1], 'Message', 'Most Common Words in Spam Emails')

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    df['Message'], 
    df['Spam'],
    test_size=0.25,
    random_state=42,
    stratify=df['Spam']
)

# Enhanced model evaluation function
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Comprehensive model evaluation with proper validation"""
    # Train model
    model.fit(X_train, y_train)
    
    # Generate predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Probability estimates
    proba_train = model.predict_proba(X_train)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'train_precision': precision_score(y_train, y_pred_train),
        'test_precision': precision_score(y_test, y_pred_test),
        'train_recall': recall_score(y_train, y_pred_train),
        'test_recall': recall_score(y_test, y_pred_test),
        'train_f1': f1_score(y_train, y_pred_train),
        'test_f1': f1_score(y_test, y_pred_test),
        'train_roc_auc': roc_auc_score(y_train, proba_train),
        'test_roc_auc': roc_auc_score(y_test, proba_test)
    }
    
    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    for data, color, label in zip(
        [(y_train, proba_train), (y_test, proba_test)],
        ['blue', 'red'],
        ['Training', 'Testing']
    ):
        fpr, tpr, _ = roc_curve(data[0], data[1])
        plt.plot(fpr, tpr, color=color, 
                label=f'{label} (AUC = {roc_auc_score(data[0], data[1]):.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.show()
    
    # Confusion matrices
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for idx, (y_true, y_pred, title) in enumerate(zip(
        [y_train, y_test],
        [y_pred_train, y_pred_test],
        ['Training', 'Testing']
    )):
        sns.heatmap(confusion_matrix(y_true, y_pred),
                   annot=True, fmt='d', cmap='Blues',
                   ax=ax[idx], cbar=False)
        ax[idx].set_title(f'{title} Confusion Matrix')
        ax[idx].set_xlabel('Predicted Label')
        ax[idx].set_ylabel('True Label')
        ax[idx].set_xticklabels(['Ham', 'Spam'])
        ax[idx].set_yticklabels(['Ham', 'Spam'])
    
    plt.tight_layout()
    plt.show()
    
    # Classification reports
    print("\nTraining Classification Report:")
    print(classification_report(y_train, y_pred_train))
    
    print("\nTesting Classification Report:")
    print(classification_report(y_test, y_pred_test))
    
    return metrics

# Create and evaluate model pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(max_features=1000, stop_words='english')),
    ('classifier', MultinomialNB(alpha=0.1))
])

metrics = evaluate_model(pipeline, X_train, X_test, y_train, y_test)

# Enhanced spam detection function
class SpamDetector:
    def __init__(self, model):
        self.model = model
        
    def predict(self, email_text):
        """Make prediction with confidence score"""
        prediction = self.model.predict([email_text])[0]
        proba = self.model.predict_proba([email_text])[0]
        
        return {
            'prediction': 'Spam' if prediction == 1 else 'Ham',
            'confidence': max(proba),
            'spam_probability': proba[1]
        }

# Initialize detector with trained model
detector = SpamDetector(pipeline)

# Example usage
test_emails = [
    "Congratulations! You've won a $1000 prize! Click here to claim!",
    "Meeting reminder: Tomorrow at 2 PM in conference room",
    "Your account needs verification. Please update your details"
]

print("\nSpam Detection Results:")
for email in test_emails:
    result = detector.predict(email)
    print(f"\nEmail: {email[:50]}...")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Spam Probability: {result['spam_probability']:.2%}")