"""
Phishing URL Detection Model Training Script
Author: Rananjay Singh Chauhan
Description: Comparative analysis of machine learning models for phishing URL detection
"""

# ==================== Imports ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from keras.models import Model
from keras.layers import Input, Dense
import joblib

# ==================== Data Loading & Preparation ====================
print("\n[1/6] Loading and preprocessing dataset...")
phishing_data = pd.read_csv(r"C:\Users\ranan\OneDrive\Documents\Phishing app\Datasets\URL\urldata.csv")

# Initial dataset inspection
print("\nDataset sample:")
print(phishing_data.head())

# Feature engineering
print("\nRemoving domain column...")
processed_data = phishing_data.drop('Domain', axis=1)

# ==================== Exploratory Data Analysis ====================
print("\n[2/6] Performing exploratory analysis...")
plt.figure(figsize=(14,10))
sns.heatmap(processed_data.corr(), cmap='viridis', annot=False)
plt.title("Feature Correlation Matrix")
plt.show()

# ==================== Data Splitting & Normalization ====================
print("\n[3/6] Preparing training data...")
target = processed_data['Label']
features = processed_data.drop('Label', axis=1)

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== Model Training Framework ====================
model_registry = {}
performance_metrics = {}

def train_model(model_name, model, use_scaling=True):
    print(f"\nTraining {model_name}...")
    X_tr = X_train_scaled if use_scaling else X_train
    X_te = X_test_scaled if use_scaling else X_test
    
    model.fit(X_tr, y_train)
    
    # Performance evaluation
    train_acc = model.score(X_tr, y_train)
    test_acc = model.score(X_te, y_test)
    print(f"{model_name} Training Accuracy: {train_acc:.2%}")
    print(f"{model_name} Validation Accuracy: {test_acc:.2%}")
    
    # Detailed evaluation
    y_pred = model.predict(X_te)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()
    
    model_registry[model_name] = model
    performance_metrics[model_name] = test_acc
    return test_acc

# ==================== Model Training Execution ====================
print("\n[4/6] Training machine learning models...")

# Decision Tree
dt_params = {'max_depth': 7, 'min_samples_split': 5}
dt_score = train_model("Decision Tree", 
                      DecisionTreeClassifier(**dt_params), False)

# Random Forest
rf_score = train_model("Random Forest", 
                      RandomForestClassifier(n_estimators=100, max_depth=9), False)

# Neural Network
nn_score = train_model("Multilayer Perceptron", 
                      MLPClassifier(hidden_layer_sizes=(128, 64, 32), 
                                   alpha=0.001, early_stopping=True))

# XGBoost
xgb_score = train_model("XGBoost", 
                       XGBClassifier(learning_rate=0.3, max_depth=6, 
                                    n_estimators=150))

# Support Vector Machine
svm_score = train_model("Support Vector Machine", 
                       SVC(C=1.5, kernel='rbf', probability=True))

# ==================== Autoencoder Feature Learning ====================
print("\n[5/6] Training autoencoder feature extractor...")

# Data normalization for autoencoder
ae_scaler = MinMaxScaler()
X_ae_train = ae_scaler.fit_transform(X_train)
X_ae_test = ae_scaler.transform(X_test)

# Autoencoder architecture
input_dim = X_ae_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoder = Dense(128, activation="relu")(input_layer)
encoder = Dense(64, activation="relu")(encoder)
code_layer = Dense(32, activation='relu')(encoder)
decoder = Dense(64, activation='relu')(code_layer)
decoder = Dense(128, activation='relu')(decoder)
output_layer = Dense(input_dim, activation='sigmoid')(decoder)
autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# Model training
autoencoder.fit(X_ae_train, X_ae_train,
                epochs=25,
                batch_size=128,
                shuffle=True,
                validation_data=(X_ae_test, X_ae_test),
                verbose=0)

# Feature extraction
encoder_model = Model(inputs=input_layer, outputs=code_layer)
X_train_encoded = encoder_model.predict(X_ae_train)
X_test_encoded = encoder_model.predict(X_ae_test)

# Classifier training on encoded features
from sklearn.linear_model import LogisticRegression
ae_classifier = LogisticRegression(C=0.8)
ae_classifier.fit(X_train_encoded, y_train)
ae_score = ae_classifier.score(X_test_encoded, y_test)
print(f"\nAutoencoder Feature Classifier Accuracy: {ae_score:.2%}")
performance_metrics["Autoencoder+LogisticRegression"] = ae_score

# ==================== Results Analysis ====================
print("\n[6/6] Generating performance report...")

# Comparative analysis
performance_df = pd.DataFrame({
    'Model': list(performance_metrics.keys()),
    'Validation Accuracy': list(performance_metrics.values())
}).sort_values('Validation Accuracy', ascending=False)

print("\nModel Performance Summary:")
print(performance_df)

# Visualization
plt.figure(figsize=(12,6))
sns.barplot(x='Validation Accuracy', y='Model', data=performance_df, palette='viridis')
plt.title("Model Performance Comparison")
plt.xlabel("Accuracy Score")
plt.ylabel("Machine Learning Model")
plt.show()

# Model deployment
best_model_name = performance_df.iloc[0]['Model']
print(f"\nOptimal model selected for deployment: {best_model_name}")

if 'XGBoost' in best_model_name:
    joblib.dump(model_registry[best_model_name], 'production_model.pkl')
    print("XGBoost model exported for production use")
else:
    print("Note: While XGBoost generally offers strong performance, final model selection should consider computational requirements and interpretability needs")