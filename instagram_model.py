import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# Function to process Instagram dataset and train a single hybrid model
def process_instagram(dataset_path):
    print("Processing Instagram dataset...")
    
    # Load dataset
    data = pd.read_csv(dataset_path)
    X = data.drop('is_fake', axis=1)  # Assuming 'is_fake' is the target column
    y = data['is_fake']
    
    # Handle missing values
    X.fillna(X.mean(), inplace=True)
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Define and train a single hybrid model
    model = Sequential([
        Dense(128, activation='relu', input_dim=X_train.shape[1]),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1, callbacks=[early_stopping])
    
    # Evaluate the model
    final_preds = (model.predict(X_test) > 0.5).astype(int).flatten()
    final_proba = model.predict(X_test).flatten()
    
    accuracy = accuracy_score(y_test, final_preds)
    auc_score = roc_auc_score(y_test, final_proba)
    report = classification_report(y_test, final_preds)
    cm = confusion_matrix(y_test, final_preds)
    fpr, tpr, _ = roc_curve(y_test, final_proba)
    
    print(f"Instagram Model Accuracy: {accuracy:.4f}")
    print(f"Instagram Model AUC-ROC: {auc_score:.4f}")
    print("Classification Report:")
    print(report)
    
    # Save model and results
    results_folder = 'results/instagram'
    models_folder = 'models/instagram'
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(models_folder, exist_ok=True)
    model.save(os.path.join(models_folder, 'hybrid_model.h5'))
    joblib.dump(scaler, os.path.join(models_folder, 'scaler.pkl'))
    
    with open(os.path.join(results_folder, 'classification_report.txt'), 'w') as f:
        f.write(f"Instagram Model Accuracy: {accuracy:.4f}\n")
        f.write(f"Instagram Model AUC-ROC: {auc_score:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Save confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_folder, 'confusion_matrix.png'))
    plt.close()
    
    # Save AUC-ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC-ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(results_folder, 'auc_roc_curve.png'))
    plt.close()

# Instagram dataset path
instagram_path = 'datasets/instagram_dataset.csv'  # Update with actual path

# Process Instagram dataset
process_instagram(instagram_path)