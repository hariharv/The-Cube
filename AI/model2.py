import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os

class XGBoostExoplanetModel:
    """
    XGBoost-based exoplanet classification model
    """
    
    def __init__(self, model_params=None):
        """
        Initialize the XGBoost model with default or custom parameters
        """
        if model_params is None:
            self.model_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'early_stopping_rounds': 10,
                'enable_categorical': False
            }
        else:
            self.model_params = model_params
        
        self.model = xgb.XGBClassifier(**self.model_params)
        self.scaler = StandardScaler()
        self.feature_names = ['period', 'tranDur', 'radE', 'earthFlux', 'eqTemp', 'effTemp', 'radS']
        
    def load_data(self, train_file, test_file):
        """
        Load training and testing data from CSV files
        """
        print("Loading data...")
        
        # Load training data
        train_data = np.loadtxt(train_file, delimiter=",", skiprows=1)
        self.y_train = train_data[:, 0].astype(int)
        self.X_train = train_data[:, 1:]
        
        # Load testing data
        test_data = np.loadtxt(test_file, delimiter=",", skiprows=1)
        self.y_test = test_data[:, 0].astype(int)
        self.X_test = test_data[:, 1:]
        
        print(f"Training data shape: {self.X_train.shape}")
        print(f"Testing data shape: {self.X_test.shape}")
        print(f"Training labels distribution: {np.bincount(self.y_train)}")
        print(f"Testing labels distribution: {np.bincount(self.y_test)}")
        
    def preprocess_data(self, scale_features=True):
        """
        Preprocess the data with optional feature scaling
        """
        if scale_features:
            print("Scaling features...")
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            return self.X_train_scaled, self.X_test_scaled
        else:
            return self.X_train, self.X_test
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=10):
        """
        Train the XGBoost model
        """
        print("Training XGBoost model...")
        
        if X_val is not None and y_val is not None:
            # Use validation set for evaluation during training
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=True
            )
        else:
            self.model.fit(X_train, y_train, verbose=True)
        
        print("Model training completed!")
        
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model performance
        """
        print("Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy, y_pred, y_pred_proba
    
    def get_feature_importance(self):
        """
        Get and display feature importance
        """
        importance = self.model.feature_importances_
        feature_importance = list(zip(self.feature_names, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("\nFeature Importance:")
        for feature, importance in feature_importance:
            print(f"{feature}: {importance:.4f}")
        
        return feature_importance
    
    def save_model(self, model_path="exoplanet_xgboost_model.pkl", scaler_path="feature_scaler.pkl"):
        """
        Save the trained model and scaler
        """
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path="exoplanet_xgboost_model.pkl", scaler_path="feature_scaler.pkl"):
        """
        Load a pre-trained model and scaler
        """
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"Model loaded from {model_path}")
            print(f"Scaler loaded from {scaler_path}")
            return True
        else:
            print("Model or scaler files not found!")
            return False
    
    def predict(self, X):
        """
        Make predictions on new data
        """
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        return predictions, probabilities
    
    def generate_predictions_csv(self, test_file="sample_30.csv", output_file="predicted.csv", threshold=0.5):
        """
        Generate predictions with confidence scores and save to CSV
        
        Args:
            test_file: Path to the test data file
            output_file: Path for the output CSV file
            threshold: Threshold for classification (default 0.5)
        """
        print(f"Generating predictions for {test_file}...")
        
        # Load test data with original format (including labels if present)
        test_data = np.loadtxt(test_file, delimiter=",", skiprows=1)
        
        # Extract features (columns 1 onwards are features)
        X_test = test_data[:, 1:]
        actual_labels = test_data[:, 0].astype(int) if test_data.shape[1] > len(self.feature_names) else None
        
        # Make predictions
        predictions, probabilities = self.predict(X_test)
        
        # Get confidence scores (probability of positive class)
        confidence_scores = probabilities[:, 1]  # Probability of class 1 (exoplanet)
        
        # Create binary predictions based on threshold
        binary_predictions = (confidence_scores >= threshold).astype(int)
        
        # Create output dataframe
        output_data = []
        
        for i in range(len(X_test)):
            row = []
            
            # Add original features
            for j, feature_name in enumerate(self.feature_names):
                row.append(X_test[i][j])
            
            # Add actual label if available
            if actual_labels is not None:
                row.append(actual_labels[i])
            
            # Add prediction results
            row.append(binary_predictions[i])  # Binary prediction (0 or 1)
            row.append(confidence_scores[i])   # Confidence score (probability)
            row.append(1 - confidence_scores[i])  # Confidence for negative class
            
            output_data.append(row)
        
        # Create column names
        columns = self.feature_names.copy()
        if actual_labels is not None:
            columns.append('actual_label')
        columns.extend(['predicted_label', 'confidence_positive', 'confidence_negative'])
        
        # Convert to numpy array and save to CSV
        output_array = np.array(output_data)
        
        # Create header string
        header = ','.join(columns)
        
        # Save to CSV
        np.savetxt(output_file, output_array, delimiter=',', header=header, comments='', fmt='%.6f')
        
        print(f"Predictions saved to {output_file}")
        print(f"Total predictions: {len(output_data)}")
        print(f"Predicted exoplanets (positive): {np.sum(binary_predictions)}")
        print(f"Predicted non-exoplanets (negative): {len(binary_predictions) - np.sum(binary_predictions)}")
        
        if actual_labels is not None:
            accuracy = np.mean(binary_predictions == actual_labels)
            print(f"Accuracy with threshold {threshold}: {accuracy:.4f}")
        
        # Print threshold recommendation
        print(f"\nCurrent threshold: {threshold}")
        print("Threshold guidelines:")
        print("- 0.5: Balanced classification (default)")
        print("- 0.3-0.4: More sensitive (catches more exoplanets, but more false positives)")
        print("- 0.6-0.7: More specific (fewer false positives, but might miss some exoplanets)")
        print(f"- Confidence scores range from 0.0 to 1.0")
        
        return output_file, binary_predictions, confidence_scores

def main():
    """
    Main training script
    """
    # Initialize model
    xgb_model = XGBoostExoplanetModel()
    
    # Load data
    xgb_model.load_data("remaining_70.csv", "sample_30.csv")
    
    # Preprocess data
    X_train_scaled, X_test_scaled = xgb_model.preprocess_data(scale_features=True)
    
    # Split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, xgb_model.y_train, test_size=0.2, random_state=42, stratify=xgb_model.y_train
    )
    
    # Train model with validation
    xgb_model.train_model(X_train_split, y_train_split, X_val_split, y_val_split)
    
    # Evaluate on test set
    accuracy, y_pred, y_pred_proba = xgb_model.evaluate_model(X_test_scaled, xgb_model.y_test)
    
    # Display feature importance
    xgb_model.get_feature_importance()
    
    # Save model
    xgb_model.save_model()
    
    # Generate predictions CSV for visualization
    print("\n" + "="*50)
    print("GENERATING PREDICTIONS CSV")
    print("="*50)
    output_file, predictions, confidence_scores = xgb_model.generate_predictions_csv(
        test_file="sample_30.csv", 
        output_file="predicted.csv", 
        threshold=0.5
    )
    
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
