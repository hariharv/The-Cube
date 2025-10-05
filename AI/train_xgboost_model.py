#!/usr/bin/env python3
"""
XGBoost Exoplanet Classification Training Script

This script sets up the environment and trains an XGBoost model for exoplanet classification.
It includes automatic package installation and comprehensive model evaluation.
"""

import subprocess
import sys
import os

def install_packages():
    """
    Install required packages for XGBoost model training
    """
    required_packages = [
        'xgboost',
        'scikit-learn',
        'pandas',
        'numpy',
        'joblib',
        'matplotlib',
        'seaborn'
    ]
    
    print("Installing required packages...")
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            return False
    
    print("All packages installed successfully!")
    return True

def run_training():
    """
    Run the XGBoost model training
    """
    print("\n" + "="*50)
    print("Starting XGBoost Exoplanet Classification Training")
    print("="*50)
    
    try:
        # Import after package installation
        from model2 import XGBoostExoplanetModel
        import numpy as np
        from sklearn.model_selection import train_test_split
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Initialize model with optimized parameters
        model_params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        xgb_model = XGBoostExoplanetModel(model_params)
        
        # Load data
        print("\n1. Loading dataset...")
        xgb_model.load_data("remaining_70.csv", "sample_30.csv")
        
        # Preprocess data
        print("\n2. Preprocessing data...")
        X_train_scaled, X_test_scaled = xgb_model.preprocess_data(scale_features=True)
        
        # Split training data for validation
        print("\n3. Creating validation split...")
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_scaled, xgb_model.y_train, 
            test_size=0.2, 
            random_state=42, 
            stratify=xgb_model.y_train
        )
        
        print(f"Training set size: {len(X_train_split)}")
        print(f"Validation set size: {len(X_val_split)}")
        print(f"Test set size: {len(X_test_scaled)}")
        
        # Train model
        print("\n4. Training XGBoost model...")
        xgb_model.train_model(X_train_split, y_train_split, X_val_split, y_val_split, early_stopping_rounds=20)
        
        # Evaluate on validation set
        print("\n5. Evaluating on validation set...")
        val_accuracy, _, _ = xgb_model.evaluate_model(X_val_split, y_val_split)
        
        # Evaluate on test set
        print("\n6. Evaluating on test set...")
        test_accuracy, y_pred, y_pred_proba = xgb_model.evaluate_model(X_test_scaled, xgb_model.y_test)
        
        # Display feature importance
        print("\n7. Analyzing feature importance...")
        feature_importance = xgb_model.get_feature_importance()
        
        # Save model
        print("\n8. Saving model...")
        xgb_model.save_model("exoplanet_xgboost_model.pkl", "feature_scaler.pkl")
        
        # Create visualization (optional)
        try:
            create_visualizations(xgb_model, feature_importance, y_pred_proba, xgb_model.y_test)
        except Exception as e:
            print(f"Visualization creation failed: {e}")
        
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Model saved successfully!")
        print("="*50)
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all packages are installed correctly.")
        return False
    except Exception as e:
        print(f"Training failed: {e}")
        return False

def create_visualizations(model, feature_importance, y_pred_proba, y_true):
    """
    Create and save visualization plots
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import roc_curve, auc
        
        plt.style.use('default')
        
        # Feature importance plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Feature Importance
        features, importance = zip(*feature_importance)
        axes[0, 0].barh(features, importance)
        axes[0, 0].set_title('Feature Importance')
        axes[0, 0].set_xlabel('Importance')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend(loc="lower right")
        
        # 3. Prediction Distribution
        axes[1, 0].hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7, label='Class 0', density=True)
        axes[1, 0].hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7, label='Class 1', density=True)
        axes[1, 0].set_xlabel('Prediction Probability')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Prediction Probability Distribution')
        axes[1, 0].legend()
        
        # 4. Class Distribution
        class_counts = [sum(y_true == 0), sum(y_true == 1)]
        axes[1, 1].pie(class_counts, labels=['Non-Exoplanet', 'Exoplanet'], autopct='%1.1f%%')
        axes[1, 1].set_title('Class Distribution in Test Set')
        
        plt.tight_layout()
        plt.savefig('xgboost_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualization saved as 'xgboost_model_analysis.png'")
        
    except Exception as e:
        print(f"Could not create visualizations: {e}")

def main():
    """
    Main execution function
    """
    print("XGBoost Exoplanet Classification Setup and Training")
    print("="*60)
    
    # Check if we're in the correct directory
    if not (os.path.exists("remaining_70.csv") and os.path.exists("sample_30.csv")):
        print("Error: CSV files not found in current directory!")
        print("Please make sure you're in the AI directory with the data files.")
        return False
    
    # Install packages
    if not install_packages():
        print("Package installation failed. Exiting.")
        return False
    
    # Run training
    if not run_training():
        print("Training failed. Exiting.")
        return False
    
    print("\nTraining completed successfully!")
    print("Files created:")
    print("- model2.py: XGBoost model class")
    print("- exoplanet_xgboost_model.pkl: Trained model")
    print("- feature_scaler.pkl: Feature scaler")
    print("- xgboost_model_analysis.png: Visualization plots")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
