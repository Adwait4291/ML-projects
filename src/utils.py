import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle
    
    Args:
        file_path: Path where the object will be saved
        obj: Python object to be saved
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully at: {file_path}")
    except Exception as e:
        logging.error(f"Error in save_object: {e}")
        raise CustomException(e, sys) from e

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluate multiple machine learning models with hyperparameter tuning
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        models: Dictionary of models to evaluate {model_name: model_instance}
        params: Dictionary of parameter grids for each model {model_name: param_grid}
        
    Returns:
        Dictionary containing test R² scores for each model
    """
    try:
        report = {}
        
        for model_name, model in models.items():
            logging.info(f"Training and evaluating {model_name}")
            
            # Get parameter grid for this model
            param_grid = params[model_name]
            
            # If parameters exist, perform GridSearchCV
            if param_grid:
                try:
                    logging.info(f"Performing hyperparameter tuning for {model_name}")
                    gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
                    gs.fit(X_train, y_train)
                    
                    # Set best parameters to the model
                    model.set_params(**gs.best_params_)
                    logging.info(f"Best parameters for {model_name}: {gs.best_params_}")
                except Exception as hyper_e:
                    logging.warning(f"GridSearchCV failed for {model_name}: {hyper_e}")
                    logging.info(f"Training {model_name} with default parameters")
            
            # Train the model (with best params if GridSearchCV was used)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate R² scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            # Store test score in report
            report[model_name] = test_model_score
            
            logging.info(f"{model_name} - Train R²: {train_model_score:.4f}, Test R²: {test_model_score:.4f}")
        
        # Print the final report
        print("\nModel Evaluation Results (Test R² Scores):")
        for model_name, r2 in report.items():
            print(f"{model_name}: {r2:.4f}")

        # Find and print the best model
        best_model_name = max(report, key=report.get)
        best_score = report[best_model_name]
        print(f"\nBest performing model: {best_model_name} with R² score: {best_score:.4f}")
        
        return report
    except Exception as e:
        logging.error(f"Error in evaluating models: {e}")
        raise CustomException(e, sys) from e

def load_object(file_path):
    """
    Load a Python object from a file using pickle
    
    Args:
        file_path: Path to the file containing the object
        
    Returns:
        The loaded Python object
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logging.info(f"Object loaded successfully from: {file_path}")
        return obj
    except Exception as e:
        logging.error(f"Error in load_object: {e}")
        raise CustomException(e, sys) from e

# Example usage
if __name__ == "__main__":
    try:
        # This section would be run when utils.py is executed directly
        # Just a placeholder to show how to use these functions
        print("Utility functions for ML pipeline:")
        print("1. save_object: Save Python objects to disk")
        print("2. load_object: Load Python objects from disk")
        print("3. evaluate_models: Evaluate and compare multiple ML models")
    except Exception as e:
        print(f"Error: {e}")