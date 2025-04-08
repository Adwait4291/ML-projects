import os
import sys
import pickle
import numpy as np
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
            return pickle.load(file_obj)
    except Exception as e:
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
                logging.info(f"Performing hyperparameter tuning for {model_name}")
                grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
                grid.fit(X_train, y_train)
                
                # Set best parameters to the model
                best_params = grid.best_params_
                model.set_params(**best_params)
                logging.info(f"Best parameters for {model_name}: {best_params}")
            
            # Train the model (with best params if GridSearchCV was used)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate R² scores
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Store test score in report
            report[model_name] = test_r2
            
            logging.info(f"{model_name} - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        
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

# Example usage
if __name__ == "__main__":
    # Import necessary libraries for demo
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    
    # Load sample dataset
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define models to evaluate
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'RandomForest': RandomForestRegressor(),
        'GradientBoosting': GradientBoostingRegressor()
    }
    
    # Define parameter grids for hyperparameter tuning
    params = {
        'LinearRegression': {},  # No hyperparameters to tune
        'Ridge': {'alpha': [0.1, 1.0, 10.0]},
        'Lasso': {'alpha': [0.1, 1.0, 10.0]},
        'RandomForest': {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'GradientBoosting': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    }
    
    # Evaluate models
    results = evaluate_models(X_train, y_train, X_test, y_test, models, params)
    
    # Save the best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    
    save_object(
        file_path="artifacts/model.pkl",
        obj=best_model
    )
    
    logging.info(f"Best model ({best_model_name}) saved to artifacts/model.pkl")