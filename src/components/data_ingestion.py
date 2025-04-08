import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Using standardized path format with forward slashes
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')
            logging.info(f'Dataset shape: {df.shape}')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved to {self.ingestion_config.raw_data_path}")
            
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info(f"Train set shape: {train_set.shape}, Test set shape: {test_set.shape}")
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info(f"Train data saved to {self.ingestion_config.train_data_path}")
            
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Test data saved to {self.ingestion_config.test_data_path}")
            
            logging.info("Ingestion of the data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise CustomException(e, sys) from e

if __name__ == "__main__":
    try:
        # Step 1: Data Ingestion
        obj = DataIngestion()
        train_data, test_data, raw_data = obj.initiate_data_ingestion()
        
        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)
        
        # Step 3: Model Training
        modeltrainer = ModelTrainer()
        r2_score = modeltrainer.initiate_model_trainer(train_arr, test_arr)
        
        # Print results
        print(f"\nPipeline executed successfully!")
        print(f"Preprocessor saved at: {preprocessor_path}")
        print(f"Model saved at: {ModelTrainerConfig().trained_model_file_path}")
        print(f"Final Model Performance (R² score): {r2_score:.4f}")
    except Exception as e:
        print(f"Error occurred: {e}")

