import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

print("Current working directory:", os.getcwd())

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Step 1: Read the dataset
            logging.info("Reading the dataset from 'notebook/data/stud.csv'")
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info(f"Dataset shape: {df.shape}")
            logging.info("Read the dataset as dataframe")
            
            # Step 2: Create the artifacts directory
            artifacts_dir = os.path.dirname(self.ingestion_config.train_data_path)
            logging.info(f"Creating directory: {artifacts_dir}")
            os.makedirs(artifacts_dir, exist_ok=True)
            
            # Step 3: Save raw data
            logging.info(f"Saving raw data to {self.ingestion_config.raw_data_path}")
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # Step 4: Train-test split
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info(f"Train set shape: {train_set.shape}, Test set shape: {test_set.shape}")
            
            # Step 5: Save train and test data
            logging.info(f"Saving train data to {self.ingestion_config.train_data_path}")
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            
            logging.info(f"Saving test data to {self.ingestion_config.test_data_path}")
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")
            
            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path,
                    self.ingestion_config.raw_data_path)
        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise CustomException(e, sys) from e

# This is the main executable part
if __name__ == "__main__":
    try:
        # Create a DataIngestion object
        obj = DataIngestion()
        
        # Call the data ingestion method
        train_path, test_path, raw_path = obj.initiate_data_ingestion()
        
        # Initialize data transformation after ingestion is complete
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)
        
        print(f"Data ingestion completed successfully!")
        print(f"Raw data saved to: {raw_path}")
        print(f"Train data saved to: {train_path}")
        print(f"Test data saved to: {test_path}")
        print(f"Data transformation completed successfully!")
    except Exception as e:
        print(f"Error occurred: {e}")