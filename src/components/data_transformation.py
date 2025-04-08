import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehotencoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Separate features and target
            logging.info("Separating features and target variable")
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply preprocessing
            logging.info("Applying preprocessing on training and test datasets")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features and target into arrays for modeling
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )
            logging.info("Preprocessor object saved successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CustomException(e, sys) from e


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


# Runner function for the complete pipeline
def run_pipeline():
    try:
        # Step 1: Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path, raw_path = ingestion.initiate_data_ingestion()
        
        # Step 2: Data Transformation
        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
            train_path, test_path
        )
        
        # Step 3: Model Training (commented out as not provided in the original code)
        # from src.components.model_trainer import ModelTrainer
        # model_trainer = ModelTrainer()
        # r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        # print(f"Model performance (R² score): {r2_score}")
        
        return train_arr, test_arr, preprocessor_path
        
    except Exception as e:
        logging.error(f"Error in pipeline: {e}")
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    try:
        train_arr, test_arr, preprocessor_path = run_pipeline()
        print(f"Pipeline executed successfully!")
        print(f"Preprocessor saved at: {preprocessor_path}")
        print(f"Train array shape: {train_arr.shape}")
        print(f"Test array shape: {test_arr.shape}")
    except Exception as e:
        print(f"Error occurred: {e}")