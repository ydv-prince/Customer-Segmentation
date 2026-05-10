import json
import sys
from typing import Tuple, Union
import pandas as pd
# from evidently.model_profile import Profile
# from evidently.model_profile.sections import DataDriftProfileSection

from scipy.stats import ks_2samp, chi2_contingency
import numpy as np

from pandas import DataFrame

from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig

from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils, write_yaml_file


class DataValidation:
    def __init__(self, 
                 data_ingestion_artifact: DataIngestionArtifact, 
                 data_validation_config: DataValidationConfig ):
        
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        
        self.utils = MainUtils()

        self._schema_config = self.utils.read_schema_config_file()

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomerException(e, sys)

    def validate_schema_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_schema_columns
        Description :   This method validates the schema columns for the particular dataframe 
        
        Output      :   True or False value is returned based on the schema 
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        try:
            
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info("Is required column present[{status}]")

            return status

        except Exception as e:
            raise CustomerException(e, sys) from e

   

    def validate_dataset_schema_columns(self, train_set, test_set) -> Tuple[bool, bool]:
        """
        Method Name :   validate_dataset_schema_columns
        Description :   This method validates the schema for schema columns for both train and test set 
        
        Output      :   True or False value is returned based on the schema 
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info(
            "Entered validate_dataset_schema_columns method of Data_Validation class"
        )

        try:
            logging.info("Validating dataset schema columns")

            train_schema_status = self.validate_schema_columns(train_set)

            logging.info("Validated dataset schema columns on the train set")

            test_schema_status = self.validate_schema_columns(test_set)

            logging.info("Validated dataset schema columns on the test set")

            logging.info("Validated dataset schema columns")

            return train_schema_status, test_schema_status

        except Exception as e:
            raise CustomerException(e, sys) from e

    

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """
        Detect data drift using statistical tests.

        Numerical columns:
            Kolmogorov-Smirnov Test

        Categorical columns:
            Chi-Square Test

        Returns:
            True  -> Drift detected
            False -> No significant drift
        """

        try:
            drift_report = {}

            numeric_columns = reference_df.select_dtypes(include=np.number).columns
            categorical_columns = reference_df.select_dtypes(exclude=np.number).columns

            drift_detected = False

            # -----------------------
            # Numerical Features
            # -----------------------
            for col in numeric_columns:

                statistic, p_value = ks_2samp(
                    reference_df[col].dropna(),
                    current_df[col].dropna()
                )

                drift = p_value < 0.05

                drift_report[col] = {
                    "type": "numeric",
                    "p_value": float(p_value),
                    "drift": drift
                }

                if drift:
                    drift_detected = True

            # -----------------------
            # Categorical Features
            # -----------------------
            for col in categorical_columns:

                reference_counts = reference_df[col].value_counts()
                current_counts = current_df[col].value_counts()

                categories = reference_counts.index.union(current_counts.index)

                observed = np.array([
                    reference_counts.reindex(categories, fill_value=0),
                    current_counts.reindex(categories, fill_value=0)
                ])

                _, p_value, _, _ = chi2_contingency(observed)

                drift = p_value < 0.05

                drift_report[col] = {
                    "type": "categorical",
                    "p_value": float(p_value),
                    "drift": drift
                }

                if drift:
                    drift_detected = True

            # Save report
            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=drift_report
            )

            logging.info(f"Dataset Drift Detected: {drift_detected}")

            return drift_detected

        except Exception as e:
            raise CustomerException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered initiate_data_validation method of Data_Validation class")

        try:
            logging.info("Initiated data validation for the dataset")

            train_df, test_df = (DataValidation.read_data(file_path = self.data_ingestion_artifact.trained_file_path),
                                DataValidation.read_data(file_path = self.data_ingestion_artifact.test_file_path))
            
            
            
            drift = self.detect_dataset_drift(train_df, test_df)

            (
                schema_train_col_status,
                schema_test_col_status,
            ) = self.validate_dataset_schema_columns(train_set=train_df, test_set=test_df)

            logging.info(
                f"Schema train cols status is {schema_train_col_status} and schema test cols status is {schema_test_col_status}"
            )

            logging.info("Validated dataset schema columns")

            if (schema_train_col_status and schema_test_col_status):
                validation_status = True

                if drift:
                    logging.info("Dataset drift detected between train and test split.")

            else:
                validation_status = False
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            print("=" * 80)
            print("Validation Status:", validation_status)
            print("Schema Train:", schema_train_col_status)
            print("Schema Test :", schema_test_col_status)
            print("Drift Status:", drift)
            print("=" * 80)
            
            return data_validation_artifact
        except Exception as e:
            raise CustomerException(e, sys) from e
