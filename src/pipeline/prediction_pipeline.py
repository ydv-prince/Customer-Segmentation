from src.ml.model.s3_estimator import CustomerClusterEstimator
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig, ModelTrainerConfig, Prediction_config, PredictionPipelineConfig
from src.constant.training_pipeline import *
from src.utils.main_utils import MainUtils
from src.exception import CustomerException
import pandas as pd
import numpy as np
import sys


class CustomerData:
    def __init__(self):
        pass

    def get_input_dataset(self, column_schema: dict, input_data):
        columns = column_schema.keys()

        input_dataset = pd.DataFrame([input_data], columns=columns)
        for key,value in column_schema.items():
            input_dataset[key] = input_dataset[key].astype(value)

        return input_dataset

    @staticmethod
    def form_input_dataframe(data):
        prediction_config = Prediction_config()
        prediction_schema = prediction_config.__dict__
        column_schema = prediction_schema["prediction_schema"]["columns"]

        customerData = CustomerData()
        input_dataset = customerData.get_input_dataset(column_schema=column_schema, input_data=data)

        return input_dataset


class PredictionPipeline:
    def __init__(self):
        self.utils = MainUtils

    def prepare_input_data(self, input_data: list) -> pd.DataFrame:
        """
        Method Name: prepare_input_data 
        Objective: This method creates a dataframe taking the column names from prediction schema file with the input values for prediction and returns it
        Args:
            input_data (list): input data 
        Raises:
            CustomerException
        Returns:
            customerDataframe: pd.DataFrame: a dataframe containing the input values
        """

        try:
            customerDataFrame = CustomerData.form_input_dataframe(data=input_data)
            logging.info("CustomerDataFrame has been created")

            return customerDataFrame
            

        except Exception as e:
            raise CustomerException(e, sys) from e

    def get_trained_model(self, ModelTrainerConfig = ModelTrainerConfig):
        """
        Method: get_trained_model
        Objective: this method returns the model
        Args: ModelTrainerConfig
        Raises: CustomerException: 
        Returns:
            Model: Latest Trained Model
        """

        try:
            Prediction_config = PredictionPipelineConfig()
            model = CustomerClusterEstimator(bucket_name=Prediction_config.model_bucket_name, model_path=Prediction_config.model_file_name)

            return model

        except Exception as e:
            raise CustomerException(e, sys) from e


    def run_pipeline(self, input_data: list):
        """
        Method: run_pipeline
        Objective: run_pipeline method runs the whole prediction pipeline.
        Raises:
            CustomerException: 
        """

        try:
            input_dataframe = self.prepare_input_data(input_data)
            model = self.get_trained_model()
            prediction = model.predict(input_dataframe)
            return prediction

        except Exception as e:
            raise CustomerException(e, sys) from e

        