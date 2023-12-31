import numpy as np
import pandas as pd
from sensor.entity import config_entity, artifact_entity
from sensor.exception import SensorException
from sensor.logger import logging
from scipy.stats import ks_2samp
from typing import Optional
import os, sys
from sensor.utils import write_yaml_file
from sensor.utils import convert_columns_to_float
from sensor.config import TARGET_COLUMN

class DataValidation:
    def __init__(self, data_validation_config:config_entity.DataValidationConfig, data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()
        except Exception as e:
            raise SensorException(e, sys)
        

    def is_required_columns_exists(self, base_df:pd.DataFrame, current_df:pd.DataFrame, report_key_name:str)->bool:
        try:
            logging.info("Checking If Required Columns Exists")
            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns = []
            for base_column in base_columns:
                if base_column not in current_columns:
                    missing_columns.append(base_column)
            logging.info(f"Columns {missing_columns} does not exist")
            self.validation_error[report_key_name] = missing_columns
            if len(missing_columns)>0:
                return False
            return True
        except Exception as e:
            raise SensorException(e, sys)
        
    def data_drift(self, base_df:pd.DataFrame, current_df:pd.DataFrame, report_key_name:str):
        try:
            logging.info("Checking for data drift")
            drift_report = dict()
            base_columns = base_df.columns
            current_columns = current_df.columns

            for base_column in base_columns:
                base_data, current_data = base_df[base_column], current_df[base_column]
                same_distribution = ks_2samp(base_data, current_data)

                if same_distribution.pvalue > 0.05:
                    drift_report[base_column]={
                        "pvalue" : float(same_distribution.pvalue),
                        "is_same_distribution" : True
                    }
                else:
                    drift_report[base_column]={
                        "pvalue": float(same_distribution.pvalue),
                        "is_same_distribution": False
                    }
            logging.info(f"Drift Report: {drift_report}")
            self.validation_error[report_key_name] = drift_report
        except Exception as e:
            raise SensorException(e, sys)

    def drop_missing_values_columns(self, df:pd.DataFrame, report_key_name:str)->Optional[pd.DataFrame]:
        try:
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isna().sum()/df.shape[0]
            logging.info(f"selecting column name which contains null above to {threshold}")
            drop_column_names = null_report[null_report>threshold].index

            logging.info(f"Columns to drop: {list(drop_column_names)}")
            self.validation_error[report_key_name]=list(drop_column_names)
            df.drop(list(drop_column_names),axis=1,inplace=True)

            if len(df.columns)==0:
                return None
            return df
        except Exception as e:
            raise SensorException(e, sys)


    def initiate_data_validation(self)->artifact_entity.DataValidationArtifact:
        try:
            logging.info("Initiating Data Validation")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            base_df.replace({"na":np.NAN}, inplace=True)

            base_df = self.drop_missing_values_columns(df=base_df, report_key_name="missing_values_within_base_dataset")

            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            train_df = self.drop_missing_values_columns(df=train_df, report_key_name="missing_values_within_train_dataset")
            test_df = self.drop_missing_values_columns(df=test_df, report_key_name="missing_values_within_test_dataset")

            exclude_columns = [TARGET_COLUMN]
            base_df = convert_columns_to_float(base_df, exclude_columns)
            train_df = convert_columns_to_float(train_df, exclude_columns)
            test_df = convert_columns_to_float(test_df, exclude_columns)

            train_df_columns_status = self.is_required_columns_exists(base_df=base_df, current_df=train_df, report_key_name="missing_columns_within_train_dataset")
            test_df_columns_status = self.is_required_columns_exists(base_df=base_df, current_df=test_df, report_key_name="missing_columns_within_test_dataset")

            if train_df_columns_status:
                self.data_drift(base_df=base_df, current_df=train_df, report_key_name="Train data drift")
            if test_df_columns_status:
                self.data_drift(base_df=base_df, current_df=test_df, report_key_name="Test data drift")
            logging.info(f"Writing data validation report in {self.data_validation_config.report_file_path}")
            write_yaml_file(self.data_validation_config.report_file_path, self.validation_error)

            return artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)
        except Exception as e:
            raise SensorException(e, sys)