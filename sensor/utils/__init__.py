import pandas as pd
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.config import mongo_client
import os, sys

def get_collection_as_dataframe(database_name:str, collection_name:str) -> pd.DataFrame:
    '''
    Description: This function returns aps dataset as pandas.DataFrame
    Params:
    database_name: database name
    collection_name: collection name
    =============================================
    return Pandas dataframe of collection
    '''
    try:
        logging.info(f"Reading data from database: {database_name} and collection: {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"Found columns: {df.columns}")
        if "_id" in df.columns:
            logging.info(f"Dropping column: _id")
            df = df.drop("_id", axis=1)
        return df
    except Exception as e:
        raise SensorException(e, sys)