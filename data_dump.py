import json
import pymongo
import pandas as pd

client = pymongo.MongoClient("mongodb://localhost:27017")

DATA_FILE_PATH = "./aps_failure_training_set1.csv"
DATABASE_NAME = "aps"
COLLECTION_NAME = "sensor"

collection = client[DATABASE_NAME][COLLECTION_NAME]

if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    
    df.reset_index(drop=True, inplace=True)
    json_records = list(json.loads(df.T.to_json()).values())

    collection.insert_many(json_records)
