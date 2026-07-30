from modules.raw_data_handler import Raw_Data_Handler
from modules.dataset_design import Dataset_Designer
from modules.feature_extractor import Feature_Extractor

import joblib
from typing import Dict, List
import numpy as np
import pandas as pd
import json
import os
class Pipeline:
    def __init__(self, version: str = None):
        self.version = version
        if version:
            self.model = self.load_model(version)
        self.history = {}
        self.prepare_data()

    def prepare_data(self):
        raw_data_handler = Raw_Data_Handler()
        raw_data_handler.extract(
            customer_information_filename = "data_sources/customer_release.csv", 
            transaction_filename="data_sources/transactions_release.parquet", 
            fraud_information_filename="data_sources/fraud_release.json")
        raw_data_handler.transform()
        raw_data_handler.load('v1.0')

        dataset_designer = Dataset_Designer()
        dataset_designer.extract('v1.0')
        dataset_designer.sample()
        dataset_designer.load('v1.0')

        feature_extractor = Feature_Extractor()
        feature_extractor.extract('v1.0_train', 'v1.0_test')
        feature_extractor.transform()
        feature_extractor.load('v1.0')

    def load_model(self, version: str):
        model_path = f"storage/models/artifacts/{version}.joblib"
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None

    def predict(self, input_data: Dict) -> bool:
        features = self.preprocess(input_data)

        prediction = self.model.predict(features)[0]

        input_data_key = json.dumps(input_data, sort_keys=True)
        self.history[input_data_key] = bool(prediction)
       
        return bool(prediction)

    def select_model(self, version: str) -> None:
        self.version = version
        self.model = self.load_model(version)

    def get_history(self) -> Dict:
        return self.history
    
    def bulk_predict(self, input_data_list: List[Dict]) -> List[bool]:
        predictions = []
        for input_data in input_data_list:
            prediction = self.predict(input_data)
            predictions.append(prediction)
        return predictions

    def get_model_info(self) -> Dict:
        return {
            "version": self.version,
            # "feature_importance": self.model.feature_importances_
        }

    def preprocess(self, input_data: Dict) -> np.array:
        df = pd.DataFrame([input_data])
        print(df.columns)
        # Extract transaction time
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
  
        # Cyclical time features
        df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
        df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
        df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
        df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
            
        # Transaction amount features
        df['log_amt'] = np.log1p(df['amt'])
        
        # Merchant category features
        df['category'] = pd.Categorical(df['category']).codes
        df['merchant'] = pd.Categorical(df['merchant']).codes
        
        # We can't calculate rapid_transactions for a single transaction
        df['rapid_transactions'] = -1
        
        # We can't calculate distance without customer's location
        df['distance'] = 0
        
        # Select final features to match the model's
        features = ['category', 'merchant', 'merch_lat', 'merch_long', 'hour_sin', 'hour_cos', 
                    'log_amt', 'rapid_transactions', 'distance']
        
        return df[features].values

if __name__ == "__main__":
    # Test Pipeline
    pipeline = Pipeline('random_forest')
    with open('../test.json', 'r') as f:
        sample_input = json.load(f)

    # Predict 
    prediction = pipeline.predict(sample_input)
    print(f"Prediction: {prediction}")

    # Get the history
    history = pipeline.get_history()
    print(f"History: {history}")