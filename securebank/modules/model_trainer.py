from modules.raw_data_handler import Raw_Data_Handler
from modules.dataset_design import Dataset_Designer
from modules.feature_extractor import Feature_Extractor

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import pandas as pd
import json
import os

class ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42),
            'svm': SVC(random_state=42)
        }

        self.data_sources = {
            'customers': 'data_sources/customer_release.csv',
            'transactions': 'data_sources/transactions_release.parquet',
            'fraud': 'data_sources/fraud_release.json'
        }
        self.customers_df = None
        self.transactions_df = None
        self.fraud_df = None

    def load_data(self, version: str = None):
        # Create paths using the single version string
        customer_path = self.construct_path('customers', version)
        transactions_path = self.construct_path('transactions', version)
        fraud_path = self.construct_path('fraud', version)

        if not version:
            version = 'v1.1'

        raw_data_handler = Raw_Data_Handler()
        raw_data_handler.extract(
            customer_information_filename = customer_path, 
            transaction_filename=transactions_path, 
            fraud_information_filename=fraud_path)
        raw_data_handler.transform()
        raw_data_handler.load(f'{version}')
        raw_data_handler.describe()

        dataset_designer = Dataset_Designer()
        dataset_designer.extract(f'{version}')
        dataset_designer.sample()
        dataset_designer.load(f'{version}')

        feature_extractor = Feature_Extractor()
        feature_extractor.extract(f'{version}_train', f'{version}_test')
        processed_data = feature_extractor.transform()
        feature_extractor.load(f'{version}')

        return processed_data
    
    def construct_path(self, source_key: str, version: str) -> str:
        default_path = self.data_sources[source_key]
        if version:
            # Extract the base name and file extension from the default path
            base_name, extension = default_path.rsplit('.', 1)
            # Construct the path using the version
            return f"{base_name}_{version}.{extension}"
        return default_path

    def train(self, model_name, data_version=None):
        if data_version == 'None':
            data_version = None

        # Load and preprocess data
        X_train, y_train, X_test, y_test  = self.load_data(data_version)

        results = {}
        model = self.models[model_name]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
            
        results[model_name] = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
            
        # Save the model
        os.makedirs('storage/models/artifacts', exist_ok=True)
        joblib.dump(model, f'storage/models/artifacts/{model_name}.joblib')

        return results

if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.train('random_forest')
    print(results)