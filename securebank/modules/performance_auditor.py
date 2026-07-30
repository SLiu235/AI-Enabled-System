from modules.raw_data_handler import Raw_Data_Handler
from modules.dataset_design import Dataset_Designer
from modules.feature_extractor import Feature_Extractor
from modules.pipeline import Pipeline

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class PerformanceAuditor:
    def __init__(self):
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
    
    def audit(self, pipeline, data_version=None):
        if data_version == 'None':
            data_version = None

        # Load and preprocess data
        X, y, _, _  = self.load_data(data_version)
        X_df = pd.DataFrame(X)
        X_dict = X_df.to_dict(orient='records')
        y_pred = pipeline.predict(X_dict)
        # y_pred = [pipeline.predict(row.to_dict()) for _, row in X.iterrows()]
        
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

        # Calculate FPR and FNR
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

        return {
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate
        }

if __name__ == "__main__":
    pipe = Pipeline('logistic_regression')
    auditor = PerformanceAuditor()
    result = auditor.audit(pipe)
    print(result)