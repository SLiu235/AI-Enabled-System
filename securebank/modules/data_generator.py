from modules.raw_data_handler import Raw_Data_Handler
from modules.dataset_design import Dataset_Designer
from modules.feature_extractor import Feature_Extractor

import pandas as pd
import json
from typing import List, Dict
import numpy as np
from datetime import datetime, timedelta
import os

class DataGenerator:
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

        # Load the data
        self.customers_df = pd.read_csv(customer_path)
        self.transactions_df = pd.read_parquet(transactions_path)
        
        with open(fraud_path, 'r') as f:
            fraud_data = json.load(f)

        if isinstance(fraud_data, dict):  
            fraud_data = [fraud_data]  
        self.fraud_df = pd.DataFrame(fraud_data)

        return [self.customers_df, self.transactions_df, self.fraud_df]
    
    def construct_path(self, source_key: str, version: str) -> str:
        default_path = self.data_sources[source_key]
        if version:
            # Extract the base name and file extension from the default path
            base_name, extension = default_path.rsplit('.', 1)
            # Construct the path using the version
            return f"{base_name}_{version}.{extension}"
        return default_path

    def generate_new_customers(self, num_customers: int) -> pd.DataFrame:
        new_customers = self.customers_df.sample(n=num_customers, replace=True).reset_index(drop=True)
        new_customers['cc_num'] = np.random.randint(1000000000000000, 9999999999999999, size=num_customers)
        return new_customers

    def generate_new_transactions(self, num_transactions: int, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        # Sample from the original transactions with replacement to maintain the feature distributions
        sampled_transactions = self.transactions_df.sample(n=num_transactions, replace=True).reset_index(drop=True)
        
        # Adjust numerical features slightly to avoid exact duplication
        sampled_transactions['amt'] *= np.random.uniform(0.95, 1.05, size=num_transactions)  # Adjust amounts by ±5%
        
        # Adjust geographical features slightly
        sampled_transactions['merch_lat'] += np.random.uniform(-0.001, 0.001, size=num_transactions)
        sampled_transactions['merch_long'] += np.random.uniform(-0.001, 0.001, size=num_transactions)
        
        # Generate new transaction dates uniformly within the given range
        transaction_dates = pd.to_datetime(
            np.random.randint(
                int(start_date.timestamp()), 
                int(end_date.timestamp()), 
                size=num_transactions), unit='s')
        sampled_transactions['trans_date_trans_time'] = transaction_dates
        
        # Update credit card numbers to match those in the new customers dataset
        sampled_transactions['cc_num'] = np.random.choice(self.customers_df['cc_num'], num_transactions)
        
        return sampled_transactions

    def generate_new_fraud_data(self, new_transactions: pd.DataFrame, fraud_ratio: float) -> pd.DataFrame:
        num_fraud = int(len(new_transactions) * fraud_ratio)
        
        # Sample fraudulent transactions
        fraud_transactions = new_transactions.sample(n=num_fraud, random_state=42).copy()
        fraud_transactions['is_fraud'] = 1

        return fraud_transactions

    def generate_and_save_data(self, version: str, num_customers: int = 1000, num_transactions: int = 10000, fraud_ratio: float = 0.005):
        self.load_data()

        # Generate new customers, transactions, and fraud data
        new_customers = self.generate_new_customers(num_customers)
        new_transactions = self.generate_new_transactions(
            num_transactions, 
            datetime.now() - timedelta(days=365), 
            datetime.now()
        )
        new_fraud = self.generate_new_fraud_data(new_transactions, fraud_ratio)

        if not version:
            version = 'v1.1'

        # Save new datasets with versioned filenames
        new_customers.to_csv(f'data_sources/customer_release_{version}.csv', index=False)
        new_transactions.to_parquet(f'data_sources/transactions_release_{version}.parquet')
        new_fraud.to_json(f'data_sources/fraud_release_{version}.json', orient='records', lines=False, indent=4)

if __name__ == "__main__":
    data_gen = DataGenerator()
    data_gen.generate_and_save_data(version='v2.0', num_customers=500, num_transactions=20000, fraud_ratio=0.01)
    data_gen.load_data('v2.0')