import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class Feature_Extractor:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.train_feature = None
        self.train_target = None
        self.test_feature = None
        self.test_target = None

    def extract(self, training_dataset_filename: str, testing_dataset_filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        current_dir = os.getcwd()
        file_dir = os.path.join(os.path.dirname(current_dir), 'storage/partitioned_data')

        self.train_data = pd.read_parquet(f"{file_dir}/{training_dataset_filename}")
        self.test_data = pd.read_parquet(f"{file_dir}/{testing_dataset_filename}")

        return [self.train_data, self.test_data]
    
    def transform(self) -> List[pd.DataFrame]:
        def extract_features(df):
            # Sort the dataframe by cc_num and transaction time
            df = df.sort_values(by=['cc_num', 'trans_date_trans_time'])

            # Time-based features
            df['hour'] = df['trans_date_trans_time'].dt.hour
            df['hour_sin'] = np.sin(df['hour'] * (2 * np.pi / 24))
            df['hour_cos'] = np.cos(df['hour'] * (2 * np.pi / 24))
            df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
            df['day_of_week_sin'] = np.sin(df['day_of_week'] * (2 * np.pi / 7))
            df['day_of_week_cos'] = np.cos(df['day_of_week'] * (2 * np.pi / 7))
            
            # Time difference between transactions
            df['time_diff'] = df.groupby('cc_num')['trans_date_trans_time'].diff().dt.total_seconds()
            
            # Feature to capture rapid successive transactions
            df['rapid_transactions'] = df.groupby('cc_num')['time_diff'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            
            # Transaction amount features
            df['log_amt'] = np.log1p(df['amt'])
            
            # Merchant category features
            df['category'] = pd.Categorical(df['category']).codes
            df['merchant'] = pd.Categorical(df['merchant']).codes
            
            # Location-based features
            df['distance'] = self.haversine_distance(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
            
            # Select final features
            features = ['category', 'merchant', 'merch_lat', 'merch_long', 'hour_sin', 'hour_cos', 
                        'log_amt', 'rapid_transactions', 'distance']
            target = 'is_fraud'
            
            return df[features], df[target]

        X_train, y_train = extract_features(self.train_data)
        X_test, y_test = extract_features(self.test_data)

        # Handle missing values
        numerical_features = ['merch_lat', 'merch_long', 'log_amt', 'rapid_transactions', 'distance',
                              'hour_sin', 'hour_cos']
        categorical_features = ['category', 'merchant']

        # Impute missing values in training set
        num_imputer = SimpleImputer(strategy='mean')
        cat_imputer = SimpleImputer(strategy='most_frequent')

        X_train[numerical_features] = num_imputer.fit_transform(X_train[numerical_features])
        X_train[categorical_features] = cat_imputer.fit_transform(X_train[categorical_features])

        # Impute missing values in test set
        X_test[numerical_features] = num_imputer.transform(X_test[numerical_features])
        X_test[categorical_features] = cat_imputer.transform(X_test[categorical_features])

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.train_feature = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        self.train_target = pd.DataFrame(y_train)
        self.test_feature = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        self.test_target = pd.DataFrame(y_test)

        return [self.train_feature, self.train_target, self.test_feature, self.test_target]

 
    def describe(self) -> Dict:
        description = {
            'version': 'v1.0',
            'storage': 'securebank/storage/features/',
            'description': {}
        }
        
        data = [self.train_feature, self.train_target, self.test_feature, self.test_target]
        for i, dataset in enumerate(['train_features', 'train_target', 'test_features', 'test_target']):
            description['description'][dataset] = {
                'shape': data[i].shape,
                'columns': data[i].columns.tolist() if isinstance(data[i], pd.DataFrame) else None,
                'dtypes': data[i].dtypes.to_dict() if isinstance(data[i], pd.DataFrame) else None,
                'null_count': data[i].isnull().sum().to_dict() if isinstance(data[i], pd.DataFrame) else None
            }
        
        return description

    def load(self, output_filename: str) -> None:
        current_dir = os.getcwd()
        save_to_dir = os.path.join(os.path.dirname(current_dir), 'storage/features')

        data = [self.train_feature, self.train_target, self.test_feature, self.test_target]
        for i, dataset in enumerate(['train_features', 'train_target', 'test_features', 'test_target']):
            data[i].to_parquet(f"{save_to_dir}/{output_filename}_{dataset}")

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in kilometers

        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c

        return distance