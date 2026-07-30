import os
import pandas as pd
from typing import Dict, List
from sklearn.model_selection import GroupShuffleSplit

class Dataset_Designer:
    def __init__(self):
        self.raw_data = None
        self.train_data = None
        self.test_data = None

    def extract(self, raw_dataset_filename: str) -> pd.DataFrame:
        current_dir = os.getcwd()
        file_dir = os.path.join(os.path.dirname(current_dir), 'storage/raw_data')
        file_path = os.path.join(file_dir, raw_dataset_filename)
        self.raw_dataset = pd.read_parquet(file_path)

        return self.raw_dataset

    def sample(self) -> List[pd.DataFrame]:
        # Use GroupShuffleSplit to keep all data with the same cc_num together
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        
        # Get the indices for train and test sets
        train_idx, test_idx = next(gss.split(self.raw_dataset, groups=self.raw_dataset['cc_num']))
        
        # Split the data
        self.train_data = self.raw_dataset.iloc[train_idx]
        self.test_data = self.raw_dataset.iloc[test_idx]
        
        return [self.train_data, self.test_data]
    
    def describe(self) -> Dict:
        description = {
            'version': 'v1.0',
            'storage': 'securebank/storage/partitioned_data/',
            'description': {}
        }
        
        description['description']['train'] = {
                'data_type': 'train',
                'shape': self.train_data.shape,
                'fraud_ratio': self.train_data['is_fraud'].mean(),
                'unique_cc_nums': self.train_data['cc_num'].nunique()
        }

        
        description['description']['test'] = {
                'data_type': 'test',
                'shape': self.test_data.shape,
                'fraud_ratio': self.test_data['is_fraud'].mean(),
                'unique_cc_nums': self.test_data['cc_num'].nunique()
        }

        return description
    
    def load(self, output_filename: str) -> None:
        current_dir = os.getcwd()
        save_to_dir = os.path.join(os.path.dirname(current_dir), 'storage/partitioned_data')
        
        self.test_data.to_parquet(f"{save_to_dir}/{output_filename}_test")
        self.train_data.to_parquet(f"{save_to_dir}/{output_filename}_train")
