import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

class Pipeline:
    def __init__(self):
        self.ratings_df = None
        self.movies_df = None
    
    def load_data(self, file_path, display_details=True):
        """
        Load MovieLens dataset and inspect its structure.
        
        Args:
            file_path (str): Path to the dataset file
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            self.ratings_df = pd.read_csv(file_path, sep='\t', 
                                          names=['user_id', 'movie_id', 'rating', 'timestamp'])
            if display_details:
                print("Dataset loaded successfully.")
                print("Dataset Shape:", self.ratings_df.shape)
                print("\nDataset Info:")
                print(self.ratings_df.info())
                print("\nFirst few rows:")
                print(self.ratings_df.head())
            return self.ratings_df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def partition_data(self, ratings_df=None, partition_type='stratified', test_size=0.2, save_path=None, data_name=None):
        """
        Split data into training and testing sets.
        
        Args:
            ratings_df (pd.DataFrame): DataFrame to split
            partition_type (str): Type of partitioning 
                - 'stratified': User-stratified sampling
                - 'temporal': Time-based splitting
        
        Returns:
            tuple: Train and test datasets
        """
        if ratings_df is None:
            ratings_df = self.ratings_df
        
        if ratings_df is None:
            raise ValueError("No ratings dataframe provided.")
        
        if partition_type == 'stratified':
            # User-stratified sampling
            unique_users = ratings_df['user_id'].unique()
        
            train_users, test_users = train_test_split(unique_users, test_size=test_size)
            
            train_df = ratings_df[ratings_df['user_id'].isin(train_users)]
            test_df = ratings_df[ratings_df['user_id'].isin(test_users)]
            
        elif partition_type == 'temporal':
            # Time-based splitting
            sorted_df = ratings_df.sort_values('timestamp')
            split_idx = int(len(sorted_df) * (1 - test_size))
            train_df = sorted_df.iloc[:split_idx]
            test_df = sorted_df.iloc[split_idx:]
        else:
            raise ValueError("Invalid partition type.")
        
        if save_path and data_name:
            train_dir = os.path.join(save_path, 'train')
            test_dir = os.path.join(save_path, 'test')
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            
            train_path = os.path.join(train_dir, f'{data_name}')
            test_path = os.path.join(test_dir, f'{data_name}')
            
            train_df.to_csv(train_path, index=False, header=False, sep='\t')
            test_df.to_csv(test_path, index=False, header=False, sep='\t')     
        
        return train_df, test_df


if __name__ == '__main__':
    pipeline = Pipeline()
    ratings_df = pipeline.load_data('storage/u.data')  
    train_df, test_df = pipeline.partition_data(ratings_df, partition_type='temporal', save_path='storage', data_name='v1')
