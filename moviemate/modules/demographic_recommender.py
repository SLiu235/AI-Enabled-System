import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class DemographicRecommender:
    """
    A recommender system based on user demographic information.
    """

    def __init__(self, ratings_file, user_file, item_file):
        """
        Initialize the demographic-based recommender.

        Parameters
        ----------
        ratings_file : str
            Path to the ratings dataset file.
        user_file : str
            Path to the user demographic dataset file.
        item_file : str
            Path to the item metadata dataset file.
        """
        self.ratings_df = pd.read_csv(
            ratings_file, sep='\t', names=['user', 'item', 'rating', 'timestamp']
        )
        self.user_df = pd.read_csv(
            user_file, sep='|', encoding='latin-1',
            names=['user', 'age', 'gender', 'occupation', 'zip_code']
        )
        self.item_df = pd.read_csv(
            item_file, sep='|', encoding='latin-1',
            names=[
                'item', 'title', 'release_date', 'video_release_date',
                'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
            ]
        )
        
        self.ratings_df = self.ratings_df.merge(self.user_df, on='user', how='left')
        self.ratings_df = self.ratings_df.merge(self.item_df, on='item', how='left')

    def fit(self):
        """
        Prepare the demographic model by grouping ratings by demographic factors.
        """
        # Group ratings by demographic factors and compute mean ratings for each group
        self.demographic_profiles = self.ratings_df.groupby(
            ['age', 'gender', 'occupation']
        )['rating'].mean().reset_index()

        self.item_genre_profiles = self.ratings_df.groupby(
            ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
             'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
             'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        )['rating'].mean().reset_index()

    def predict(self, user_id, item_id):
        """
        Predict the rating of a movie for a user based on demographics.

        Parameters
        ----------
        user_id : int
            ID of the user.
        item_id : int
            ID of the item.

        Returns
        -------
        float
            Predicted rating.
        """
        # Retrieve user demographics
        user_data = self.user_df[self.user_df['user'] == user_id]
        if user_data.empty:
            raise ValueError(f"User {user_id} not found in demographic data.")

        age = user_data['age'].values[0]
        gender = user_data['gender'].values[0]
        occupation = user_data['occupation'].values[0]

        # Find average rating for the demographic group
        group = self.demographic_profiles[
            (self.demographic_profiles['age'] == age) &
            (self.demographic_profiles['gender'] == gender) &
            (self.demographic_profiles['occupation'] == occupation)
        ]

        if group.empty:
            item_data = self.item_df[self.item_df['item'] == item_id]
            if item_data.empty:
                return 3 

            # Check genres for the item
            genres = item_data.iloc[0, 5:].values  
            genre_group = self.item_genre_profiles[
                (self.item_genre_profiles.iloc[:, :-1].values == genres).all(axis=1)
            ]

            if not genre_group.empty:
                return genre_group['rating'].values[0]

            return 3 
        
        return group['rating'].values[0]

if __name__ == "__main__":
    # Initialize the demographic recommender
    demographic_recommender = DemographicRecommender(
        ratings_file='storage/u.data',
        user_file='storage/u.user',
        item_file='storage/u.item'
    )

    # Fit the model
    demographic_recommender.fit()

    # Predict the rating for user 1 and item 50
    predicted_rating = demographic_recommender.predict(user_id=3, item_id=50)
    print(f"Predicted rating for user 3 on item 50: {predicted_rating}")
