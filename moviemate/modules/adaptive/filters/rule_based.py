import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

class RuleBasedFiltering:
    """
    A rule-based recommender system.

    This class uses predefined rules and item metadata to recommend items
    and predict ratings based on user preferences and item characteristics.
    """

    def __init__(self, ratings_file, metadata_file):
        """
        Initialize the rule-based recommender system.

        Parameters
        ----------
        ratings_file : str
            Path to the ratings dataset file (user, item, rating).
        metadata_file : str
            Path to the item metadata file (item, features).
        """
        self.ratings_file = ratings_file
        self.metadata_file = metadata_file
        self.ratings = None
        self.items_metadata = None
        self._load_data()
        self._compute_global_statistics()

    def _load_data(self):
        """Load the ratings and item metadata datasets."""
        self.ratings = pd.read_csv(
            self.ratings_file,
            sep='\t',
            names=['user', 'item', 'rating', 'timestamp']
        )
        self.items_metadata = pd.read_csv(
            self.metadata_file,
            sep='|',
            encoding='latin-1',
            names=[
                'item', 'title', 'release_date', 'video_release_date',
                'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
            ]
        )

    def _compute_global_statistics(self):
        """
        Compute global statistics used in rule-based recommendations.
        """
        # Average rating per genre
        self.genre_avg_ratings = {}
        genre_columns = self.items_metadata.columns[6:]
        
        for genre in genre_columns:
            genre_items = self.items_metadata[self.items_metadata[genre] == 1]['item']
            genre_ratings = self.ratings[self.ratings['item'].isin(genre_items)]['rating']
            self.genre_avg_ratings[genre] = genre_ratings.mean() if len(genre_ratings) > 0 else 3.0

        # Global average rating
        self.global_avg_rating = self.ratings['rating'].mean()

        # User rating behavior
        self.user_rating_stats = self.ratings.groupby('user')['rating'].agg(['mean', 'count'])
    
    def predict(self, user_id, item_id):
        """
        Predict the rating for a given user and item using rule-based methods.
        Supports both existing and new users.

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
        # Retrieve item metadata
        try:
            item_metadata = self.items_metadata[self.items_metadata['item'] == item_id].iloc[0]
        except IndexError:
            raise ValueError(f"Item {item_id} not found in the metadata.")

        # Handling new user scenario
        if user_id not in self.ratings['user'].unique():
            # Strategy for new users:
            # 1. Use genre popularity
            # 2. Fall back to global average
            genre_boosts = []
            for genre in self.items_metadata.columns[6:]:
                if item_metadata[genre] == 1:
                    genre_boosts.append(self.genre_avg_ratings[genre])
            
            # If genre found, use genre average, else use global average
            if genre_boosts:
                new_user_rating = np.mean(genre_boosts)
            else:
                new_user_rating = self.global_avg_rating
            
            return max(1, min(5, new_user_rating))

        # Existing user prediction logic
        # Rule 1: Base prediction on user's average rating
        user_avg_rating = self.user_rating_stats.loc[user_id, 'mean']

        # Rule 2: Adjust based on genre popularity
        genre_boosts = []
        for genre in self.items_metadata.columns[6:]:
            if item_metadata[genre] == 1:
                genre_boosts.append(self.genre_avg_ratings[genre])

        # Rule 3: Penalize/boost based on user rating history
        user_rating_count = self.user_rating_stats.loc[user_id, 'count']
        user_rating_diversity_factor = 1 + (user_rating_count / 100)  # Small boost for users with more ratings

        # Combine rules
        predicted_rating = (
            user_avg_rating * 0.5 +  # User's average rating
            (np.mean(genre_boosts) if genre_boosts else self.global_avg_rating) * 0.3 +  # Genre popularity or global average
            self.global_avg_rating * 0.2  # Global average as baseline
        ) * user_rating_diversity_factor

        # Clip rating between 1 and 5
        return max(1, min(5, predicted_rating))

    def evaluate(self, sample_size=1000):
        """
        Evaluate the model by calculating the RMSE on a sample of user-item ratings.

        Parameters
        ----------
        sample_size : int, optional
            Number of random user-item pairs to evaluate. Default is 1000.

        Returns
        -------
        float
            RMSE value.
        """
        sample_ratings = self.ratings.sample(n=sample_size, random_state=42)

        true_ratings = []
        predicted_ratings = []

        for _, row in sample_ratings.iterrows():
            user_id, item_id, true_rating = row['user'], row['item'], row['rating']
            try:
                predicted_rating = self.predict(user_id, item_id)
                true_ratings.append(true_rating)
                predicted_ratings.append(predicted_rating)
            except ValueError:
                continue

        return np.sqrt(mean_squared_error(true_ratings, predicted_ratings))

    def recommend_items(self, user_id, top_n=5):
        """
        Recommend top N items for a given user based on rule-based prediction.

        Parameters
        ----------
        user_id : int
            ID of the user to recommend items for.
        top_n : int, optional
            Number of recommendations to return. Default is 5.

        Returns
        -------
        list
            Top N recommended item IDs.
        """
        # Get items the user hasn't rated
        rated_items = set(self.ratings[self.ratings['user'] == user_id]['item'])
        unrated_items = self.items_metadata[~self.items_metadata['item'].isin(rated_items)]['item']

        # Predict ratings for unrated items
        item_predictions = []
        for item_id in unrated_items:
            try:
                predicted_rating = self.predict(user_id, item_id)
                item_predictions.append((item_id, predicted_rating))
            except ValueError:
                continue

        # Sort and return top N recommendations
        recommendations = sorted(item_predictions, key=lambda x: x[1], reverse=True)[:top_n]
        return [item_id for item_id, _ in recommendations]


if __name__ == "__main__":
    recommender = RuleBasedFiltering(
        ratings_file='storage/u.data',
        metadata_file='storage/u.item'
    )

    rmse = recommender.evaluate(sample_size=100)
    print(f"RMSE on sample: {rmse}")

    user_id = 1  # Example user
    item_id = 242  # Example movie
    predicted_rating = recommender.predict(user_id, item_id)
    print(f"Predicted rating for user {user_id} and item {item_id}: {predicted_rating}")
