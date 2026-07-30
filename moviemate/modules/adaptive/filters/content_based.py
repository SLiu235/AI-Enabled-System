import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import numpy as np


class ContentBasedFiltering:
    """
    A content-based recommender system.

    This class uses item metadata and user ratings to recommend items
    based on their similarity to items the user has interacted with.
    """

    def __init__(self, ratings_file, metadata_file):
        """
        Initialize the content-based recommender system.

        Parameters
        ----------
        ratings_file : str
            Path to the ratings dataset file (user, item, rating).
        metadata_file : str
            Path to the item metadata file (item, features).
        """
        self.ratings_file = ratings_file
        self.metadata_file = metadata_file
        self.item_profiles = None
        self.user_profiles = None
        self.similarity_matrix = None
        self.ratings = None
        self.items_metadata = None
        self._load_data()
        self._build_item_profiles()

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

        # Combine genre columns into a single 'features' column
        self.items_metadata['features'] = self.items_metadata.iloc[:, 6:].apply(
            lambda x: ' '.join([
                col for col in self.items_metadata.columns[6:] if x[col] == 1
            ]),
            axis=1
        )

    def _build_item_profiles(self):
        """Create item profiles based on item features using TF-IDF."""
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(self.items_metadata['features'])
        self.item_profiles = pd.DataFrame(
            tfidf_matrix.toarray(),
            index=self.items_metadata['item']
        )
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def _get_user_profile(self, user_id):
        """Create a user profile based on the user's past ratings."""
        user_ratings = self.ratings[self.ratings['user'] == user_id]
        user_profile = np.zeros(self.item_profiles.shape[1])

        for _, row in user_ratings.iterrows():
            item_vector = self.item_profiles.loc[row['item']].values
            user_profile += item_vector * row['rating']

        return user_profile / np.linalg.norm(user_profile)

    def predict(self, user_id, item_id):
        """
        Predict the rating for a given user and item using content-based similarity.
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
        # Check if item exists in profiles
        if item_id not in self.item_profiles.index:
            raise ValueError(f"Item {item_id} not found in the metadata.")

        # Handling new user scenario
        if user_id not in self.ratings['user'].unique():
            # Strategy for new users:
            # 1. Create a default user profile based on item metadata
            # Find items with similar features to the target item
            item_vector = self.item_profiles.loc[item_id].values
            
            # Compute similarity of the target item with all items
            item_similarities = cosine_similarity(
                item_vector.reshape(1, -1), 
                self.item_profiles.values
            )[0]
            
            # Find top 5 most similar items
            top_similar_indices = item_similarities.argsort()[-5:][::-1]
            similar_items = self.item_profiles.index[top_similar_indices]
            
            # Create a default user profile as an average of similar items
            default_user_profile = np.mean(
                self.item_profiles.loc[similar_items].values, 
                axis=0
            )
            
            # Normalize the default user profile
            default_user_profile /= np.linalg.norm(default_user_profile)
            
            # Compute similarity between default profile and target item
            epsilon = 1e-10
            return np.dot(default_user_profile, item_vector) / (
                np.linalg.norm(default_user_profile) * np.linalg.norm(item_vector) + epsilon
            )

        # Existing user prediction logic
        user_profile = self._get_user_profile(user_id)
        item_vector = self.item_profiles.loc[item_id].values

        epsilon = 1e-10
        return np.dot(user_profile, item_vector) / (
            np.linalg.norm(user_profile) * np.linalg.norm(item_vector) + epsilon
        )

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
                predicted_rating = 5 * self.predict(user_id, item_id)
                true_ratings.append(true_rating)
                predicted_ratings.append(predicted_rating)
            except ValueError:
                continue

        return np.sqrt(mean_squared_error(true_ratings, predicted_ratings))


if __name__ == "__main__":
    recommender = ContentBasedFiltering(
        ratings_file='storage/u.data',
        metadata_file='storage/u.item'
    )

    rmse = recommender.evaluate(sample_size=100)
    print(f"RMSE on sample: {rmse}")

    user_id = 1  # Example user
    item_id = 242  # Example movie
    predicted_rating = recommender.predict(user_id, item_id)
    print(f"Predicted rating for user {user_id} and item {item_id}: {predicted_rating}")
