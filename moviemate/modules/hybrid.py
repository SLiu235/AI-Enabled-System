import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridContentProfileRecommender:
    """
    A hybrid recommender system that combines both demographic and content-based filtering methods.

    This class uses both user demographic information and item metadata to predict ratings for items.
    """

    def __init__(self, ratings_file, user_file, item_file):
        """
        Initialize the hybrid recommender system.

        Parameters
        ----------
        ratings_file : str
            Path to the ratings dataset file (user, item, rating).
        user_file : str
            Path to the user demographic dataset file (user id, age, gender, occupation, zip code).
        item_file : str
            Path to the item metadata dataset file (item id, title, genres, etc.).
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
        
        self.item_profiles = None
        self.similarity_matrix = None
        self.demographic_profiles = self.ratings_df.groupby(
            ['age', 'gender', 'occupation']
        )['rating'].mean().reset_index()

        self._build_item_profiles()

    def _build_item_profiles(self):
        """Create item profiles based on item genres using TF-IDF."""
        self.item_df['features'] = self.item_df.iloc[:, 5:].apply(
            lambda x: ' '.join([col for col in self.item_df.columns[5:] if x[col] == 1]), axis=1
        )
        
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(self.item_df['features'])
        self.item_profiles = pd.DataFrame(
            tfidf_matrix.toarray(),
            index=self.item_df['item']
        )
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def _get_user_profile(self, user_id):
        user_ratings = self.ratings_df[self.ratings_df['user'] == user_id]
        
        # New User
        if user_ratings.empty:
            default_profile = np.mean(self.item_profiles.values, axis=0)
            return default_profile / np.linalg.norm(default_profile)
        
        # Existing User
        user_profile = np.zeros(self.item_profiles.shape[1])
        
        for _, row in user_ratings.iterrows():
            item_vector = self.item_profiles.loc[row['item']].values
            user_profile += item_vector * row['rating']

        return user_profile / np.linalg.norm(user_profile)

    def predict(self, user_id, item_id):
        """
        Predict the rating for a given user and item using the hybrid approach.

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
        demographic_group = self.demographic_profiles[
            (self.demographic_profiles['age'] == age) &
            (self.demographic_profiles['gender'] == gender) &
            (self.demographic_profiles['occupation'] == occupation)
        ]
        
        if demographic_group.empty:
            demographic_prediction = 3  # Default 
        else:
            demographic_prediction = demographic_group['rating'].values[0]

        # Content-based prediction
        if item_id not in self.item_profiles.index:
            raise ValueError(f"Item {item_id} not found in the item profiles.")

        # Create user profile
        user_profile = self._get_user_profile(user_id)
        item_vector = self.item_profiles.loc[item_id].values

        content_prediction = np.dot(user_profile, item_vector) / (
            np.linalg.norm(user_profile) * np.linalg.norm(item_vector) 
        )

        # Combine demographic and content-based predictions
        hybrid_prediction = 0.8 * demographic_prediction + 0.2 * content_prediction

        return hybrid_prediction

if __name__ == "__main__":
    ratings_file = 'storage/train/stratified'
    user_file = 'storage/u.user'
    item_file = 'storage/u.item'

    recommender = HybridContentProfileRecommender(
        ratings_file, 
        user_file, 
        item_file
    )
    
    user_id = 3 
    movie_id = 50  
    predicted_rating = recommender.predict(user_id, movie_id)
    print(f"Predicted Rating for User {user_id} and Movie {movie_id}: {predicted_rating:.2f}")

