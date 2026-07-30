from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


class Diversifier:
    """
    A class to re-rank recommendations for increased diversity using Maximal Marginal Relevance (MMR).

    Parameters:
    ----------
    metadata : DataFrame
        Item metadata, must include 'item' and 'features' columns.
    lambda_diversity : float, optional (default=0.5)
        Trade-off parameter between relevance and diversity. 
        0 = full diversity, 1 = full relevance.
    """

    def __init__(self, metadata, lambda_diversity=0.5):
        self.metadata = metadata
        self.lambda_diversity = lambda_diversity
        self.similarity_dict = self._compute_similarity_matrix()

    def _compute_similarity_matrix(self):
        """Computes the item similarity matrix using TF-IDF and cosine similarity."""
        item_features = self.metadata.set_index('item')['features']
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(item_features)

        # Compute cosine similarity between items
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        item_indices = list(self.metadata['item'])

        return {item: sim for item, sim in zip(item_indices, similarity_matrix)}

    def rerank(self, recommendations, top_n=10):
        """
        Re-ranks a list of recommended items to increase diversity.

        Parameters:
        ----------
        recommendations : DataFrame
            A DataFrame with recommended items and scores.
        top_n : int, optional (default=10)
            Number of top recommendations to return after re-ranking.

        Returns:
        -------
        DataFrame
            A re-ranked DataFrame with items and scores.
        """
        selected_items = []
        remaining_items = recommendations['item'].tolist()
        item_indices = list(self.metadata['item'])

        while len(selected_items) < top_n and remaining_items:
            mmr_scores = []
            for candidate in remaining_items:
                relevance = recommendations.loc[
                    recommendations['item'] == candidate, 'score'
                ].values[0]

                diversity = 0 if not selected_items else max(
                    self.similarity_dict[candidate][item_indices.index(sel)]
                    for sel in selected_items
                )

                mmr_score = (
                    self.lambda_diversity * relevance
                    - (1 - self.lambda_diversity) * diversity
                )
                mmr_scores.append((candidate, mmr_score))

            # Select the item with the highest MMR score
            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            best_item = mmr_scores[0][0]
            selected_items.append(best_item)
            remaining_items.remove(best_item)

        # Return the re-ranked recommendations
        reranked_recommendations = recommendations[
            recommendations['item'].isin(selected_items)
        ].reset_index(drop=True)

        return reranked_recommendations.sort_values(by='score', ascending=False)


if __name__ == "__main__":
    import sys
    sys.path.append("..")  # Adds higher directory to Python modules path

    from adaptive.filters.collaborative import CollaborativeFiltering
    from surprise import SVD

    svd_params = {
        'n_factors': 200,
        'n_epochs': 100,
        'lr_all': 0.01,
        'reg_all': 0.1
    }
    model = CollaborativeFiltering(
        algorithm=SVD(**svd_params),
        ratings_file='data/u.data',
        metadata_file='data/u.item'
    )
    model.fit()

    from recommender import Recommender

    recommender = Recommender(model=model)
    rankings = recommender.rank_items(user_id=196, top_n=10)
    print(rankings)

    diversifier = Diversifier(
        metadata=recommender.model.items_metadata,
        lambda_diversity=0.5
    )
    reranked = diversifier.rerank(rankings, top_n=10)
    print(reranked)
