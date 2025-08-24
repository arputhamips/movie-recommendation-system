"""
Collaborative Filtering recommendation models
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CollaborativeFilter:
    """Base class for collaborative filtering methods"""
    
    def __init__(self, method: str = 'user_based', k_neighbors: int = 50, 
                 min_common: int = 5):
        """
        Initialize collaborative filter
        
        Args:
            method: 'user_based' or 'item_based'
            k_neighbors: Number of neighbors to consider
            min_common: Minimum common items/users for similarity
        """
        self.method = method
        self.k_neighbors = k_neighbors
        self.min_common = min_common
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user_means = None
        self.item_means = None
        
    def fit(self, ratings_df: pd.DataFrame):
        """
        Fit the collaborative filter model
        
        Args:
            ratings_df: DataFrame with columns [user_id, item_id, rating]
        """
        logger.info(f"Fitting {self.method} collaborative filter...")
        
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            fill_value=0
        )
        
        # Calculate means for predictions
        self.user_means = ratings_df.groupby('user_id')['rating'].mean()
        self.item_means = ratings_df.groupby('item_id')['rating'].mean()
        self.global_mean = ratings_df['rating'].mean()
        
        # Compute similarity matrix
        if self.method == 'user_based':
            self.similarity_matrix = self._compute_user_similarity()
        elif self.method == 'item_based':
            self.similarity_matrix = self._compute_item_similarity()
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        logger.info(f"Model fitted with {self.user_item_matrix.shape[0]} users and {self.user_item_matrix.shape[1]} items")
        
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        if user_id not in self.user_item_matrix.index:
            # Cold start - return global mean
            return self.global_mean
            
        if item_id not in self.user_item_matrix.columns:
            # New item - return user mean
            return self.user_means.get(user_id, self.global_mean)
        
        if self.method == 'user_based':
            return self._predict_user_based(user_id, item_id)
        else:
            return self._predict_item_based(user_id, item_id)
    
    def recommend(self, user_id: int, n_recommendations: int = 10,
                 exclude_seen: bool = True) -> pd.DataFrame:
        """
        Get top-N recommendations for a user
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            exclude_seen: Whether to exclude already rated items
            
        Returns:
            DataFrame with recommended items and predicted ratings
        """
        if user_id not in self.user_item_matrix.index:
            # Cold start - recommend popular items
            logger.warning(f"User {user_id} not found. Returning popular items.")
            return self._recommend_popular(n_recommendations)
        
        # Get all items
        all_items = self.user_item_matrix.columns
        
        # Get user's rated items
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index
        
        # Items to predict
        if exclude_seen:
            items_to_predict = [item for item in all_items if item not in rated_items]
        else:
            items_to_predict = all_items
        
        # Predict ratings for all unseen items
        predictions = []
        for item_id in items_to_predict:
            pred_rating = self.predict(user_id, item_id)
            predictions.append({
                'item_id': item_id,
                'predicted_rating': pred_rating
            })
        
        # Sort and return top N
        recommendations = pd.DataFrame(predictions)
        recommendations = recommendations.sort_values('predicted_rating', ascending=False)
        
        return recommendations.head(n_recommendations)
    
    def _compute_user_similarity(self) -> pd.DataFrame:
        """Compute user-user similarity matrix"""
        logger.info("Computing user similarity matrix...")
        
        # Use cosine similarity
        user_similarity = cosine_similarity(self.user_item_matrix.values)
        
        # Convert to DataFrame
        similarity_df = pd.DataFrame(
            user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        # Set diagonal to 0 (don't compare with self)
        np.fill_diagonal(similarity_df.values, 0)
        
        return similarity_df
    
    def _compute_item_similarity(self) -> pd.DataFrame:
        """Compute item-item similarity matrix"""
        logger.info("Computing item similarity matrix...")
        
        # Use cosine similarity (transpose for items)
        item_similarity = cosine_similarity(self.user_item_matrix.T.values)
        
        # Convert to DataFrame
        similarity_df = pd.DataFrame(
            item_similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        
        # Set diagonal to 0
        np.fill_diagonal(similarity_df.values, 0)
        
        return similarity_df
    
    def _predict_user_based(self, user_id: int, item_id: int) -> float:
        """Predict rating using user-based collaborative filtering"""
        
        # Get similar users who rated this item
        item_ratings = self.user_item_matrix[item_id]
        users_who_rated = item_ratings[item_ratings > 0].index
        
        if len(users_who_rated) == 0:
            return self.user_means.get(user_id, self.global_mean)
        
        # Get similarities for users who rated this item
        similarities = self.similarity_matrix.loc[user_id, users_who_rated]
        
        # Get top-k similar users
        top_k_users = similarities.nlargest(self.k_neighbors).index
        
        if len(top_k_users) == 0:
            return self.user_means.get(user_id, self.global_mean)
        
        # Weighted average of ratings
        numerator = 0
        denominator = 0
        
        for similar_user in top_k_users:
            similarity = similarities[similar_user]
            rating = self.user_item_matrix.loc[similar_user, item_id]
            
            if rating > 0 and similarity > 0:
                # Adjust for user bias
                adjusted_rating = rating - self.user_means.get(similar_user, self.global_mean)
                numerator += similarity * adjusted_rating
                denominator += abs(similarity)
        
        if denominator == 0:
            return self.user_means.get(user_id, self.global_mean)
        
        # Prediction with user bias
        prediction = self.user_means.get(user_id, self.global_mean) + (numerator / denominator)
        
        # Clip to valid rating range
        return np.clip(prediction, 1, 5)
    
    def _predict_item_based(self, user_id: int, item_id: int) -> float:
        """Predict rating using item-based collaborative filtering"""
        
        # Get items rated by this user
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index
        
        if len(rated_items) == 0:
            return self.item_means.get(item_id, self.global_mean)
        
        # Get similarities for rated items
        similarities = self.similarity_matrix.loc[item_id, rated_items]
        
        # Get top-k similar items
        top_k_items = similarities.nlargest(self.k_neighbors).index
        
        if len(top_k_items) == 0:
            return self.item_means.get(item_id, self.global_mean)
        
        # Weighted average of ratings
        numerator = 0
        denominator = 0
        
        for similar_item in top_k_items:
            similarity = similarities[similar_item]
            rating = self.user_item_matrix.loc[user_id, similar_item]
            
            if rating > 0 and similarity > 0:
                numerator += similarity * rating
                denominator += abs(similarity)
        
        if denominator == 0:
            return self.item_means.get(item_id, self.global_mean)
        
        prediction = numerator / denominator
        
        # Clip to valid rating range
        return np.clip(prediction, 1, 5)
    
    def _recommend_popular(self, n_recommendations: int) -> pd.DataFrame:
        """Recommend popular items for cold start"""
        
        # Calculate item popularity
        item_popularity = self.user_item_matrix.mean(axis=0).sort_values(ascending=False)
        
        recommendations = pd.DataFrame({
            'item_id': item_popularity.head(n_recommendations).index,
            'predicted_rating': item_popularity.head(n_recommendations).values
        })
        
        return recommendations


class MatrixFactorization:
    """Matrix Factorization for collaborative filtering"""
    
    def __init__(self, method: str = 'svd', n_factors: int = 50, 
                 random_state: int = 42):
        """
        Initialize matrix factorization model
        
        Args:
            method: 'svd' or 'nmf'
            n_factors: Number of latent factors
            random_state: Random seed
        """
        self.method = method
        self.n_factors = n_factors
        self.random_state = random_state
        self.model = None
        self.user_factors = None
        self.item_factors = None
        self.user_item_matrix = None
        self.user_means = None
        self.global_mean = None
        
    def fit(self, ratings_df: pd.DataFrame):
        """
        Fit the matrix factorization model
        
        Args:
            ratings_df: DataFrame with columns [user_id, item_id, rating]
        """
        logger.info(f"Fitting {self.method} matrix factorization...")
        
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            fill_value=0
        )
        
        # Calculate means
        self.user_means = ratings_df.groupby('user_id')['rating'].mean()
        self.global_mean = ratings_df['rating'].mean()
        
        # Normalize matrix by subtracting user means
        matrix_normalized = self.user_item_matrix.copy()
        for user_id in matrix_normalized.index:
            user_mean = self.user_means.get(user_id, self.global_mean)
            user_ratings = matrix_normalized.loc[user_id]
            # Only normalize non-zero values
            mask = user_ratings > 0
            matrix_normalized.loc[user_id, mask] -= user_mean
        
        # Apply matrix factorization
        if self.method == 'svd':
            self.model = TruncatedSVD(
                n_components=self.n_factors,
                random_state=self.random_state
            )
            self.user_factors = self.model.fit_transform(matrix_normalized)
            self.item_factors = self.model.components_.T
            
        elif self.method == 'nmf':
            # NMF requires non-negative values
            matrix_nonneg = matrix_normalized.copy()
            matrix_nonneg[matrix_nonneg < 0] = 0
            
            self.model = NMF(
                n_components=self.n_factors,
                init='nndsvd',
                random_state=self.random_state,
                max_iter=200
            )
            self.user_factors = self.model.fit_transform(matrix_nonneg)
            self.item_factors = self.model.components_.T
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        logger.info(f"Factorization complete: {self.user_factors.shape[0]} users, "
                   f"{self.item_factors.shape[0]} items, {self.n_factors} factors")
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        if user_id not in self.user_item_matrix.index:
            return self.global_mean
            
        if item_id not in self.user_item_matrix.columns:
            return self.user_means.get(user_id, self.global_mean)
        
        # Get user and item indices
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        item_idx = self.user_item_matrix.columns.get_loc(item_id)
        
        # Compute prediction
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
        # Add back user mean
        prediction += self.user_means.get(user_id, self.global_mean)
        
        # Clip to valid range
        return np.clip(prediction, 1, 5)
    
    def recommend(self, user_id: int, n_recommendations: int = 10,
                 exclude_seen: bool = True) -> pd.DataFrame:
        """
        Get top-N recommendations for a user
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            exclude_seen: Whether to exclude already rated items
            
        Returns:
            DataFrame with recommended items and predicted ratings
        """
        if user_id not in self.user_item_matrix.index:
            logger.warning(f"User {user_id} not found. Returning popular items.")
            return self._recommend_popular(n_recommendations)
        
        # Get user index
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        
        # Compute predictions for all items
        predictions = np.dot(self.user_factors[user_idx], self.item_factors.T)
        predictions += self.user_means.get(user_id, self.global_mean)
        
        # Create DataFrame
        pred_df = pd.DataFrame({
            'item_id': self.user_item_matrix.columns,
            'predicted_rating': predictions
        })
        
        # Exclude seen items if requested
        if exclude_seen:
            user_ratings = self.user_item_matrix.loc[user_id]
            seen_items = user_ratings[user_ratings > 0].index
            pred_df = pred_df[~pred_df['item_id'].isin(seen_items)]
        
        # Sort and return top N
        pred_df = pred_df.sort_values('predicted_rating', ascending=False)
        return pred_df.head(n_recommendations)
    
    def _recommend_popular(self, n_recommendations: int) -> pd.DataFrame:
        """Recommend popular items for cold start"""
        
        item_popularity = self.user_item_matrix.mean(axis=0).sort_values(ascending=False)
        
        recommendations = pd.DataFrame({
            'item_id': item_popularity.head(n_recommendations).index,
            'predicted_rating': item_popularity.head(n_recommendations).values
        })
        
        return recommendations
