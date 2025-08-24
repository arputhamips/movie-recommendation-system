"""
Hybrid recommendation model combining multiple approaches
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

from .collaborative import CollaborativeFilter, MatrixFactorization
from .content_based import ContentBasedFilter

logger = logging.getLogger(__name__)


class HybridRecommender:
    """Hybrid recommender combining collaborative and content-based filtering"""
    
    def __init__(self, cf_weight: float = 0.7, cb_weight: float = 0.3,
                 cf_method: str = 'user_based', mf_method: str = 'svd'):
        """
        Initialize hybrid recommender
        
        Args:
            cf_weight: Weight for collaborative filtering
            cb_weight: Weight for content-based filtering
            cf_method: Method for collaborative filtering
            mf_method: Method for matrix factorization
        """
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        
        # Initialize models
        self.cf_model = CollaborativeFilter(method=cf_method)
        self.mf_model = MatrixFactorization(method=mf_method)
        self.cb_model = ContentBasedFilter()
        
        self.is_fitted = False
        
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """
        Fit all component models
        
        Args:
            ratings_df: DataFrame with ratings
            movies_df: DataFrame with movie features
        """
        logger.info("Fitting hybrid recommender...")
        
        # Fit collaborative filtering
        logger.info("Fitting collaborative filtering model...")
        self.cf_model.fit(ratings_df)
        
        # Fit matrix factorization
        logger.info("Fitting matrix factorization model...")
        self.mf_model.fit(ratings_df)
        
        # Fit content-based filtering
        logger.info("Fitting content-based model...")
        self.cb_model.fit(movies_df, ratings_df)
        
        self.is_fitted = True
        logger.info("Hybrid recommender fitted successfully")
        
    def recommend(self, user_id: int, n_recommendations: int = 10,
                 method: str = 'weighted', exclude_seen: bool = True) -> pd.DataFrame:
        """
        Get hybrid recommendations
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            method: 'weighted', 'switching', or 'mixed'
            exclude_seen: Whether to exclude seen items
            
        Returns:
            DataFrame with recommendations
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if method == 'weighted':
            return self._weighted_hybrid(user_id, n_recommendations, exclude_seen)
        elif method == 'switching':
            return self._switching_hybrid(user_id, n_recommendations, exclude_seen)
        elif method == 'mixed':
            return self._mixed_hybrid(user_id, n_recommendations, exclude_seen)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _weighted_hybrid(self, user_id: int, n_recommendations: int,
                        exclude_seen: bool) -> pd.DataFrame:
        """Weighted average of different models"""
        
        # Get recommendations from each model
        cf_recs = self.cf_model.recommend(user_id, n_recommendations * 3, exclude_seen)
        mf_recs = self.mf_model.recommend(user_id, n_recommendations * 3, exclude_seen)
        cb_recs = self.cb_model.recommend_for_user(user_id, n_recommendations * 3, exclude_seen)
        
        # Combine recommendations
        all_recs = {}
        
        # Add CF recommendations
        for _, row in cf_recs.iterrows():
            item_id = row['item_id']
            score = row['predicted_rating'] * self.cf_weight * 0.5
            all_recs[item_id] = all_recs.get(item_id, 0) + score
        
        # Add MF recommendations
        for _, row in mf_recs.iterrows():
            item_id = row['item_id']
            score = row['predicted_rating'] * self.cf_weight * 0.5
            all_recs[item_id] = all_recs.get(item_id, 0) + score
        
        # Add CB recommendations
        for _, row in cb_recs.iterrows():
            item_id = row['item_id']
            score = row['predicted_rating'] * self.cb_weight
            all_recs[item_id] = all_recs.get(item_id, 0) + score
        
        # Sort and return top N
        recommendations = pd.DataFrame(
            [(item_id, score) for item_id, score in all_recs.items()],
            columns=['item_id', 'predicted_rating']
        )
        
        recommendations = recommendations.sort_values('predicted_rating', ascending=False)
        return recommendations.head(n_recommendations)
    
    def _switching_hybrid(self, user_id: int, n_recommendations: int,
                         exclude_seen: bool) -> pd.DataFrame:
        """Switch between models based on data availability"""
        
        # Check if user has enough ratings for CF
        if user_id in self.cf_model.user_item_matrix.index:
            user_ratings = self.cf_model.user_item_matrix.loc[user_id]
            n_ratings = (user_ratings > 0).sum()
            
            if n_ratings >= 10:
                # Use collaborative filtering for users with enough data
                logger.info(f"Using CF for user {user_id} ({n_ratings} ratings)")
                return self.mf_model.recommend(user_id, n_recommendations, exclude_seen)
            else:
                # Use content-based for users with few ratings
                logger.info(f"Using CB for user {user_id} ({n_ratings} ratings)")
                return self.cb_model.recommend_for_user(user_id, n_recommendations, exclude_seen)
        else:
            # Cold start - use content-based
            logger.info(f"Cold start for user {user_id} - using CB")
            return self.cb_model.recommend_for_user(user_id, n_recommendations, exclude_seen)
    
    def _mixed_hybrid(self, user_id: int, n_recommendations: int,
                     exclude_seen: bool) -> pd.DataFrame:
        """Mix recommendations from different models"""
        
        # Get equal number from each model
        n_per_model = n_recommendations // 3
        remainder = n_recommendations % 3
        
        # Get recommendations from each model
        cf_recs = self.cf_model.recommend(user_id, n_per_model + (1 if remainder > 0 else 0), exclude_seen)
        mf_recs = self.mf_model.recommend(user_id, n_per_model + (1 if remainder > 1 else 0), exclude_seen)
        cb_recs = self.cb_model.recommend_for_user(user_id, n_per_model, exclude_seen)
        
        # Combine and remove duplicates
        all_recs = pd.concat([cf_recs, mf_recs, cb_recs])
        all_recs = all_recs.drop_duplicates(subset=['item_id'], keep='first')
        
        # Return top N
        return all_recs.head(n_recommendations)
    
    def explain_recommendation(self, user_id: int, item_id: int) -> Dict:
        """
        Explain why an item was recommended
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Dictionary with explanation details
        """
        explanation = {
            'user_id': user_id,
            'item_id': item_id,
            'reasons': []
        }
        
        # CF explanation
        cf_pred = self.cf_model.predict(user_id, item_id)
        explanation['cf_score'] = cf_pred
        
        if cf_pred > 3.5:
            explanation['reasons'].append(f"Users similar to you rated this highly ({cf_pred:.2f}/5)")
        
        # Content-based explanation
        if user_id in self.cb_model.user_profiles:
            # Find similar movies the user liked
            user_ratings = self.cf_model.user_item_matrix.loc[user_id]
            liked_movies = user_ratings[user_ratings >= 4].index
            
            for liked_movie in liked_movies[:3]:
                similar_movies = self.cb_model.get_movie_similarity(liked_movie, 20)
                if item_id in similar_movies['item_id'].values:
                    similarity = similar_movies[similar_movies['item_id'] == item_id]['similarity_score'].values[0]
                    explanation['reasons'].append(
                        f"Similar to Movie {liked_movie} that you liked (similarity: {similarity:.2f})"
                    )
                    break
        
        return explanation


class EnsembleRecommender:
    """Ensemble of multiple recommendation models with voting"""
    
    def __init__(self, models: List = None):
        """
        Initialize ensemble recommender
        
        Args:
            models: List of recommendation models
        """
        if models is None:
            # Default ensemble
            self.models = [
                ('cf_user', CollaborativeFilter(method='user_based')),
                ('cf_item', CollaborativeFilter(method='item_based')),
                ('mf_svd', MatrixFactorization(method='svd')),
                ('mf_nmf', MatrixFactorization(method='nmf')),
                ('content', ContentBasedFilter())
            ]
        else:
            self.models = models
        
        self.is_fitted = False
        
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """
        Fit all models in the ensemble
        
        Args:
            ratings_df: DataFrame with ratings
            movies_df: DataFrame with movie features
        """
        logger.info("Fitting ensemble recommender...")
        
        for name, model in self.models:
            logger.info(f"Fitting {name}...")
            
            if isinstance(model, ContentBasedFilter):
                model.fit(movies_df, ratings_df)
            else:
                model.fit(ratings_df)
        
        self.is_fitted = True
        logger.info("Ensemble fitted successfully")
        
    def recommend(self, user_id: int, n_recommendations: int = 10,
                 voting: str = 'rank', exclude_seen: bool = True) -> pd.DataFrame:
        """
        Get ensemble recommendations
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            voting: 'rank' or 'score' voting
            exclude_seen: Whether to exclude seen items
            
        Returns:
            DataFrame with recommendations
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before making recommendations")
        
        if voting == 'rank':
            return self._rank_voting(user_id, n_recommendations, exclude_seen)
        elif voting == 'score':
            return self._score_voting(user_id, n_recommendations, exclude_seen)
        else:
            raise ValueError(f"Unknown voting method: {voting}")
    
    def _rank_voting(self, user_id: int, n_recommendations: int,
                    exclude_seen: bool) -> pd.DataFrame:
        """Recommendations based on rank voting"""
        
        item_scores = {}
        
        for name, model in self.models:
            # Get recommendations from model
            if isinstance(model, ContentBasedFilter):
                recs = model.recommend_for_user(user_id, n_recommendations * 2, exclude_seen)
            else:
                recs = model.recommend(user_id, n_recommendations * 2, exclude_seen)
            
            # Add rank scores (higher rank = higher score)
            for i, (_, row) in enumerate(recs.iterrows()):
                item_id = row['item_id']
                rank_score = len(recs) - i
                
                if item_id not in item_scores:
                    item_scores[item_id] = 0
                item_scores[item_id] += rank_score
        
        # Sort by total rank score
        recommendations = pd.DataFrame(
            [(item_id, score) for item_id, score in item_scores.items()],
            columns=['item_id', 'predicted_rating']
        )
        
        recommendations = recommendations.sort_values('predicted_rating', ascending=False)
        return recommendations.head(n_recommendations)
    
    def _score_voting(self, user_id: int, n_recommendations: int,
                     exclude_seen: bool) -> pd.DataFrame:
        """Recommendations based on score voting"""
        
        item_scores = {}
        item_counts = {}
        
        for name, model in self.models:
            # Get recommendations from model
            if isinstance(model, ContentBasedFilter):
                recs = model.recommend_for_user(user_id, n_recommendations * 2, exclude_seen)
            else:
                recs = model.recommend(user_id, n_recommendations * 2, exclude_seen)
            
            # Add scores
            for _, row in recs.iterrows():
                item_id = row['item_id']
                score = row['predicted_rating']
                
                if item_id not in item_scores:
                    item_scores[item_id] = 0
                    item_counts[item_id] = 0
                
                item_scores[item_id] += score
                item_counts[item_id] += 1
        
        # Calculate average scores
        recommendations = []
        for item_id in item_scores:
            avg_score = item_scores[item_id] / item_counts[item_id]
            recommendations.append({
                'item_id': item_id,
                'predicted_rating': avg_score,
                'vote_count': item_counts[item_id]
            })
        
        recommendations = pd.DataFrame(recommendations)
        
        # Sort by score and vote count
        recommendations = recommendations.sort_values(
            ['predicted_rating', 'vote_count'],
            ascending=[False, False]
        )
        
        return recommendations[['item_id', 'predicted_rating']].head(n_recommendations)
