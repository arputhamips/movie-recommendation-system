"""
Evaluation metrics for recommendation systems
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class RecommenderEvaluator:
    """Evaluate recommendation system performance"""
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """
        Initialize evaluator
        
        Args:
            k_values: List of k values for precision@k and recall@k
        """
        self.k_values = k_values
        
    def evaluate_rating_prediction(self, model, test_df: pd.DataFrame) -> Dict:
        """
        Evaluate rating prediction accuracy
        
        Args:
            model: Trained recommendation model
            test_df: Test DataFrame with true ratings
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating rating prediction...")
        
        predictions = []
        actuals = []
        
        for _, row in test_df.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            actual_rating = row['rating']
            
            # Get prediction
            try:
                pred_rating = model.predict(user_id, item_id)
                predictions.append(pred_rating)
                actuals.append(actual_rating)
            except:
                continue
        
        if not predictions:
            logger.warning("No valid predictions found")
            return {}
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'n_predictions': len(predictions)
        }
        
        logger.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return metrics
    
    def evaluate_ranking(self, model, test_df: pd.DataFrame, 
                        train_df: pd.DataFrame) -> Dict:
        """
        Evaluate ranking metrics (precision, recall, F1)
        
        Args:
            model: Trained recommendation model
            test_df: Test DataFrame
            train_df: Train DataFrame
            
        Returns:
            Dictionary with ranking metrics
        """
        logger.info("Evaluating ranking metrics...")
        
        metrics = {f'precision@{k}': [] for k in self.k_values}
        metrics.update({f'recall@{k}': [] for k in self.k_values})
        metrics.update({f'f1@{k}': [] for k in self.k_values})
        
        # Get unique users in test set
        test_users = test_df['user_id'].unique()
        
        for user_id in test_users:
            # Get items rated highly in test set (relevant items)
            user_test = test_df[test_df['user_id'] == user_id]
            relevant_items = set(user_test[user_test['rating'] >= 4]['item_id'].values)
            
            if len(relevant_items) == 0:
                continue
            
            # Get items seen in training (to exclude)
            user_train = train_df[train_df['user_id'] == user_id]
            seen_items = set(user_train['item_id'].values)
            
            # Get recommendations
            try:
                recommendations = model.recommend(user_id, max(self.k_values), exclude_seen=True)
                recommended_items = list(recommendations['item_id'].values)
                
                # Calculate metrics for each k
                for k in self.k_values:
                    top_k = recommended_items[:k]
                    
                    # Precision@k
                    n_relevant_in_top_k = len(set(top_k) & relevant_items)
                    precision = n_relevant_in_top_k / k if k > 0 else 0
                    metrics[f'precision@{k}'].append(precision)
                    
                    # Recall@k
                    recall = n_relevant_in_top_k / len(relevant_items) if len(relevant_items) > 0 else 0
                    metrics[f'recall@{k}'].append(recall)
                    
                    # F1@k
                    if precision + recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                    else:
                        f1 = 0
                    metrics[f'f1@{k}'].append(f1)
            except:
                continue
        
        # Calculate averages
        result = {}
        for metric_name, values in metrics.items():
            if values:
                result[metric_name] = np.mean(values)
            else:
                result[metric_name] = 0.0
        
        logger.info(f"Precision@10: {result.get('precision@10', 0):.4f}, "
                   f"Recall@10: {result.get('recall@10', 0):.4f}")
        
        return result
    
    def evaluate_diversity(self, model, user_ids: List[int], 
                          n_recommendations: int = 10) -> Dict:
        """
        Evaluate diversity of recommendations
        
        Args:
            model: Trained recommendation model
            user_ids: List of user IDs to evaluate
            n_recommendations: Number of recommendations per user
            
        Returns:
            Dictionary with diversity metrics
        """
        logger.info("Evaluating recommendation diversity...")
        
        all_recommendations = []
        intra_list_similarities = []
        
        for user_id in user_ids:
            try:
                recs = model.recommend(user_id, n_recommendations)
                rec_items = list(recs['item_id'].values)
                all_recommendations.extend(rec_items)
                
                # Calculate intra-list similarity (diversity within a single recommendation list)
                if hasattr(model, 'cb_model') and hasattr(model.cb_model, 'tfidf_matrix'):
                    similarities = []
                    for i in range(len(rec_items)):
                        for j in range(i + 1, len(rec_items)):
                            # Get similarity between items i and j
                            sim = self._get_item_similarity(model, rec_items[i], rec_items[j])
                            if sim is not None:
                                similarities.append(sim)
                    
                    if similarities:
                        avg_similarity = np.mean(similarities)
                        intra_list_similarities.append(1 - avg_similarity)  # Diversity = 1 - similarity
            except:
                continue
        
        # Calculate coverage (what percentage of items are being recommended)
        unique_recommendations = len(set(all_recommendations))
        total_items = len(model.cf_model.user_item_matrix.columns) if hasattr(model, 'cf_model') else 1000
        coverage = unique_recommendations / total_items
        
        # Calculate aggregate diversity (how many different items are recommended across all users)
        aggregate_diversity = unique_recommendations
        
        # Calculate average intra-list diversity
        avg_diversity = np.mean(intra_list_similarities) if intra_list_similarities else 0
        
        metrics = {
            'coverage': coverage,
            'aggregate_diversity': aggregate_diversity,
            'avg_intra_list_diversity': avg_diversity,
            'unique_items_recommended': unique_recommendations
        }
        
        logger.info(f"Coverage: {coverage:.4f}, Aggregate Diversity: {aggregate_diversity}")
        
        return metrics
    
    def evaluate_novelty(self, model, test_df: pd.DataFrame, 
                        train_df: pd.DataFrame, n_recommendations: int = 10) -> Dict:
        """
        Evaluate novelty of recommendations
        
        Args:
            model: Trained recommendation model
            test_df: Test DataFrame
            train_df: Train DataFrame
            n_recommendations: Number of recommendations
            
        Returns:
            Dictionary with novelty metrics
        """
        logger.info("Evaluating recommendation novelty...")
        
        # Calculate item popularity from training data
        item_popularity = train_df['item_id'].value_counts()
        max_popularity = item_popularity.max()
        
        novelty_scores = []
        test_users = test_df['user_id'].unique()
        
        for user_id in test_users:
            try:
                recs = model.recommend(user_id, n_recommendations)
                
                # Calculate novelty for each recommended item
                for item_id in recs['item_id'].values:
                    popularity = item_popularity.get(item_id, 0)
                    # Novelty = -log(popularity / max_popularity)
                    if popularity > 0:
                        novelty = -np.log2(popularity / max_popularity)
                    else:
                        novelty = 10  # High novelty for unseen items
                    novelty_scores.append(novelty)
            except:
                continue
        
        avg_novelty = np.mean(novelty_scores) if novelty_scores else 0
        
        metrics = {
            'avg_novelty': avg_novelty,
            'n_evaluated': len(novelty_scores)
        }
        
        logger.info(f"Average Novelty: {avg_novelty:.4f}")
        
        return metrics
    
    def evaluate_all(self, model, train_df: pd.DataFrame, 
                    test_df: pd.DataFrame, movies_df: pd.DataFrame = None) -> Dict:
        """
        Run all evaluation metrics
        
        Args:
            model: Trained recommendation model
            train_df: Training DataFrame
            test_df: Test DataFrame
            movies_df: Optional movies DataFrame
            
        Returns:
            Dictionary with all metrics
        """
        logger.info("Running complete evaluation...")
        
        results = {}
        
        # Rating prediction metrics
        rating_metrics = self.evaluate_rating_prediction(model, test_df)
        results.update(rating_metrics)
        
        # Ranking metrics
        ranking_metrics = self.evaluate_ranking(model, test_df, train_df)
        results.update(ranking_metrics)
        
        # Diversity metrics
        test_users = test_df['user_id'].unique()[:100]  # Sample for efficiency
        diversity_metrics = self.evaluate_diversity(model, test_users)
        results.update(diversity_metrics)
        
        # Novelty metrics
        novelty_metrics = self.evaluate_novelty(model, test_df, train_df)
        results.update(novelty_metrics)
        
        return results
    
    def _get_item_similarity(self, model, item1: int, item2: int) -> float:
        """Helper function to get similarity between two items"""
        
        # Try to get similarity from content-based model
        if hasattr(model, 'cb_model'):
            cb_model = model.cb_model
            if hasattr(cb_model, 'movies_df') and hasattr(cb_model, 'tfidf_matrix'):
                movies_df = cb_model.movies_df
                
                if item1 in movies_df['item_id'].values and item2 in movies_df['item_id'].values:
                    idx1 = movies_df[movies_df['item_id'] == item1].index[0]
                    idx2 = movies_df[movies_df['item_id'] == item2].index[0]
                    
                    from sklearn.metrics.pairwise import cosine_similarity
                    sim = cosine_similarity(
                        cb_model.tfidf_matrix[idx1],
                        cb_model.tfidf_matrix[idx2]
                    )[0, 0]
                    return sim
        
        return None


class CrossValidator:
    """Cross-validation for recommendation systems"""
    
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        """
        Initialize cross-validator
        
        Args:
            n_folds: Number of folds
            random_state: Random seed
        """
        self.n_folds = n_folds
        self.random_state = random_state
        
    def cross_validate(self, model_class, ratings_df: pd.DataFrame, 
                      movies_df: pd.DataFrame = None, **model_params) -> pd.DataFrame:
        """
        Perform cross-validation
        
        Args:
            model_class: Model class to instantiate
            ratings_df: Ratings DataFrame
            movies_df: Optional movies DataFrame
            **model_params: Parameters for model initialization
            
        Returns:
            DataFrame with cross-validation results
        """
        logger.info(f"Starting {self.n_folds}-fold cross-validation...")
        
        # Shuffle data
        ratings_df = ratings_df.sample(frac=1, random_state=self.random_state)
        
        # Create folds
        fold_size = len(ratings_df) // self.n_folds
        evaluator = RecommenderEvaluator()
        
        fold_results = []
        
        for fold in range(self.n_folds):
            logger.info(f"Processing fold {fold + 1}/{self.n_folds}")
            
            # Split data
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.n_folds - 1 else len(ratings_df)
            
            test_idx = ratings_df.index[test_start:test_end]
            train_idx = ratings_df.index.difference(test_idx)
            
            train_df = ratings_df.loc[train_idx]
            test_df = ratings_df.loc[test_idx]
            
            # Train model
            model = model_class(**model_params)
            
            if movies_df is not None and hasattr(model, 'fit'):
                # For hybrid models
                model.fit(train_df, movies_df)
            else:
                model.fit(train_df)
            
            # Evaluate
            metrics = evaluator.evaluate_all(model, train_df, test_df, movies_df)
            metrics['fold'] = fold + 1
            fold_results.append(metrics)
        
        # Compile results
        results_df = pd.DataFrame(fold_results)
        
        # Add summary statistics
        summary = results_df.describe().loc[['mean', 'std']]
        
        logger.info("Cross-validation complete")
        logger.info(f"Average RMSE: {results_df['rmse'].mean():.4f} ± {results_df['rmse'].std():.4f}")
        
        return results_df
