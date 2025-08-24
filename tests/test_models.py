"""
Unit tests for recommendation models
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.data.data_loader import MovieLensLoader, DataPreprocessor
from src.models.collaborative import CollaborativeFilter, MatrixFactorization
from src.models.content_based import ContentBasedFilter
from src.models.hybrid import HybridRecommender
from src.evaluation.metrics import RecommenderEvaluator


class TestDataLoader(unittest.TestCase):
    """Test data loading functionality"""
    
    def setUp(self):
        self.loader = MovieLensLoader('data/raw/')
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation"""
        ratings = self.loader._generate_synthetic_ratings(
            n_users=10, n_items=20, n_ratings=100
        )
        
        self.assertEqual(len(ratings), 100)
        self.assertIn('user_id', ratings.columns)
        self.assertIn('item_id', ratings.columns)
        self.assertIn('rating', ratings.columns)
        self.assertTrue((ratings['rating'] >= 1).all())
        self.assertTrue((ratings['rating'] <= 5).all())
    
    def test_train_test_split(self):
        """Test train/test splitting"""
        ratings = self.loader.load_ratings(min_ratings=2)
        train_df, test_df = self.loader.create_train_test_split(test_size=0.2)
        
        # Check sizes
        total_size = len(train_df) + len(test_df)
        self.assertAlmostEqual(len(test_df) / total_size, 0.2, places=1)
        
        # Check no overlap
        train_pairs = set(zip(train_df['user_id'], train_df['item_id']))
        test_pairs = set(zip(test_df['user_id'], test_df['item_id']))
        self.assertEqual(len(train_pairs & test_pairs), 0)
    
    def test_user_item_matrix(self):
        """Test user-item matrix creation"""
        ratings = self.loader.load_ratings(min_ratings=2)
        matrix = self.loader.create_user_item_matrix(ratings)
        
        self.assertIsInstance(matrix, pd.DataFrame)
        self.assertEqual(len(matrix), ratings['user_id'].nunique())
        self.assertEqual(len(matrix.columns), ratings['item_id'].nunique())


class TestCollaborativeFilter(unittest.TestCase):
    """Test collaborative filtering models"""
    
    def setUp(self):
        # Create small synthetic dataset
        self.ratings_df = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'item_id': [1, 2, 3, 1, 2, 4, 2, 3, 4],
            'rating':  [5, 4, 3, 4, 5, 2, 3, 5, 4],
            'timestamp': [1] * 9
        })
    
    def test_user_based_cf(self):
        """Test user-based collaborative filtering"""
        model = CollaborativeFilter(method='user_based', k_neighbors=2)
        model.fit(self.ratings_df)
        
        # Test prediction
        prediction = model.predict(user_id=1, item_id=4)
        self.assertIsInstance(prediction, float)
        self.assertTrue(1 <= prediction <= 5)
        
        # Test recommendation
        recommendations = model.recommend(user_id=1, n_recommendations=2)
        self.assertEqual(len(recommendations), 1)  # Only 1 unseen item
        self.assertIn('item_id', recommendations.columns)
        self.assertIn('predicted_rating', recommendations.columns)
    
    def test_item_based_cf(self):
        """Test item-based collaborative filtering"""
        model = CollaborativeFilter(method='item_based', k_neighbors=2)
        model.fit(self.ratings_df)
        
        # Test prediction
        prediction = model.predict(user_id=1, item_id=4)
        self.assertIsInstance(prediction, float)
        self.assertTrue(1 <= prediction <= 5)
        
        # Test recommendation
        recommendations = model.recommend(user_id=2, n_recommendations=2)
        self.assertGreater(len(recommendations), 0)
    
    def test_cold_start(self):
        """Test handling of cold start users"""
        model = CollaborativeFilter(method='user_based')
        model.fit(self.ratings_df)
        
        # New user
        prediction = model.predict(user_id=999, item_id=1)
        self.assertIsInstance(prediction, float)
        
        recommendations = model.recommend(user_id=999, n_recommendations=3)
        self.assertGreater(len(recommendations), 0)


class TestMatrixFactorization(unittest.TestCase):
    """Test matrix factorization models"""
    
    def setUp(self):
        # Create larger synthetic dataset for MF
        np.random.seed(42)
        n_users, n_items = 20, 30
        n_ratings = 200
        
        users = np.random.choice(range(1, n_users + 1), n_ratings)
        items = np.random.choice(range(1, n_items + 1), n_ratings)
        ratings = np.random.choice([1, 2, 3, 4, 5], n_ratings)
        
        self.ratings_df = pd.DataFrame({
            'user_id': users,
            'item_id': items,
            'rating': ratings,
            'timestamp': [1] * n_ratings
        })
    
    def test_svd(self):
        """Test SVD matrix factorization"""
        model = MatrixFactorization(method='svd', n_factors=10)
        model.fit(self.ratings_df)
        
        # Check factorization dimensions
        n_users = self.ratings_df['user_id'].nunique()
        n_items = self.ratings_df['item_id'].nunique()
        
        self.assertEqual(model.user_factors.shape, (n_users, 10))
        self.assertEqual(model.item_factors.shape, (n_items, 10))
        
        # Test prediction
        prediction = model.predict(user_id=1, item_id=1)
        self.assertIsInstance(prediction, float)
        self.assertTrue(1 <= prediction <= 5)
    
    def test_nmf(self):
        """Test NMF matrix factorization"""
        model = MatrixFactorization(method='nmf', n_factors=10)
        model.fit(self.ratings_df)
        
        # Check non-negativity
        self.assertTrue((model.user_factors >= 0).all())
        self.assertTrue((model.item_factors >= 0).all())
        
        # Test recommendation
        recommendations = model.recommend(user_id=1, n_recommendations=5)
        self.assertGreater(len(recommendations), 0)
        self.assertTrue((recommendations['predicted_rating'] >= 1).all())
        self.assertTrue((recommendations['predicted_rating'] <= 5).all())


class TestContentBasedFilter(unittest.TestCase):
    """Test content-based filtering"""
    
    def setUp(self):
        # Create movie data with genres
        self.movies_df = pd.DataFrame({
            'item_id': [1, 2, 3, 4, 5],
            'title': ['Action Movie', 'Comedy Movie', 'Action Comedy', 'Drama', 'Action Drama'],
            'genres': ['Action', 'Comedy', 'Action|Comedy', 'Drama', 'Action|Drama']
        })
        
        self.ratings_df = pd.DataFrame({
            'user_id': [1, 1, 2, 2],
            'item_id': [1, 3, 2, 4],
            'rating':  [5, 4, 5, 3],
            'timestamp': [1] * 4
        })
    
    def test_content_features(self):
        """Test content feature creation"""
        model = ContentBasedFilter()
        model.fit(self.movies_df, self.ratings_df)
        
        # Check TF-IDF matrix
        self.assertIsNotNone(model.tfidf_matrix)
        self.assertEqual(model.tfidf_matrix.shape[0], len(self.movies_df))
    
    def test_movie_similarity(self):
        """Test finding similar movies"""
        model = ContentBasedFilter()
        model.fit(self.movies_df, self.ratings_df)
        
        # Movies 1 and 3 should be similar (both have Action)
        similar = model.get_movie_similarity(movie_id=1, n_similar=2)
        self.assertGreater(len(similar), 0)
        
        # Check similarity scores are between 0 and 1
        self.assertTrue((similar['similarity_score'] >= 0).all())
        self.assertTrue((similar['similarity_score'] <= 1).all())
    
    def test_user_profile(self):
        """Test user profile creation"""
        model = ContentBasedFilter()
        model.fit(self.movies_df, self.ratings_df)
        
        # User 1 liked Action movies
        recommendations = model.recommend_for_user(user_id=1, n_recommendations=2)
        self.assertGreater(len(recommendations), 0)


class TestHybridRecommender(unittest.TestCase):
    """Test hybrid recommendation model"""
    
    def setUp(self):
        # Create test data
        np.random.seed(42)
        n_users, n_items = 10, 20
        n_ratings = 100
        
        users = np.random.choice(range(1, n_users + 1), n_ratings)
        items = np.random.choice(range(1, n_items + 1), n_ratings)
        ratings = np.random.choice([1, 2, 3, 4, 5], n_ratings)
        
        self.ratings_df = pd.DataFrame({
            'user_id': users,
            'item_id': items,
            'rating': ratings,
            'timestamp': [1] * n_ratings
        })
        
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi']
        self.movies_df = pd.DataFrame({
            'item_id': range(1, n_items + 1),
            'title': [f'Movie_{i}' for i in range(1, n_items + 1)],
            'genres': [np.random.choice(genres) for _ in range(n_items)]
        })
    
    def test_hybrid_weighted(self):
        """Test weighted hybrid approach"""
        model = HybridRecommender(cf_weight=0.6, cb_weight=0.4)
        model.fit(self.ratings_df, self.movies_df)
        
        recommendations = model.recommend(user_id=1, n_recommendations=5, method='weighted')
        self.assertGreater(len(recommendations), 0)
        self.assertIn('item_id', recommendations.columns)
        self.assertIn('predicted_rating', recommendations.columns)
    
    def test_hybrid_switching(self):
        """Test switching hybrid approach"""
        model = HybridRecommender()
        model.fit(self.ratings_df, self.movies_df)
        
        recommendations = model.recommend(user_id=1, n_recommendations=5, method='switching')
        self.assertGreater(len(recommendations), 0)
    
    def test_explain_recommendation(self):
        """Test recommendation explanation"""
        model = HybridRecommender()
        model.fit(self.ratings_df, self.movies_df)
        
        explanation = model.explain_recommendation(user_id=1, item_id=1)
        self.assertIn('user_id', explanation)
        self.assertIn('item_id', explanation)
        self.assertIn('reasons', explanation)
        self.assertIsInstance(explanation['reasons'], list)


class TestEvaluator(unittest.TestCase):
    """Test evaluation metrics"""
    
    def setUp(self):
        # Create test data
        self.train_df = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3],
            'item_id': [1, 2, 1, 3, 2, 3],
            'rating':  [5, 4, 3, 5, 4, 3],
            'timestamp': [1] * 6
        })
        
        self.test_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'item_id': [3, 2, 1],
            'rating':  [4, 4, 5],
            'timestamp': [1] * 3
        })
    
    def test_rmse_mae(self):
        """Test RMSE and MAE calculation"""
        model = CollaborativeFilter(method='user_based')
        model.fit(self.train_df)
        
        evaluator = RecommenderEvaluator()
        metrics = evaluator.evaluate_rating_prediction(model, self.test_df)
        
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertGreater(metrics['rmse'], 0)
        self.assertGreater(metrics['mae'], 0)
    
    def test_ranking_metrics(self):
        """Test precision and recall calculation"""
        model = CollaborativeFilter(method='user_based')
        model.fit(self.train_df)
        
        evaluator = RecommenderEvaluator(k_values=[2, 3])
        metrics = evaluator.evaluate_ranking(model, self.test_df, self.train_df)
        
        self.assertIn('precision@2', metrics)
        self.assertIn('recall@2', metrics)
        self.assertIn('f1@2', metrics)
        
        # Check values are between 0 and 1
        for key, value in metrics.items():
            if 'precision' in key or 'recall' in key or 'f1' in key:
                self.assertTrue(0 <= value <= 1)


if __name__ == '__main__':
    unittest.main()
