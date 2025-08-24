"""
Demo script for the movie recommendation system
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.data_loader import MovieLensLoader
from src.models.collaborative import CollaborativeFilter, MatrixFactorization
from src.models.content_based import ContentBasedFilter
from src.models.hybrid import HybridRecommender
from src.evaluation.metrics import RecommenderEvaluator


def print_recommendations(recommendations, title="Recommendations"):
    """Pretty print recommendations"""
    print(f"\n{title}")
    print("-" * 50)
    for i, row in recommendations.iterrows():
        print(f"{i+1}. Item {row['item_id']}: Score = {row['predicted_rating']:.2f}")


def demo_collaborative_filtering():
    """Demo collaborative filtering"""
    print("\n" + "="*60)
    print("COLLABORATIVE FILTERING DEMO")
    print("="*60)
    
    # Load data
    loader = MovieLensLoader('data/raw/')
    ratings_df = loader.load_ratings(min_ratings=5)
    
    # Split data
    train_df, test_df = loader.create_train_test_split(test_size=0.2)
    
    # User-based collaborative filtering
    print("\n1. User-Based Collaborative Filtering")
    cf_user = CollaborativeFilter(method='user_based', k_neighbors=20)
    cf_user.fit(train_df)
    
    # Get recommendations for user 1
    user_id = 1
    recommendations = cf_user.recommend(user_id, n_recommendations=5)
    print_recommendations(recommendations, f"Top 5 recommendations for User {user_id} (User-Based)")
    
    # Item-based collaborative filtering
    print("\n2. Item-Based Collaborative Filtering")
    cf_item = CollaborativeFilter(method='item_based', k_neighbors=20)
    cf_item.fit(train_df)
    
    recommendations = cf_item.recommend(user_id, n_recommendations=5)
    print_recommendations(recommendations, f"Top 5 recommendations for User {user_id} (Item-Based)")
    
    # Evaluate
    evaluator = RecommenderEvaluator()
    metrics = evaluator.evaluate_rating_prediction(cf_user, test_df)
    print(f"\nUser-Based CF - RMSE: {metrics.get('rmse', 0):.4f}, MAE: {metrics.get('mae', 0):.4f}")
    
    metrics = evaluator.evaluate_rating_prediction(cf_item, test_df)
    print(f"Item-Based CF - RMSE: {metrics.get('rmse', 0):.4f}, MAE: {metrics.get('mae', 0):.4f}")


def demo_matrix_factorization():
    """Demo matrix factorization"""
    print("\n" + "="*60)
    print("MATRIX FACTORIZATION DEMO")
    print("="*60)
    
    # Load data
    loader = MovieLensLoader('data/raw/')
    ratings_df = loader.load_ratings(min_ratings=5)
    train_df, test_df = loader.create_train_test_split(test_size=0.2)
    
    # SVD
    print("\n1. Singular Value Decomposition (SVD)")
    svd_model = MatrixFactorization(method='svd', n_factors=20)
    svd_model.fit(train_df)
    
    user_id = 1
    recommendations = svd_model.recommend(user_id, n_recommendations=5)
    print_recommendations(recommendations, f"Top 5 recommendations for User {user_id} (SVD)")
    
    # NMF
    print("\n2. Non-negative Matrix Factorization (NMF)")
    nmf_model = MatrixFactorization(method='nmf', n_factors=20)
    nmf_model.fit(train_df)
    
    recommendations = nmf_model.recommend(user_id, n_recommendations=5)
    print_recommendations(recommendations, f"Top 5 recommendations for User {user_id} (NMF)")
    
    # Evaluate
    evaluator = RecommenderEvaluator()
    metrics = evaluator.evaluate_rating_prediction(svd_model, test_df)
    print(f"\nSVD - RMSE: {metrics.get('rmse', 0):.4f}, MAE: {metrics.get('mae', 0):.4f}")
    
    metrics = evaluator.evaluate_rating_prediction(nmf_model, test_df)
    print(f"NMF - RMSE: {metrics.get('rmse', 0):.4f}, MAE: {metrics.get('mae', 0):.4f}")


def demo_content_based():
    """Demo content-based filtering"""
    print("\n" + "="*60)
    print("CONTENT-BASED FILTERING DEMO")
    print("="*60)
    
    # Load data
    loader = MovieLensLoader('data/raw/')
    ratings_df = loader.load_ratings(min_ratings=5)
    movies_df = loader.load_movies()
    train_df, test_df = loader.create_train_test_split(test_size=0.2)
    
    # Create content-based model
    cb_model = ContentBasedFilter()
    cb_model.fit(movies_df, train_df)
    
    # Find similar movies
    movie_id = 1
    similar_movies = cb_model.get_movie_similarity(movie_id, n_similar=5)
    print(f"\nMovies similar to Movie {movie_id}:")
    print("-" * 50)
    for i, row in similar_movies.iterrows():
        print(f"{i+1}. Item {row['item_id']}: Similarity = {row['similarity_score']:.3f}")
    
    # User recommendations
    user_id = 1
    recommendations = cb_model.recommend_for_user(user_id, n_recommendations=5)
    print_recommendations(recommendations, f"\nTop 5 recommendations for User {user_id} (Content-Based)")
    
    # Evaluate
    evaluator = RecommenderEvaluator()
    metrics = evaluator.evaluate_rating_prediction(cb_model, test_df)
    print(f"\nContent-Based - RMSE: {metrics.get('rmse', 0):.4f}, MAE: {metrics.get('mae', 0):.4f}")


def demo_hybrid():
    """Demo hybrid recommendation"""
    print("\n" + "="*60)
    print("HYBRID RECOMMENDATION DEMO")
    print("="*60)
    
    # Load data
    loader = MovieLensLoader('data/raw/')
    ratings_df = loader.load_ratings(min_ratings=5)
    movies_df = loader.load_movies()
    train_df, test_df = loader.create_train_test_split(test_size=0.2)
    
    # Create hybrid model
    hybrid_model = HybridRecommender(cf_weight=0.7, cb_weight=0.3)
    hybrid_model.fit(train_df, movies_df)
    
    user_id = 1
    
    # Weighted hybrid
    print("\n1. Weighted Hybrid Approach")
    recommendations = hybrid_model.recommend(user_id, n_recommendations=5, method='weighted')
    print_recommendations(recommendations, f"Top 5 recommendations for User {user_id} (Weighted)")
    
    # Switching hybrid
    print("\n2. Switching Hybrid Approach")
    recommendations = hybrid_model.recommend(user_id, n_recommendations=5, method='switching')
    print_recommendations(recommendations, f"Top 5 recommendations for User {user_id} (Switching)")
    
    # Mixed hybrid
    print("\n3. Mixed Hybrid Approach")
    recommendations = hybrid_model.recommend(user_id, n_recommendations=5, method='mixed')
    print_recommendations(recommendations, f"Top 5 recommendations for User {user_id} (Mixed)")
    
    # Explain a recommendation
    if not recommendations.empty:
        item_to_explain = recommendations.iloc[0]['item_id']
        explanation = hybrid_model.explain_recommendation(user_id, item_to_explain)
        print(f"\nExplanation for recommending Item {item_to_explain} to User {user_id}:")
        print("-" * 50)
        for reason in explanation['reasons']:
            print(f"• {reason}")
    
    # Evaluate
    evaluator = RecommenderEvaluator()
    metrics = evaluator.evaluate_all(hybrid_model, train_df, test_df, movies_df)
    print(f"\nHybrid Model Performance:")
    print("-" * 50)
    print(f"RMSE: {metrics.get('rmse', 0):.4f}")
    print(f"MAE: {metrics.get('mae', 0):.4f}")
    print(f"Precision@10: {metrics.get('precision@10', 0):.4f}")
    print(f"Recall@10: {metrics.get('recall@10', 0):.4f}")
    print(f"Coverage: {metrics.get('coverage', 0):.4f}")
    print(f"Diversity: {metrics.get('avg_intra_list_diversity', 0):.4f}")


def demo_evaluation_metrics():
    """Demo comprehensive evaluation"""
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION DEMO")
    print("="*60)
    
    # Load data
    loader = MovieLensLoader('data/raw/')
    ratings_df = loader.load_ratings(min_ratings=10)
    movies_df = loader.load_movies()
    train_df, test_df = loader.create_train_test_split(test_size=0.2)
    
    # Train multiple models
    models = {
        'User-Based CF': CollaborativeFilter(method='user_based'),
        'Item-Based CF': CollaborativeFilter(method='item_based'),
        'SVD': MatrixFactorization(method='svd', n_factors=30),
        'Content-Based': ContentBasedFilter(),
        'Hybrid': HybridRecommender()
    }
    
    evaluator = RecommenderEvaluator(k_values=[5, 10, 20])
    results = []
    
    print("\nTraining and evaluating models...")
    print("-" * 50)
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Fit model
        if isinstance(model, (ContentBasedFilter, HybridRecommender)):
            model.fit(movies_df, train_df) if isinstance(model, ContentBasedFilter) else model.fit(train_df, movies_df)
        else:
            model.fit(train_df)
        
        # Evaluate
        metrics = evaluator.evaluate_all(model, train_df, test_df, movies_df)
        metrics['Model'] = name
        results.append(metrics)
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    
    # Display results
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    
    metrics_to_show = ['Model', 'rmse', 'mae', 'precision@10', 'recall@10', 'coverage']
    
    print("\nRating Prediction Metrics:")
    print(results_df[['Model', 'rmse', 'mae']].to_string(index=False))
    
    print("\nRanking Metrics:")
    print(results_df[['Model', 'precision@10', 'recall@10', 'f1@10']].to_string(index=False))
    
    print("\nDiversity & Coverage Metrics:")
    print(results_df[['Model', 'coverage', 'avg_intra_list_diversity']].to_string(index=False))
    
    # Find best model
    best_rmse_model = results_df.loc[results_df['rmse'].idxmin(), 'Model']
    best_precision_model = results_df.loc[results_df['precision@10'].idxmax(), 'Model']
    
    print(f"\nBest model by RMSE: {best_rmse_model}")
    print(f"Best model by Precision@10: {best_precision_model}")


def main():
    """Run all demos"""
    
    print("\n" + "="*70)
    print(" MOVIE RECOMMENDATION SYSTEM - COMPREHENSIVE DEMO")
    print("="*70)
    print("\nThis demo showcases various recommendation algorithms:")
    print("1. Collaborative Filtering (User-based and Item-based)")
    print("2. Matrix Factorization (SVD and NMF)")
    print("3. Content-Based Filtering")
    print("4. Hybrid Approaches")
    print("5. Comprehensive Evaluation")
    
    # Run demos
    try:
        demo_collaborative_filtering()
        demo_matrix_factorization()
        demo_content_based()
        demo_hybrid()
        demo_evaluation_metrics()
        
        print("\n" + "="*70)
        print(" DEMO COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nKey Takeaways:")
        print("• Collaborative filtering leverages user-item interactions")
        print("• Matrix factorization reduces dimensionality for scalability")
        print("• Content-based filtering uses item features for recommendations")
        print("• Hybrid approaches combine multiple methods for better performance")
        print("• Comprehensive evaluation is crucial for model selection")
        print("\nNext steps:")
        print("1. Download real MovieLens data from https://grouplens.org/datasets/movielens/")
        print("2. Run training script: python src/train.py --model hybrid")
        print("3. Explore notebooks in notebooks/ directory")
        print("4. Build your own recommendation model!")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        print("Note: This demo uses synthetic data. Download real MovieLens data for better results.")


if __name__ == "__main__":
    main()
