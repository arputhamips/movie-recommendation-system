"""
Main training script for recommendation models
"""

import argparse
import json
import pickle
from pathlib import Path
import logging
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.data_loader import MovieLensLoader, DataPreprocessor
from src.models.collaborative import CollaborativeFilter, MatrixFactorization
from src.models.content_based import ContentBasedFilter
from src.models.hybrid import HybridRecommender, EnsembleRecommender
from src.evaluation.metrics import RecommenderEvaluator, CrossValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_collaborative(train_df, test_df, method='user_based', **kwargs):
    """Train collaborative filtering model"""
    
    logger.info(f"Training collaborative filtering ({method})...")
    
    model = CollaborativeFilter(method=method, **kwargs)
    model.fit(train_df)
    
    # Evaluate
    evaluator = RecommenderEvaluator()
    metrics = evaluator.evaluate_all(model, train_df, test_df)
    
    return model, metrics


def train_matrix_factorization(train_df, test_df, method='svd', **kwargs):
    """Train matrix factorization model"""
    
    logger.info(f"Training matrix factorization ({method})...")
    
    model = MatrixFactorization(method=method, **kwargs)
    model.fit(train_df)
    
    # Evaluate
    evaluator = RecommenderEvaluator()
    metrics = evaluator.evaluate_all(model, train_df, test_df)
    
    return model, metrics


def train_content_based(train_df, test_df, movies_df, **kwargs):
    """Train content-based filtering model"""
    
    logger.info("Training content-based filtering...")
    
    model = ContentBasedFilter(**kwargs)
    model.fit(movies_df, train_df)
    
    # Evaluate
    evaluator = RecommenderEvaluator()
    metrics = evaluator.evaluate_all(model, train_df, test_df, movies_df)
    
    return model, metrics


def train_hybrid(train_df, test_df, movies_df, **kwargs):
    """Train hybrid recommendation model"""
    
    logger.info("Training hybrid recommender...")
    
    model = HybridRecommender(**kwargs)
    model.fit(train_df, movies_df)
    
    # Evaluate
    evaluator = RecommenderEvaluator()
    metrics = evaluator.evaluate_all(model, train_df, test_df, movies_df)
    
    return model, metrics


def train_ensemble(train_df, test_df, movies_df):
    """Train ensemble recommendation model"""
    
    logger.info("Training ensemble recommender...")
    
    model = EnsembleRecommender()
    model.fit(train_df, movies_df)
    
    # Evaluate
    evaluator = RecommenderEvaluator()
    metrics = evaluator.evaluate_all(model, train_df, test_df, movies_df)
    
    return model, metrics


def save_model(model, model_path: Path, metrics: dict = None):
    """Save trained model and metrics"""
    
    logger.info(f"Saving model to {model_path}")
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metrics
    if metrics:
        metrics_path = model_path.with_suffix('.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Metrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='Train recommendation models')
    parser.add_argument('--model', type=str, default='collaborative',
                       choices=['collaborative', 'matrix_factorization', 'content_based', 
                               'hybrid', 'ensemble'],
                       help='Model type to train')
    parser.add_argument('--data_path', type=str, default='data/raw/ml-100k',
                       help='Path to MovieLens dataset')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--method', type=str, default='user_based',
                       help='Method for collaborative/matrix factorization')
    parser.add_argument('--n_factors', type=int, default=50,
                       help='Number of factors for matrix factorization')
    parser.add_argument('--k_neighbors', type=int, default=50,
                       help='Number of neighbors for collaborative filtering')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--min_ratings', type=int, default=20,
                       help='Minimum ratings per user/item')
    parser.add_argument('--cross_validate', action='store_true',
                       help='Perform cross-validation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    logger.info("Loading data...")
    loader = MovieLensLoader(args.data_path)
    ratings_df = loader.load_ratings(min_ratings=args.min_ratings)
    movies_df = loader.load_movies()
    users_df = loader.load_users()
    
    # Create train/test split
    train_df, test_df = loader.create_train_test_split(test_size=args.test_size)
    
    # Train model based on type
    if args.cross_validate:
        # Perform cross-validation
        validator = CrossValidator(n_folds=5)
        
        if args.model == 'collaborative':
            model_class = CollaborativeFilter
            model_params = {'method': args.method, 'k_neighbors': args.k_neighbors}
            results = validator.cross_validate(model_class, ratings_df, **model_params)
        elif args.model == 'matrix_factorization':
            model_class = MatrixFactorization
            model_params = {'method': args.method, 'n_factors': args.n_factors}
            results = validator.cross_validate(model_class, ratings_df, **model_params)
        elif args.model == 'hybrid':
            model_class = HybridRecommender
            results = validator.cross_validate(model_class, ratings_df, movies_df)
        else:
            logger.error(f"Cross-validation not implemented for {args.model}")
            return
        
        # Save cross-validation results
        results_path = output_dir / f"{args.model}_cv_results.csv"
        results.to_csv(results_path, index=False)
        logger.info(f"Cross-validation results saved to {results_path}")
        
        # Print summary
        print("\nCross-Validation Results:")
        print(results[['fold', 'rmse', 'mae', 'precision@10', 'recall@10']].to_string())
        print(f"\nAverage RMSE: {results['rmse'].mean():.4f} ± {results['rmse'].std():.4f}")
        
    else:
        # Train single model
        if args.model == 'collaborative':
            model, metrics = train_collaborative(
                train_df, test_df, 
                method=args.method,
                k_neighbors=args.k_neighbors
            )
        elif args.model == 'matrix_factorization':
            model, metrics = train_matrix_factorization(
                train_df, test_df,
                method=args.method,
                n_factors=args.n_factors
            )
        elif args.model == 'content_based':
            model, metrics = train_content_based(train_df, test_df, movies_df)
        elif args.model == 'hybrid':
            model, metrics = train_hybrid(train_df, test_df, movies_df)
        elif args.model == 'ensemble':
            model, metrics = train_ensemble(train_df, test_df, movies_df)
        else:
            logger.error(f"Unknown model type: {args.model}")
            return
        
        # Save model
        model_filename = f"{args.model}_{args.method if args.model in ['collaborative', 'matrix_factorization'] else 'model'}.pkl"
        model_path = output_dir / model_filename
        save_model(model, model_path, metrics)
        
        # Print results
        print("\nModel Performance:")
        print("-" * 50)
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric:25s}: {value:.4f}")
            else:
                print(f"{metric:25s}: {value}")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
