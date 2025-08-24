"""
Data loader module for handling MovieLens dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieLensLoader:
    """Load and preprocess MovieLens dataset"""
    
    def __init__(self, data_path: str):
        """
        Initialize data loader
        
        Args:
            data_path: Path to MovieLens dataset directory
        """
        self.data_path = Path(data_path)
        self.ratings = None
        self.movies = None
        self.users = None
        
    def load_ratings(self, min_ratings: int = 20) -> pd.DataFrame:
        """
        Load ratings data
        
        Args:
            min_ratings: Minimum number of ratings per user/item
            
        Returns:
            DataFrame with ratings
        """
        logger.info("Loading ratings data...")
        
        # MovieLens 100K format
        ratings_file = self.data_path / 'u.data'
        
        if ratings_file.exists():
            self.ratings = pd.read_csv(
                ratings_file,
                sep='\t',
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                encoding='latin-1'
            )
        else:
            # Generate synthetic data for demo
            logger.warning("Data file not found. Generating synthetic data...")
            self.ratings = self._generate_synthetic_ratings()
        
        # Filter users and items with minimum ratings
        if min_ratings > 0:
            self.ratings = self._filter_min_ratings(self.ratings, min_ratings)
        
        logger.info(f"Loaded {len(self.ratings)} ratings from {self.ratings['user_id'].nunique()} users")
        return self.ratings
    
    def load_movies(self) -> pd.DataFrame:
        """
        Load movie metadata
        
        Returns:
            DataFrame with movie information
        """
        logger.info("Loading movie metadata...")
        
        movies_file = self.data_path / 'u.item'
        
        if movies_file.exists():
            # MovieLens 100K format
            columns = ['item_id', 'title', 'release_date', 'video_release_date', 
                      'imdb_url'] + [f'genre_{i}' for i in range(19)]
            
            self.movies = pd.read_csv(
                movies_file,
                sep='|',
                names=columns,
                encoding='latin-1'
            )
            
            # Extract genres
            genre_columns = [col for col in self.movies.columns if col.startswith('genre_')]
            self.movies['genres'] = self.movies[genre_columns].apply(
                lambda x: '|'.join([col.replace('genre_', '') for col, val in x.items() if val == 1]),
                axis=1
            )
        else:
            # Generate synthetic movie data
            self.movies = self._generate_synthetic_movies()
        
        logger.info(f"Loaded {len(self.movies)} movies")
        return self.movies
    
    def load_users(self) -> pd.DataFrame:
        """
        Load user demographics
        
        Returns:
            DataFrame with user information
        """
        logger.info("Loading user data...")
        
        users_file = self.data_path / 'u.user'
        
        if users_file.exists():
            self.users = pd.read_csv(
                users_file,
                sep='|',
                names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
                encoding='latin-1'
            )
        else:
            # Generate synthetic user data
            self.users = self._generate_synthetic_users()
        
        logger.info(f"Loaded {len(self.users)} users")
        return self.users
    
    def create_train_test_split(self, test_size: float = 0.2, 
                               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split for ratings
        
        Args:
            test_size: Proportion of test data
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if self.ratings is None:
            self.load_ratings()
        
        logger.info(f"Creating train/test split with test_size={test_size}")
        
        # Ensure each user has ratings in both train and test
        from sklearn.model_selection import train_test_split
        
        train_data = []
        test_data = []
        
        for user_id in self.ratings['user_id'].unique():
            user_ratings = self.ratings[self.ratings['user_id'] == user_id]
            
            if len(user_ratings) >= 2:  # Need at least 2 ratings to split
                train, test = train_test_split(
                    user_ratings,
                    test_size=test_size,
                    random_state=random_state
                )
                train_data.append(train)
                test_data.append(test)
            else:
                train_data.append(user_ratings)
        
        train_df = pd.concat(train_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
        
        logger.info(f"Train set: {len(train_df)} ratings, Test set: {len(test_df)} ratings")
        
        return train_df, test_df
    
    def create_user_item_matrix(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create user-item interaction matrix
        
        Args:
            df: Ratings DataFrame (uses self.ratings if None)
            
        Returns:
            User-item matrix with ratings
        """
        if df is None:
            if self.ratings is None:
                self.load_ratings()
            df = self.ratings
        
        logger.info("Creating user-item matrix...")
        
        matrix = df.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            fill_value=0
        )
        
        logger.info(f"Created matrix of shape {matrix.shape}")
        return matrix
    
    def _filter_min_ratings(self, df: pd.DataFrame, min_ratings: int) -> pd.DataFrame:
        """Filter users and items with minimum number of ratings"""
        
        # Filter users
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_ratings].index
        df = df[df['user_id'].isin(valid_users)]
        
        # Filter items
        item_counts = df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= min_ratings].index
        df = df[df['item_id'].isin(valid_items)]
        
        return df
    
    def _generate_synthetic_ratings(self, n_users: int = 100, 
                                   n_items: int = 500, 
                                   n_ratings: int = 10000) -> pd.DataFrame:
        """Generate synthetic ratings data for demonstration"""
        
        np.random.seed(42)
        
        # Create power-law distribution for realistic data
        user_activity = np.random.pareto(0.5, n_users)
        user_activity = user_activity / user_activity.sum()
        
        item_popularity = np.random.pareto(1.0, n_items)
        item_popularity = item_popularity / item_popularity.sum()
        
        ratings_data = []
        
        for _ in range(n_ratings):
            user_id = np.random.choice(range(1, n_users + 1), p=user_activity)
            item_id = np.random.choice(range(1, n_items + 1), p=item_popularity)
            
            # Rating with some noise
            base_rating = np.random.choice([3, 4, 5], p=[0.3, 0.5, 0.2])
            noise = np.random.normal(0, 0.5)
            rating = np.clip(base_rating + noise, 1, 5)
            rating = int(np.round(rating))
            
            timestamp = np.random.randint(880000000, 893000000)
            
            ratings_data.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating,
                'timestamp': timestamp
            })
        
        return pd.DataFrame(ratings_data)
    
    def _generate_synthetic_movies(self) -> pd.DataFrame:
        """Generate synthetic movie data"""
        
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 
                 'Thriller', 'Sci-Fi', 'Documentary', 'Animation']
        
        movies_data = []
        
        if self.ratings is not None:
            unique_items = self.ratings['item_id'].unique()
        else:
            unique_items = range(1, 501)
        
        for item_id in unique_items:
            # Generate random genres (1-3 per movie)
            n_genres = np.random.randint(1, 4)
            movie_genres = np.random.choice(genres, n_genres, replace=False)
            
            movies_data.append({
                'item_id': item_id,
                'title': f"Movie_{item_id}",
                'genres': '|'.join(movie_genres),
                'release_year': np.random.randint(1970, 2024)
            })
        
        return pd.DataFrame(movies_data)
    
    def _generate_synthetic_users(self) -> pd.DataFrame:
        """Generate synthetic user data"""
        
        occupations = ['student', 'engineer', 'teacher', 'doctor', 'artist',
                      'writer', 'manager', 'programmer', 'retired', 'other']
        
        users_data = []
        
        if self.ratings is not None:
            unique_users = self.ratings['user_id'].unique()
        else:
            unique_users = range(1, 101)
        
        for user_id in unique_users:
            users_data.append({
                'user_id': user_id,
                'age': np.random.randint(18, 70),
                'gender': np.random.choice(['M', 'F']),
                'occupation': np.random.choice(occupations)
            })
        
        return pd.DataFrame(users_data)


class DataPreprocessor:
    """Preprocess data for recommendation models"""
    
    @staticmethod
    def normalize_ratings(ratings: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
        """
        Normalize ratings by user
        
        Args:
            ratings: Ratings DataFrame
            method: Normalization method ('mean', 'zscore')
            
        Returns:
            Normalized ratings DataFrame
        """
        ratings = ratings.copy()
        
        if method == 'mean':
            # Mean centering
            user_means = ratings.groupby('user_id')['rating'].mean()
            ratings['rating_normalized'] = ratings.apply(
                lambda x: x['rating'] - user_means[x['user_id']], axis=1
            )
        elif method == 'zscore':
            # Z-score normalization
            user_stats = ratings.groupby('user_id')['rating'].agg(['mean', 'std'])
            ratings['rating_normalized'] = ratings.apply(
                lambda x: (x['rating'] - user_stats.loc[x['user_id'], 'mean']) / 
                         (user_stats.loc[x['user_id'], 'std'] + 1e-8), axis=1
            )
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return ratings
    
    @staticmethod
    def create_implicit_feedback(ratings: pd.DataFrame, threshold: float = 3.5) -> pd.DataFrame:
        """
        Convert explicit ratings to implicit feedback
        
        Args:
            ratings: Ratings DataFrame
            threshold: Rating threshold for positive feedback
            
        Returns:
            Implicit feedback DataFrame
        """
        implicit = ratings.copy()
        implicit['feedback'] = (implicit['rating'] >= threshold).astype(int)
        implicit = implicit.drop('rating', axis=1)
        
        return implicit
