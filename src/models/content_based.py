"""
Content-based filtering recommendation model
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class ContentBasedFilter:
    """Content-based filtering using movie features"""
    
    def __init__(self, features: List[str] = ['genres', 'director', 'cast'],
                 max_features: int = 1000):
        """
        Initialize content-based filter
        
        Args:
            features: List of features to use for similarity
            max_features: Maximum number of TF-IDF features
        """
        self.features = features
        self.max_features = max_features
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.movies_df = None
        self.user_profiles = {}
        self.ratings_df = None
        
    def fit(self, movies_df: pd.DataFrame, ratings_df: Optional[pd.DataFrame] = None):
        """
        Fit the content-based model
        
        Args:
            movies_df: DataFrame with movie features
            ratings_df: Optional ratings for building user profiles
        """
        logger.info("Fitting content-based filter...")
        
        self.movies_df = movies_df.copy()
        self.ratings_df = ratings_df
        
        # Create content features
        self._create_content_features()
        
        # Create TF-IDF matrix
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.movies_df['content_features']
        )
        
        # Build user profiles if ratings provided
        if ratings_df is not None:
            self._build_user_profiles(ratings_df)
        
        logger.info(f"Content-based model fitted with {len(self.movies_df)} movies")
        
    def _create_content_features(self):
        """Create combined content features from movie attributes"""
        
        # Handle missing values
        for feature in self.features:
            if feature in self.movies_df.columns:
                self.movies_df[feature] = self.movies_df[feature].fillna('')
        
        # Combine features into single text
        if 'genres' in self.movies_df.columns:
            # Process genres (replace | with space)
            self.movies_df['genres_processed'] = self.movies_df['genres'].str.replace('|', ' ')
        else:
            self.movies_df['genres_processed'] = ''
        
        # Create combined features
        feature_texts = []
        for _, movie in self.movies_df.iterrows():
            text_parts = []
            
            if 'genres_processed' in self.movies_df.columns:
                text_parts.append(movie['genres_processed'])
            
            if 'title' in self.movies_df.columns:
                # Extract year from title if present
                title = str(movie['title'])
                text_parts.append(title)
            
            if 'release_year' in self.movies_df.columns:
                text_parts.append(f"year_{movie['release_year']}")
            
            feature_texts.append(' '.join(str(part) for part in text_parts))
        
        self.movies_df['content_features'] = feature_texts
        
    def _build_user_profiles(self, ratings_df: pd.DataFrame):
        """Build user profiles based on their ratings"""
        
        logger.info("Building user profiles...")
        
        for user_id in ratings_df['user_id'].unique():
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
            
            # Get movies rated highly by user (>= 4)
            liked_movies = user_ratings[user_ratings['rating'] >= 4]['item_id'].values
            
            if len(liked_movies) > 0:
                # Get indices of liked movies
                movie_indices = []
                for movie_id in liked_movies:
                    if movie_id in self.movies_df['item_id'].values:
                        idx = self.movies_df[self.movies_df['item_id'] == movie_id].index[0]
                        movie_indices.append(idx)
                
                if movie_indices:
                    # Average TF-IDF vectors of liked movies
                    user_profile = self.tfidf_matrix[movie_indices].mean(axis=0)
                    self.user_profiles[user_id] = user_profile
        
        logger.info(f"Built profiles for {len(self.user_profiles)} users")
        
    def get_movie_similarity(self, movie_id: int, n_similar: int = 10) -> pd.DataFrame:
        """
        Find similar movies to a given movie
        
        Args:
            movie_id: Movie ID
            n_similar: Number of similar movies to return
            
        Returns:
            DataFrame with similar movies and similarity scores
        """
        if movie_id not in self.movies_df['item_id'].values:
            logger.warning(f"Movie {movie_id} not found")
            return pd.DataFrame()
        
        # Get movie index
        movie_idx = self.movies_df[self.movies_df['item_id'] == movie_id].index[0]
        
        # Calculate similarity with all movies
        movie_vector = self.tfidf_matrix[movie_idx]
        similarities = cosine_similarity(movie_vector, self.tfidf_matrix).flatten()
        
        # Get indices of most similar movies (excluding itself)
        similar_indices = similarities.argsort()[-n_similar-1:-1][::-1]
        
        # Create results DataFrame
        results = []
        for idx in similar_indices:
            if idx != movie_idx:  # Exclude the movie itself
                results.append({
                    'item_id': self.movies_df.iloc[idx]['item_id'],
                    'title': self.movies_df.iloc[idx].get('title', f"Movie_{self.movies_df.iloc[idx]['item_id']}"),
                    'similarity_score': similarities[idx],
                    'genres': self.movies_df.iloc[idx].get('genres', '')
                })
        
        return pd.DataFrame(results)
    
    def recommend_for_user(self, user_id: int, n_recommendations: int = 10,
                           exclude_seen: bool = True) -> pd.DataFrame:
        """
        Get recommendations for a user based on their profile
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            exclude_seen: Whether to exclude already rated items
            
        Returns:
            DataFrame with recommended movies
        """
        if user_id not in self.user_profiles:
            logger.warning(f"No profile found for user {user_id}")
            return self._recommend_popular(n_recommendations)
        
        # Get user profile
        user_profile = self.user_profiles[user_id]
        
        # Calculate similarity with all movies
        similarities = cosine_similarity(user_profile, self.tfidf_matrix).flatten()
        
        # Create DataFrame with all movies and scores
        recommendations = pd.DataFrame({
            'item_id': self.movies_df['item_id'],
            'predicted_rating': similarities * 5  # Scale to rating range
        })
        
        # Exclude seen movies if requested
        if exclude_seen and self.ratings_df is not None:
            user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
            seen_movies = user_ratings['item_id'].values
            recommendations = recommendations[~recommendations['item_id'].isin(seen_movies)]
        
        # Sort and return top N
        recommendations = recommendations.sort_values('predicted_rating', ascending=False)
        return recommendations.head(n_recommendations)
    
    def recommend_for_item(self, item_id: int, n_recommendations: int = 10) -> pd.DataFrame:
        """
        Get recommendations based on item similarity
        
        Args:
            item_id: Item ID
            n_recommendations: Number of recommendations
            
        Returns:
            DataFrame with similar items
        """
        similar_movies = self.get_movie_similarity(item_id, n_recommendations)
        
        if not similar_movies.empty:
            # Convert similarity to predicted rating
            similar_movies['predicted_rating'] = similar_movies['similarity_score'] * 5
            return similar_movies[['item_id', 'predicted_rating']]
        
        return pd.DataFrame()
    
    def _recommend_popular(self, n_recommendations: int) -> pd.DataFrame:
        """Recommend popular items as fallback"""
        
        if self.ratings_df is not None:
            # Calculate average ratings
            item_ratings = self.ratings_df.groupby('item_id').agg({
                'rating': ['mean', 'count']
            })
            item_ratings.columns = ['avg_rating', 'count']
            
            # Filter items with minimum ratings
            item_ratings = item_ratings[item_ratings['count'] >= 5]
            
            # Sort by average rating
            item_ratings = item_ratings.sort_values('avg_rating', ascending=False)
            
            recommendations = pd.DataFrame({
                'item_id': item_ratings.head(n_recommendations).index,
                'predicted_rating': item_ratings.head(n_recommendations)['avg_rating'].values
            })
            
            return recommendations
        
        # If no ratings available, return random movies
        sample_movies = self.movies_df.sample(min(n_recommendations, len(self.movies_df)))
        return pd.DataFrame({
            'item_id': sample_movies['item_id'].values,
            'predicted_rating': [3.5] * len(sample_movies)  # Neutral rating
        })


class FeatureExtractor:
    """Extract and process features for content-based filtering"""
    
    @staticmethod
    def extract_year(title: str) -> Optional[int]:
        """Extract year from movie title"""
        import re
        
        # Look for year in parentheses (e.g., "Movie Title (2020)")
        match = re.search(r'\((\d{4})\)', title)
        if match:
            return int(match.group(1))
        return None
    
    @staticmethod
    def process_genres(genres_str: str) -> List[str]:
        """Process genre string into list"""
        if pd.isna(genres_str) or genres_str == '':
            return []
        
        # Split by common delimiters
        genres = genres_str.replace('|', ',').replace(';', ',').split(',')
        
        # Clean and normalize
        processed = []
        for genre in genres:
            genre = genre.strip().lower()
            if genre:
                processed.append(genre)
        
        return processed
    
    @staticmethod
    def extract_keywords(text: str, top_k: int = 10) -> List[str]:
        """Extract keywords from text using TF-IDF"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        if not text or pd.isna(text):
            return []
        
        # Simple keyword extraction
        vectorizer = TfidfVectorizer(
            max_features=top_k,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            return feature_names.tolist()
        except:
            return []
