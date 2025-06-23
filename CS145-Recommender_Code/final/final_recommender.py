import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from collections import defaultdict
import random

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, ArrayType

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

import sklearn 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sim4rec.utils import pandas_to_spark

import xgboost as xgb
from xgboost import XGBClassifier, callback

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import torch_geometric.nn as geom_nn
from torch_geometric.data import Data



class FinalRecommender:
    """
    The Final Recommender based on an AutoRegressive model. 
    """
    
    def __init__(self, seed=None, n=2, smoothing=0.5):
        """
        Initialize recommender.
        
        Args:
            seed: Random seed for reproducibility
            n: Order of the n-gram model (how many previous items to consider)
            smoothing: Additive smoothing parameter (Laplace smoothing)
        """
        self.seed = seed
        self.n = n  # Reduced from 3 to 2 to capture more patterns
        self.smoothing = smoothing  # Increased from 0.1 to 0.5 for better generalization
        self.model = None
        self.categorical_cols_ = None
        self.numeric_cols_ = None
        self.input_columns_ = None
        self.preprocessor_ = None
        self.feature_importance_ = None
        self.user_encoders_ = {}
        self.item_encoders_ = {}
        
        # AR model components
        self.ngram_counts = defaultdict(lambda: defaultdict(float))
        self.item_counts = defaultdict(float)
        self.total_items = 0
        self.category_counts = defaultdict(lambda: defaultdict(float))  # Added category-based counts
        self.user_category_preferences = defaultdict(lambda: defaultdict(float))  # Added user-category preferences
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def _get_ngram_key(self, sequence):
        """Convert a sequence of items to a tuple key for the n-gram dictionary."""
        # Filter out None values and ensure we have a valid sequence
        valid_sequence = [x for x in sequence if x is not None]
        if len(valid_sequence) == 0:
            return tuple([None] * self.n)
        return tuple(valid_sequence[-self.n:])

    def _create_interaction_features(self, feature_df):
        """Create enhanced interaction features between user and item attributes."""
        # First create all base features
        if 'u_income' in feature_df.columns and 'i_price' in feature_df.columns:
            # Basic price features
            feature_df['price_sensitivity'] = feature_df['i_price'] / (feature_df['u_income'] + 1)
            feature_df['price_affordability'] = feature_df['u_income'] / (feature_df['i_price'] + 1)
            feature_df['price_elasticity'] = feature_df['price_sensitivity'] * feature_df['price_affordability']
            feature_df['price_income_interaction'] = feature_df['i_price'] * feature_df['u_income']
            
            # Price range features with safe binning
            try:
                price_bins = np.percentile(feature_df['i_price'], [0, 20, 40, 60, 80, 100])
                if len(np.unique(price_bins)) > 1:  # Only create bins if values are different
                    feature_df['price_range'] = pd.cut(
                        feature_df['i_price'],
                        bins=price_bins,
                        labels=['very_low', 'low', 'medium', 'high', 'very_high']
                    )
                else:
                    feature_df['price_range'] = 'medium'  # Default value if all prices are the same
            except Exception:
                feature_df['price_range'] = 'medium'  # Fallback if binning fails
        
        # Create segment-specific features
        if 'u_segment' in feature_df.columns:
            # Segment-specific price sensitivity (after price_sensitivity is created)
            if 'price_sensitivity' in feature_df.columns:
                segment_price_means = feature_df.groupby('u_segment')['price_sensitivity'].mean().reset_index()
                segment_price_means = segment_price_means.rename(columns={'price_sensitivity': 'segment_price_sensitivity'})
                feature_df = feature_df.merge(segment_price_means, on='u_segment', how='left')
            
            # Segment-specific category preferences
            if 'price_affordability' in feature_df.columns:
                for segment in feature_df['u_segment'].unique():
                    feature_df[f'segment_{segment}_price_match'] = (
                        (feature_df['u_segment'] == segment) * 
                        feature_df['price_affordability']
                    )
        
        # Create category features
        if 'u_segment' in feature_df.columns and 'i_category' in feature_df.columns:
            # Category preference scores - using user_idx instead of user_id
            if 'user_idx' in feature_df.columns and 'relevance' in feature_df.columns:
                feature_df['category_preference_score'] = feature_df.groupby(['user_idx', 'i_category'])['relevance'].transform('mean')
                feature_df['category_exploration_score'] = feature_df.groupby(['user_idx', 'i_category'])['relevance'].transform('count')
            
            # Segment-category interaction strength
            segment_category_means = feature_df.groupby(['u_segment', 'i_category'])['relevance'].mean().reset_index()
            segment_category_means = segment_category_means.rename(columns={'relevance': 'segment_category_strength'})
            feature_df = feature_df.merge(segment_category_means, on=['u_segment', 'i_category'], how='left')
            
            # Category diversity score
            if 'user_idx' in feature_df.columns:
                feature_df['category_diversity'] = feature_df.groupby('user_idx')['i_category'].transform('nunique')
            
            # Price category interaction (after price_range is created)
            if 'price_range' in feature_df.columns:
                feature_df['price_category_match'] = feature_df['price_range'].astype(str) + '_' + feature_df['i_category'].astype(str)
        
        # Create user engagement features
        if 'u_user_interaction_count' in feature_df.columns:
            # User engagement metrics
            if 'u_user_avg_relevance' in feature_df.columns:
                feature_df['user_engagement_score'] = (
                    feature_df['u_user_interaction_count'] * 
                    feature_df['u_user_avg_relevance']
                )
            
            # User exploration score
            if 'i_category' in feature_df.columns and 'user_idx' in feature_df.columns:
                feature_df['user_exploration_score'] = (
                    feature_df['u_user_interaction_count'] / 
                    feature_df.groupby('user_idx')['i_category'].transform('nunique')
                )
            
            # Activity level with safe binning
            try:
                activity_bins = np.percentile(feature_df['u_user_interaction_count'], [0, 20, 40, 60, 80, 100])
                if len(np.unique(activity_bins)) > 1:  # Only create bins if values are different
                    feature_df['user_activity_level'] = pd.cut(
                        feature_df['u_user_interaction_count'],
                        bins=activity_bins,
                        labels=['very_low', 'low', 'medium', 'high', 'very_high']
                    )
                else:
                    feature_df['user_activity_level'] = 'medium'  # Default value if all counts are the same
            except Exception:
                feature_df['user_activity_level'] = 'medium'  # Fallback if binning fails
        
        # Create item engagement features
        if 'i_item_popularity' in feature_df.columns:
            # Item engagement metrics
            if 'i_item_avg_relevance' in feature_df.columns:
                feature_df['item_engagement_score'] = (
                    feature_df['i_item_popularity'] * 
                    feature_df['i_item_avg_relevance']
                )
            
            # Item diversity score
            if 'u_segment' in feature_df.columns and 'item_idx' in feature_df.columns:
                feature_df['item_diversity_score'] = (
                    feature_df['i_item_popularity'] / 
                    feature_df.groupby('item_idx')['u_segment'].transform('nunique')
                )
            
            # Popularity level with safe binning
            try:
                popularity_bins = np.percentile(feature_df['i_item_popularity'], [0, 20, 40, 60, 80, 100])
                if len(np.unique(popularity_bins)) > 1:  # Only create bins if values are different
                    feature_df['item_popularity_level'] = pd.cut(
                        feature_df['i_item_popularity'],
                        bins=popularity_bins,
                        labels=['very_low', 'low', 'medium', 'high', 'very_high']
                    )
                else:
                    feature_df['item_popularity_level'] = 'medium'  # Default value if all popularities are the same
            except Exception:
                feature_df['item_popularity_level'] = 'medium'  # Fallback if binning fails
        
        # Create polynomial features
        numeric_cols = ['u_income', 'i_price', 'u_user_avg_relevance', 'i_item_avg_relevance']
        for col in numeric_cols:
            if col in feature_df.columns:
                feature_df[f'{col}_squared'] = feature_df[col] ** 2
                feature_df[f'{col}_cubed'] = feature_df[col] ** 3
        
        return feature_df

    def _create_aggregate_features(self, log_df, user_features, item_features):
        """Create aggregate features from historical interactions."""
        # Convert to pandas if it's a Spark DataFrame
        if hasattr(log_df, 'toPandas'):
            log_df = log_df.toPandas()
        if hasattr(user_features, 'toPandas'):
            user_features = user_features.toPandas()
        if hasattr(item_features, 'toPandas'):
            item_features = item_features.toPandas()
        
        # User interaction counts and statistics
        user_stats = log_df.groupby("user_idx").agg({
            'relevance': ['count', 'mean', 'std', 'max']
        }).reset_index()
        user_stats.columns = ['user_idx', 'user_interaction_count', 'user_avg_relevance', 'user_relevance_std', 'user_max_relevance']
        
        # Item popularity and statistics
        item_stats = log_df.groupby("item_idx").agg({
            'relevance': ['count', 'mean', 'std', 'max']
        }).reset_index()
        item_stats.columns = ['item_idx', 'item_popularity', 'item_avg_relevance', 'item_relevance_std', 'item_max_relevance']
        
        # Category-level statistics
        if 'category' in item_features.columns:
            category_stats = log_df.merge(
                item_features[['item_idx', 'category']],
                on='item_idx'
            ).groupby('category').agg({
                'relevance': ['mean', 'count']
            }).reset_index()
            category_stats.columns = ['category', 'category_avg_relevance', 'category_popularity']
            
            # Join category stats back to items
            item_features = item_features.merge(
                category_stats,
                on='category',
                how='left'
            )
        
        return user_stats, item_stats

    def fit(self, log, user_features=None, item_features=None):
        """Train the recommender model with enhanced features and preprocessing."""
        # Convert to pandas if it's a Spark DataFrame
        if hasattr(log, 'toPandas'):
            log = log.toPandas()
        if hasattr(user_features, 'toPandas'):
            user_features = user_features.toPandas()
        if hasattr(item_features, 'toPandas'):
            item_features = item_features.toPandas()
        
        # Build n-gram counts and category preferences
        user_sequences = log.groupby('user_idx')
        for user_idx, user_log in user_sequences:
            items = user_log['item_idx'].tolist()
            categories = user_log['category'].tolist() if 'category' in user_log.columns else []
            
            # Update category preferences
            for item, category in zip(items, categories):
                self.user_category_preferences[user_idx][category] += 1
                self.category_counts[category][item] += 1
            
            # Skip users with too few interactions
            if len(items) <= self.n:
                continue
            
            # Count n-grams and next items
            for i in range(len(items) - self.n):
                context = items[i:i+self.n]
                next_item = items[i+self.n]
                
                # Update counts
                self.ngram_counts[self._get_ngram_key(context)][next_item] += 1
                self.item_counts[next_item] += 1
                self.total_items += 1
        
        # Create aggregate features
        user_stats, item_stats = self._create_aggregate_features(log, user_features, item_features)
        
        # Join features with user and item data
        user_features = user_features.merge(user_stats, on="user_idx", how="left")
        item_features = item_features.merge(item_stats, on="item_idx", how="left")
        
        # Add prefixes to features
        user_features = user_features.rename(columns={c: f"u_{c}" for c in user_features.columns if c != "user_idx"})
        item_features = item_features.rename(columns={c: f"i_{c}" for c in item_features.columns if c != "item_idx"})
        
        # Join all data
        joined = log.merge(user_features, on="user_idx", how="inner").merge(item_features, on="item_idx", how="inner")
        
        # Create interaction features
        feature_df = joined.copy()  # Keep all columns including IDs
        feature_df = self._create_interaction_features(feature_df)
        
        # Split columns by dtype
        categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = feature_df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
        
        # Remove ID columns from feature sets
        id_cols = ['user_idx', 'item_idx']
        categorical_cols = [col for col in categorical_cols if col not in id_cols]
        numeric_cols = [col for col in numeric_cols if col not in id_cols]
        
        # Save column information
        self.categorical_cols_ = categorical_cols
        self.numeric_cols_ = numeric_cols
        self.input_columns_ = categorical_cols + numeric_cols
        
        # Handle missing values and ensure proper types
        for col in categorical_cols:
            feature_df[col] = feature_df[col].astype(str)
            feature_df[col] = feature_df[col].fillna('missing')
            
        for col in numeric_cols:
            feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce').fillna(0)
        
        # Create preprocessing pipeline with scaling
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
        # Transform features
        feature_df = feature_df.reindex(columns=self.input_columns_)
        X_transformed = preprocessor.fit_transform(feature_df)
        
        # Convert relevance to binary (0 or 1)
        y = (joined["relevance"] > 0).astype(int)
        
        # Split data for validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_transformed, y, test_size=0.2, random_state=self.seed, stratify=y
        )
        
        # Enhanced parameter grid for L2 regularization with adaptive C values
        param_grid = {
            'C': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],  # Wider range of C values
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'max_iter': [2000],
            'tol': [1e-4, 1e-5],
            'class_weight': ['balanced']
        }
        
        base_model = LogisticRegression(
            random_state=self.seed,
            max_iter=2000,
            tol=1e-4,
            warm_start=True
        )
        
        # Use stratified cross-validation with more folds
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)
        
        # Custom scoring function that considers both classification and revenue
        def revenue_scorer(estimator, X, y):
            y_pred = estimator.predict_proba(X)[:, 1]
            # Calculate revenue score with additional regularization penalty
            revenue_score = np.mean(y_pred * y)
            # Add L2 penalty based on feature importance
            coef = np.abs(estimator.coef_[0])
            l2_penalty = np.mean(coef ** 2)  # Mean squared coefficients
            # Add stability penalty
            stability_penalty = np.std(coef)  # Penalize high variance in coefficients
            return revenue_score - 0.1 * l2_penalty - 0.05 * stability_penalty
        
        # Ensemble feature selection
        from sklearn.feature_selection import SelectFromModel
        from sklearn.ensemble import RandomForestClassifier
        
        # First pass: Use Random Forest for initial feature importance
        rf_selector = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=self.seed
        )
        rf_selector.fit(X_train, y_train)
        rf_importance = rf_selector.feature_importances_
        
        # Second pass: Use L2-based selection with RF importance as weights
        l2_selector = SelectFromModel(
            LogisticRegression(penalty='l2', solver='liblinear', C=0.01),
            prefit=False,
            threshold='median'
        )
        
        # Combine feature importances
        feature_names = (
            preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols).tolist() +
            numeric_cols
        )
        
        # Weight features by both RF and L2 importance
        combined_importance = 0.7 * rf_importance + 0.3 * np.ones_like(rf_importance)
        
        # Fit selector with weighted features
        X_train_weighted = X_train * combined_importance
        l2_selector.fit(X_train_weighted, y_train)
        
        # Transform features
        X_train_selected = l2_selector.transform(X_train)
        X_val_selected = l2_selector.transform(X_val)
        
        # Initialize grid search with cross-validation
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=cv,
            scoring=revenue_scorer,
            verbose=1,
            n_jobs=1
        )
        
        # Run grid search
        grid_search.fit(X_train_selected, y_train)
        
        # Get best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Evaluate on validation set
        val_score = revenue_scorer(best_model, X_val_selected, y_val)
        
        # Save models and transformers
        self.model = best_model
        self.selector_ = l2_selector
        self.preprocessor_ = preprocessor
        self.rf_selector_ = rf_selector
        
        # Get selected features
        selected_features = np.array(feature_names)[l2_selector.get_support()]
        
        # Calculate feature importance with combined approach
        coef = np.abs(self.model.coef_[0])
        l2_penalty = np.mean(coef ** 2)
        rf_importance_selected = rf_importance[l2_selector.get_support()]
        
        # Combine feature importance scores
        self.feature_importance_ = dict(zip(
            selected_features,
            0.7 * (coef / (1 + l2_penalty)) + 0.3 * rf_importance_selected
        ))
        
        print(f"Best parameters: {best_params}")
        print(f"Best cross-validation score: {best_score:.4f}")
        print(f"Validation score: {val_score:.4f}")
        print(f"L2 penalty: {l2_penalty:.4f}")
        print(f"\nNumber of features selected: {len(selected_features)}")
        print("\nTop 10 most important features:")
        for feature, importance in sorted(
            self.feature_importance_.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]:
            print(f"{feature}: {importance:.4f}")

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """Generate recommendations with enhanced features."""
        # Convert to pandas if they're Spark DataFrames
        if hasattr(log, 'toPandas'):
            log = log.toPandas()
        if hasattr(users, 'toPandas'):
            users = users.toPandas()
        if hasattr(items, 'toPandas'):
            items = items.toPandas()
        if hasattr(user_features, 'toPandas'):
            user_features = user_features.toPandas()
        if hasattr(item_features, 'toPandas'):
            item_features = item_features.toPandas()
        
        # Get user sequences for AR model
        user_sequences = log.groupby('user_idx')
        
        # Prepare results
        recommendations = []
        
        # Generate recommendations for each user
        for user_idx in users['user_idx'].unique():
            # Get user's interaction history
            if user_idx in user_sequences.groups:
                user_log = user_sequences.get_group(user_idx)
                items_sequence = user_log['item_idx'].tolist()
                categories_sequence = user_log['category'].tolist() if 'category' in user_log.columns else []
            else:
                items_sequence = []
                categories_sequence = []
            
            # Get candidate items
            candidate_items = items['item_idx'].tolist()
            if filter_seen_items and items_sequence:
                candidate_items = [item for item in candidate_items if item not in items_sequence]
            
            # Calculate scores for each candidate item
            item_scores = []
            for item in candidate_items:
                # Get AR model probability
                if len(items_sequence) >= self.n:
                    context = items_sequence[-self.n:]
                else:
                    context = [None] * (self.n - len(items_sequence)) + items_sequence
                
                context_key = self._get_ngram_key(context)
                
                # Calculate base probability from n-gram model
                if context_key in self.ngram_counts:
                    context_count = sum(self.ngram_counts[context_key].values())
                    next_item_count = self.ngram_counts[context_key].get(item, 0)
                    ar_prob = (next_item_count + self.smoothing) / (context_count + self.smoothing * len(candidate_items))
                else:
                    item_count = self.item_counts.get(item, 0)
                    ar_prob = (item_count + self.smoothing) / (self.total_items + self.smoothing * len(candidate_items))
                
                # Get item category and price
                item_category = items[items['item_idx'] == item]['category'].iloc[0] if 'category' in items.columns else None
                item_price = items[items['item_idx'] == item]['price'].iloc[0] if 'price' in items.columns else 1.0
                
                # Calculate category preference score
                if item_category and user_idx in self.user_category_preferences:
                    category_pref = self.user_category_preferences[user_idx].get(item_category, 0)
                    category_total = sum(self.user_category_preferences[user_idx].values())
                    category_score = (category_pref + self.smoothing) / (category_total + self.smoothing * len(self.user_category_preferences[user_idx]))
                else:
                    category_score = 0.5  # Neutral score if no category information
                
                # Combine scores with weights
                final_score = 0.7 * ar_prob + 0.3 * category_score
                
                # Calculate expected revenue with category preference
                expected_revenue = final_score * item_price
                item_scores.append((item, expected_revenue))
            
            # Sort by expected revenue and take top k
            item_scores.sort(key=lambda x: x[1], reverse=True)
            top_k_items = item_scores[:k]
            
            # Add to recommendations
            for rank, (item, score) in enumerate(top_k_items, 1):
                recommendations.append({
                    'user_idx': user_idx,
                    'item_idx': item,
                    'relevance': score,
                    'rank': rank
                })
        
        # Convert to DataFrame
        recommendations_df = pd.DataFrame(recommendations)
        
        # Convert back to Spark DataFrame
        recommendations_spark = spark.createDataFrame(recommendations_df)
        
        return recommendations_spark