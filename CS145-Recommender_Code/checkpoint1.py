# Cell: Define custom recommender template
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
import random
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import DoubleType, ArrayType
from pyspark.sql.types import IntegerType, StructType, StructField, LongType

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

import sklearn 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sim4rec.utils import pandas_to_spark
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb
from xgboost import XGBClassifier, callback

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import torch_geometric.nn as geom_nn
from torch_geometric.data import Data

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Embedding, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class LogRegRecommender:
    """
    Template class for implementing a custom recommender.
    
    This class provides the basic structure required to implement a recommender
    that can be used with the Sim4Rec simulator. Students should extend this class
    with their own recommendation algorithm.
    """
    
    def __init__(self, seed=None):
        """
        Initialize recommender.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.model = None
        self.categorical_cols_ = None
        self.numeric_cols_ = None
        self.input_columns_ = None
        self.preprocessor_ = None
        self.feature_importance_ = None
        self.user_encoders_ = {}
        self.item_encoders_ = {}
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

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
        # User interaction counts and statistics
        user_stats = log_df.groupBy("user_idx").agg(
            sf.count("*").alias("user_interaction_count"),
            sf.avg("relevance").alias("user_avg_relevance"),
            sf.stddev("relevance").alias("user_relevance_std"),
            sf.max("relevance").alias("user_max_relevance")
        )
        
        # Item popularity and statistics
        item_stats = log_df.groupBy("item_idx").agg(
            sf.count("*").alias("item_popularity"),
            sf.avg("relevance").alias("item_avg_relevance"),
            sf.stddev("relevance").alias("item_relevance_std"),
            sf.max("relevance").alias("item_max_relevance")
        )
        
        # Category-level statistics
        if 'category' in item_features.columns:
            category_stats = log_df.join(
                item_features.select("item_idx", "category"),
                on="item_idx"
            ).groupBy("category").agg(
                sf.avg("relevance").alias("category_avg_relevance"),
                sf.count("*").alias("category_popularity")
            )
            
            # Join category stats back to items
            item_features = item_features.join(
                category_stats,
                on="category",
                how="left"
            )
        
        return user_stats, item_stats
    
    def fit(self, log, user_features=None, item_features=None):
        """Train the recommender model with enhanced features and preprocessing."""
        # Create aggregate features
        user_stats, item_stats = self._create_aggregate_features(log, user_features, item_features)
        
        # Join features with user and item data
        user_features = user_features.join(user_stats, on="user_idx", how="left")
        item_features = item_features.join(item_stats, on="item_idx", how="left")
        
        # Add prefixes to features
        user_features = user_features.select(
            [sf.col("user_idx")] +
            [sf.col(c).alias(f"u_{c}") for c in user_features.columns if c != "user_idx"]
        )
        item_features = item_features.select(
            [sf.col("item_idx")] +
            [sf.col(c).alias(f"i_{c}") for c in item_features.columns if c != "item_idx"]
        )
        
        # Join all data
        joined = (
            log.alias("l")
               .join(user_features.alias("u"), on="user_idx", how="inner")
               .join(item_features.alias("i"), on="item_idx", how="inner")
        )
        
        # Convert to pandas
        joined_pd = joined.toPandas()
        
        # Create interaction features
        feature_df = joined_pd.copy()  # Keep all columns including IDs
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
        y = (joined_pd["relevance"] > 0).astype(int)
        
        # Enhanced parameter grid for Elastic Net
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['elasticnet'],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'solver': ['saga'],
            'max_iter': [2000],
            'tol': [1e-3],
            'class_weight': ['balanced']
        }
        
        base_model = LogisticRegression(
            random_state=self.seed,
            max_iter=2000,
            tol=1e-3,
            warm_start=True
        )
        
        # Use stratified cross-validation
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        
        grid_search = GridSearchCV(
            base_model, 
            param_grid, 
            cv=cv,
            scoring='f1',
            verbose=1,
            n_jobs=1
        )
        
        # Save preprocessor
        self.preprocessor_ = preprocessor
        
        # Get feature names after transformation
        feature_names = (
            preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols).tolist() +
            numeric_cols
        )
        
        # Add feature selection
        from sklearn.feature_selection import SelectFromModel
        selector = SelectFromModel(
            LogisticRegression(penalty='l1', solver='liblinear', C=0.1),
            prefit=False,
            threshold='median'
        )
        
        # Fit selector and transform features
        X_transformed_selected = selector.fit_transform(X_transformed, y)
        
        # Fit the model
        grid_search.fit(X_transformed_selected, y)
        
        # Save models and transformers
        self.model = grid_search.best_estimator_
        self.selector_ = selector
        
        # Get feature names after selection
        selected_features = np.array(feature_names)[selector.get_support()]
        
        # Calculate feature importance
        self.feature_importance_ = dict(zip(
            selected_features,
            np.abs(self.model.coef_[0])
        ))
        
        print(f"Best parameters: {grid_search.best_params_}")
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
        # Create aggregate features
        user_stats, item_stats = self._create_aggregate_features(log, user_features, item_features)
        
        # Join features
        user_features = user_features.join(user_stats, on="user_idx", how="left")
        item_features = item_features.join(item_stats, on="item_idx", how="left")
        
        # Add prefixes
        user_features = user_features.select(
            [sf.col("user_idx")] +
            [sf.col(c).alias(f"u_{c}") for c in user_features.columns if c != "user_idx"]
        )
        item_features = item_features.select(
            [sf.col("item_idx")] +
            [sf.col(c).alias(f"i_{c}") for c in item_features.columns if c != "item_idx"]
        )
        
        # Create recommendations
        recs = users.crossJoin(items)
        
        if filter_seen_items and log is not None:
            seen = log.select("user_idx", "item_idx")
            recs = recs.join(seen, on=["user_idx", "item_idx"], how="left_anti")
        
        recs = (
            recs.alias("r")
                .join(user_features.alias("u"), on="user_idx", how="inner")
                .join(item_features.alias("i"), on="item_idx", how="inner")
        )
        
        # Convert to pandas
        recs_pd = recs.toPandas()
        ids = recs_pd[["user_idx", "item_idx"]].copy()
        
        # Create interaction features
        feature_df = recs_pd.reindex(columns=self.input_columns_)
        feature_df = self._create_interaction_features(feature_df)
        
        # Handle missing values and types
        for col in self.categorical_cols_:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].astype(str)
                feature_df[col] = feature_df[col].fillna('missing')
            
        for col in self.numeric_cols_:
            if col in feature_df.columns:
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce').fillna(0)
        
        # Ensure all required columns are present
        required_columns = set(self.categorical_cols_ + self.numeric_cols_)
        missing_columns = required_columns - set(feature_df.columns)
        for col in missing_columns:
            if col in self.categorical_cols_:
                feature_df[col] = 'missing'
            else:
                feature_df[col] = 0
        
        # Transform and predict
        X_rec = self.preprocessor_.transform(feature_df)
        X_rec_selected = self.selector_.transform(X_rec)
        preds = self.model.predict_proba(X_rec_selected)[:, 1]
        
        # Create final recommendations
        scored = ids.copy()
        scored["relevance"] = preds
        
        scored_spark = (
            spark.createDataFrame(scored)
                 .withColumn("relevance", sf.col("relevance"))
        )
        
        window = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
        ranked = (
            scored_spark
            .withColumn("rank", sf.row_number().over(window))
            .filter(sf.col("rank") <= k)
            .drop("rank")
        )
        
        return ranked
    


class DecisionTreeRecommender:
    def __init__(self, max_depth=5, min_samples_leaf=10, seed = None):
        self.model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        self.feature_cols = None
        self.user_cols = []
        self.item_cols = []
        self.seed = seed
        
    def helper_prepare_features(self, df, fit=False):
        # helps features be one-hot encoded and with column alignment
        df = pd.get_dummies(df)
        if fit:
            self.feature_cols = df.columns.tolist()
        else:
            for col in self.feature_cols:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.feature_cols]
        return df

    def fit(self, log, user_features=None, item_features=None):
        log_pd = log.toPandas()
        user_pd = user_features.toPandas()
        item_pd = item_features.toPandas()

        # this lets us merge interactions with user/item features
        df = log_pd.merge(user_pd, on="user_idx")
        df = df.merge(item_pd, on="item_idx")

        self.user_cols = [c for c in user_pd.columns if c != "user_idx"]
        self.item_cols = [c for c in item_pd.columns if c != "item_idx"]
        feature_df = df[self.user_cols + self.item_cols]

        X = self.helper_prepare_features(feature_df, fit=True)
        y = df["relevance"]

        self.model.fit(X, y)

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        users_pd = user_features.toPandas()
        items_pd = item_features.toPandas()

        # this creates user-item pairs
        user_item_pairs = users_pd.assign(key=1).merge(items_pd.assign(key=1), on="key").drop("key", axis=1)
        feature_df = user_item_pairs[self.user_cols + self.item_cols]

        X_pred = self.helper_prepare_features(feature_df, fit=False)
        user_item_pairs["relevance"] = self.model.predict_proba(X_pred)[:, 1]

        if filter_seen_items and log is not None:
            seen = log.select("user_idx", "item_idx").toPandas()
            seen["seen"] = 1
            user_item_pairs = user_item_pairs.merge(seen, how="left", on=["user_idx", "item_idx"])
            user_item_pairs = user_item_pairs[user_item_pairs["seen"] != 1].drop(columns=["seen"])

        # rank and return top-k
        user_item_pairs["rank"] = user_item_pairs.groupby("user_idx")["relevance"]\
                                                .rank(method="first", ascending=False)
        top_k = user_item_pairs[user_item_pairs["rank"] <= k]

        return spark.createDataFrame(top_k[["user_idx", "item_idx", "relevance"]])
    


class GradientBoost:
    """
    Template class for implementing a custom recommender.
    
    This class provides the basic structure required to implement a recommender
    that can be used with the Sim4Rec simulator. Students should extend this class
    with their own recommendation algorithm.
    """
    
    def __init__(self, seed=None, optimize=True):
        """
        Initialize recommender.
        
        Args:
            seed: Random seed for reproducibility
        """
        # Add your initialization logic here
        self.seed = seed
        self.optimize = optimize
        self.categorical_cols = None
        self.numerical_cols = None
        self.input_cols = None
        self.pipeline = None
        self.model = None
        self.best_params = None
        self.encoder = OneHotEncoder(handle_unknown='ignore',sparse_output = False)
        self.scalar = StandardScaler()

    def _create_features(self, features):
        #average category price
        if 'i_category' in features.columns and 'i_price' in features.columns:
            features['avg_category_price'] = features.groupby('i_category')['i_price'].transform('mean')
        
        #get the average price spent by user
        if 'user_idx' in features.columns and 'i_price' in features.columns:
            features['user_avg_price'] = features.groupby('user_idx')['i_price'].transform('mean')

        #get the price of the item compared to the average amount spent by the users
        if 'user_avg_price' in features.columns and 'i_price' in features.columns:
            features['price_vs_user_mean'] = features['i_price'] - features['user_avg_price']
        
        return features

    
    def _setup_df(self, log, user_features = None, item_features = None):
        #add 'u_' prefix to the user features, helps with clarity
        user_features = user_features.select(
            [sf.col('user_idx')] + 
            [sf.col(c).alias(f'u_{c}') for c in user_features.columns if c != 'user_idx']
        )

        #add 'i_' prefix to the item features, helps with clarity
        item_features = item_features.select(
            [sf.col('item_idx')] + 
            [sf.col(c).alias(f'i_{c}') for c in item_features.columns if c != 'item_idx']
        )

        pd_log = (
            log.alias('l')
                .join(user_features.alias('u'), on='user_idx', how = 'inner')
                .join(item_features.alias('i'), on='item_idx', how = 'inner')
                .toPandas()
        )

        return pd_log, user_features, item_features
    
    def _preprocess_features(self, features):
        self.categorical_cols = features.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_cols = features.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        self.input_cols = self.categorical_cols + self.numerical_cols
        

        cat_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', self.encoder)
        ])

        num_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='mean')),
            ('scale', self.scalar)
        ])

        self.pipeline = ColumnTransformer(
            transformers = [
                ('cat', cat_pipeline, self.categorical_cols),
                ('num', num_pipeline, self.numerical_cols)
            ]
        )

        features = features.reindex(columns=self.input_cols)
        features_transformed = self.pipeline.fit_transform(features)

        return features_transformed

    def _get_best_model(self, X,y):
        print("Optimized Version")
        param_grid = {
                "n_estimators": [100, 200], #100
                "learning_rate": [0.3], #0.3
                "max_depth": [6, 10], #6
                "min_child_weight": [1], #1
            }
        
        #Best params so far: 31% increase, commented on each side
        
        base_model = XGBClassifier(
                    booster='gbtree',
                    random_state = self.seed, 
                    tree_method = 'hist',
                    eval_metric='logloss',
                    n_jobs = 4)
        grid_search = GridSearchCV(
            base_model, param_grid, 
            cv = 5, 
            scoring = 'neg_log_loss',
            n_jobs = 1)
        grid_search.fit(X, y,verbose=False)

        self.best_params = grid_search.best_params_

        return grid_search.best_estimator_

    def fit(self, log, user_features=None, item_features=None):
        """
        Train the recommender model based on interaction history.
        
        Args:
            log: Interaction log with user_idx, item_idx, and relevance columns
            user_features: User features dataframe (optional)
            item_features: Item features dataframe (optional)
        """
        # Implement your training logic here
        # For example:
        #  1. Extract relevant features from user_features and item_features
        #  2. Learn user preferences from the log
        #  3. Build item similarity matrices or latent factor models
        #  4. Store learned parameters for later prediction
        if user_features and item_features:
            pd_log, user_features, item_features = self._setup_df(log, user_features, item_features)
            pd_log = self._create_features(pd_log)
            features = pd_log.drop(columns=['user_idx', 'item_idx', 'relevance'])

            X = self._preprocess_features(features)
            y = pd_log['relevance'].values

            if self.model is None:
                if self.optimize:
                    self.model = self._get_best_model(X,y)
                    self.model = XGBClassifier(
                                    **self.best_params,
                                    random_state=self.seed,
                                    booster='gbtree',
                                    tree_method='hist',
                                    eval_metric='logloss',
                                    early_stopping_rounds = 5, #Does not even happen, because the 25 estimators is already performing well
                                    n_jobs=4)
                else:
                    self.model = XGBClassifier(
                            booster='gbtree',
                            random_state = self.seed, 
                            tree_method = 'hist',
                            eval_metric='logloss',
                            n_jobs = 4)
                    
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=self.seed)
                    
            self.model.fit(X_train,y_train, 
                        eval_set = [(X_test, y_test)],
                        verbose=False)
                    
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """
        Generate recommendations for users.
        
        Args:
            log: Interaction log with user_idx, item_idx, and relevance columns
            k: Number of items to recommend
            users: User dataframe
            items: Item dataframe
            user_features: User features dataframe (optional)
            item_features: Item features dataframe (optional)
            filter_seen_items: Whether to filter already seen items
            
        Returns:
            DataFrame: Recommendations with user_idx, item_idx, and relevance columns
        """
        # Implement your recommendation logic here
        # For example:
        #  1. Extract relevant features for prediction
        #  2. Calculate relevance scores for each user-item pair
        #  3. Rank items by relevance and select top-k
        #  4. Return a dataframe with columns: user_idx, item_idx, relevance
        candidate_df = users.crossJoin(items)

        if filter_seen_items and log is not None:
            seen = log.select("user_idx", "item_idx").distinct()
            candidate_df = candidate_df.join(seen, ["user_idx", "item_idx"], "left_anti")

        candidate_pd, _, _ = self._setup_df(candidate_df, user_features, item_features)

        candidate_pd = self._create_features(candidate_pd)

        meta_pd = candidate_pd[["user_idx", "item_idx"]].copy()

        features = candidate_pd.drop(
                        columns=[c for c in ["__iter", "relevance"] if c in candidate_pd.columns],
                        errors="ignore"
                )
    
        features = features.reindex(columns=self.input_cols, fill_value=np.nan)

        X = self.pipeline.transform(features)

        meta_pd["relevance"] = self.model.predict_proba(X)[:, 1]

        #Rank and take top k
        topk_pd = (
            meta_pd.sort_values(["user_idx", "relevance"], ascending=[True, False])
                .groupby("user_idx")
                .head(k)
            )
    
        return pandas_to_spark(topk_pd[["user_idx", "item_idx", "relevance"]])


class kNearestRecommender:
    """
    k Nearest Neighbors recommender system.
    Recommends items based on kNN classifier.
    """
    
    def __init__(self, metric='euclidean', k_range=range(3, 25), seed=None):
        """
        Initialize recommender.
        
        Args:
            seed: Random seed for reproducibility
            metric: Chosen distance metric for determining neighbors {"euclidean", "manhattan", "cosine", "minkowski"}
            k_range: Range of k values to test
        """
        self.seed = seed
        self.metric = metric
        self.k_range = k_range
        self.feature_cols = None
        self.categorical_cols = []
        self.numerical_cols = []
        self.best_k = None
        self.best_metric = None
        self.preprocess = None
        self.model = None
        self.spark = None

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def process_cols(self, data):
        self.feature_cols = data.columns.tolist()

        nonfeature_cols = ["user_idx", "relevance"]

        self.feature_cols = [col for col in data.columns if col not in nonfeature_cols]

        for col in data[self.feature_cols].columns:
            if data[col].dtype in ['object', 'bool', 'category']:
                self.categorical_cols.append(col)
            else:
                self.numerical_cols.append(col) 

        return data
    
    def create_pipeline(self):
        transformers = []

        if self.numerical_cols:
            numerical_transformer = Pipeline(
                steps=[
                    ('scaler', StandardScaler())
                ]
            )
            transformers.append(('num', numerical_transformer, self.numerical_cols))
        
        if self.categorical_cols:
            categorical_transformer = Pipeline(
                steps=[
                    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
                ]
            )
            transformers.append(('cat', categorical_transformer, self.categorical_cols))
        
        if transformers:
            self.preprocess = ColumnTransformer(transformers=transformers)
        else:
            self.preprocess = None

    def select_k(self, X, y):
        knn = KNeighborsRegressor(metric=self.metric, weights="distance")
        pipe = Pipeline([("prep", self.preprocess), ("knn", knn)])
        param_grid = {"knn__n_neighbors": list(self.k_range)}
        gs = GridSearchCV(pipe,
                        param_grid=param_grid,
                        scoring="neg_root_mean_squared_error",
                        cv=3,
                        n_jobs=-1)
        gs.fit(X, y)

        self.best_k = gs.best_params_["knn__n_neighbors"]
        self.model = gs.best_estimator_
        self.train_x = X
        self.train_y = y
    
    def fit(self, log, user_features=None, item_features=None):
        """
        Train the recommender model based on interaction history.
        """
        self.spark = log.sparkSession
        
        # Convert to pandas
        log_pd = log.toPandas()
        if user_features is not None:
            user_pd = user_features.toPandas()
        else:
            user_pd = pd.DataFrame(columns=["user_idx"])
        if item_features is not None:
            item_pd = item_features.toPandas()
        else:
            item_pd = pd.DataFrame(columns=["item_idx"])
        
        # Merge data
        data = log_pd.merge(user_pd, on='user_idx', how='left').merge(item_pd, on='item_idx', how='left')

        data = self.process_cols(data)

        self.create_pipeline()
        X = data[self.feature_cols]
        y = data['relevance'].astype(np.int32)

        # Select best k
        self.select_k(X, y)

        # Fit and transform the preprocessing pipeline
        if self.preprocess:
            X_transformed = self.preprocess.fit_transform(X)
        else:
            X_transformed = X.values

        # Fit the model with transformed data
        self.model = NearestNeighbors(n_neighbors=self.best_k,
                                    metric=self.metric,
                                    n_jobs=-1)
        
        self.model.fit(X_transformed)
        self.train_x = X_transformed
        self.train_y = y
    
    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        """
        Generate recommendations for users.
        
        Args:
            log: Interaction log with user_idx, item_idx, and relevance columns
            k: Number of items to recommend
            users: User dataframe
            items: Item dataframe
            user_features: User features dataframe (optional)
            item_features: Item features dataframe (optional)
            filter_seen_items: Whether to filter already seen items
        
        Returns:
            DataFrame: Recommendations with user_idx, item_idx, and relevance columns
        """
        users_pd = users.toPandas()
        items_pd = items.toPandas()
        
        if user_features is not None:
            user_feat_pd = user_features.toPandas()
        else:
            user_feat_pd = pd.DataFrame({'user_idx': users_pd['user_idx']})
            
        if item_features is not None:
            item_feat_pd = item_features.toPandas()
        else:
            item_feat_pd = pd.DataFrame({'item_idx': items_pd['item_idx']})
        
        # Create all user-item pairs
        user_item_pairs = pd.DataFrame(
            [(user, item) for user in users_pd['user_idx'] for item in items_pd['item_idx']],
            columns=['user_idx', 'item_idx']
        )
        
        X_test_data = user_item_pairs.merge(user_feat_pd, on='user_idx', how='left').merge(item_feat_pd, on='item_idx', how='left')
        
        for col in self.feature_cols:
            if col not in X_test_data.columns:
                if col in self.categorical_cols:
                    X_test_data[col] = 'unknown'
                else:
                    X_test_data[col] = 0

        # Add missing columns with default values to match training features
        for col in self.feature_cols:
            if col not in X_test_data.columns:
                X_test_data[col] = 0
        
        for col in self.numerical_cols:
            if col in X_test_data.columns:
                X_test_data[col] = pd.to_numeric(X_test_data[col], errors='coerce').fillna(0)
        
        # Ensure we only use the same features as training
        X_test_features = X_test_data[self.feature_cols]
        
        # Transform the test data
        if self.preprocess:
            X_test = self.preprocess.transform(X_test_features)
        else:
            X_test = X_test_features.values
        
        # Get nearest neighbors
        distances, indices = self.model.kneighbors(X_test)
        
        # Calculate weighted relevance
        train_y_array = self.train_y.values if hasattr(self.train_y, 'values') else self.train_y
        
        weights = 1 / (distances + 1e-10)
        neighbor_relevances = train_y_array[indices]
        weighted_relevance = np.sum(weights * neighbor_relevances, axis=1) / np.sum(weights, axis=1)
        
        # Create recommendations dataframe
        recommendations = X_test_data[['user_idx', 'item_idx']].copy()
        recommendations['relevance'] = np.round(weighted_relevance).astype(np.int32)
        
        # Filter seen items if requested
        if filter_seen_items and log is not None:
            log_pd = log.toPandas() if hasattr(log, 'toPandas') else log
            seen_items = set(zip(log_pd['user_idx'], log_pd['item_idx']))
            recommendations = recommendations[~recommendations.apply(lambda x: (x['user_idx'], x['item_idx']) in seen_items, axis=1)]
        
        # Get top k recommendations per user
        recommendations = (recommendations
                        .sort_values(['user_idx', 'relevance'], ascending=[True, False])
                        .groupby('user_idx')
                        .head(k)
                        .reset_index(drop=True))
        
        # Ensure correct data types
        recommendations['user_idx'] = recommendations['user_idx'].astype(int)
        recommendations['item_idx'] = recommendations['item_idx'].astype(int)
        recommendations['relevance'] = recommendations['relevance'].astype(int)
        
        # Convert back to Spark DataFrame
        schema = StructType([
            StructField("user_idx", LongType(), True),
            StructField("item_idx", LongType(), True),
            StructField("relevance", LongType(), True)
        ])
        
        spark_df = self.spark.createDataFrame(recommendations, schema)
        
        return spark_df