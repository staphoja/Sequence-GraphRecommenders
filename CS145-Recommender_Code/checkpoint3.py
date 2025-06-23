
# Cell: Define custom recommender template
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

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

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Embedding, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau



class RevenueGCN(nn.Module):
    def __init__(self, user_dim, item_dim, common_dim = 64, hidden_dim = 64, output_dim = 32):
        super(RevenueGCN, self).__init__()

        #Projection layers (because user and item have different dimensions)
        self.user_proj = nn.Linear(user_dim, common_dim)
        self.item_proj = nn.Linear(item_dim, common_dim)

        #GCN Layers:
        self.conv1 = geom_nn.GCNConv(in_channels=common_dim, out_channels = hidden_dim)
        self.conv2 = geom_nn.GCNConv(in_channels = hidden_dim, out_channels = output_dim)
    
    def forward(self, user_tensor, item_tensor, edge_index, edge_weight = None):
        #project features into embeddings:
        user_embedding = self.user_proj(user_tensor)
        item_embedding = self.item_proj(item_tensor)

        x = torch.cat([user_embedding, item_embedding], dim=0)

        #GCN layers:
        x = self.conv1(x, edge_index, edge_weight)
        x = nn.functional.relu(x)
        x = self.conv2(x, edge_index, edge_weight)

        return x




class GCNRecommender:
    def __init__(self, seed):
        self.seed = seed

        self.model = None

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
        pd_log = (
            log.alias('l')
                .join(user_features.alias('u'), on='user_idx', how = 'inner')
                .join(item_features.alias('i'), on='item_idx', how = 'inner')
                .toPandas()
        )

        return pd_log, user_features, item_features
    
    def _preprocess_features(self, features):
        categorical_cols = features.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = features.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        input_cols = categorical_cols + numerical_cols
        
        encoder = OneHotEncoder(handle_unknown='ignore',sparse_output = False)
        scalar = StandardScaler()

        cat_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', encoder)
        ])

        num_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='mean')),
            ('scale', scalar)
        ])

        pipeline = ColumnTransformer(
            transformers = [
                ('cat', cat_pipeline, categorical_cols),
                ('num', num_pipeline, numerical_cols)
            ]
        )

        features = features.reindex(columns=input_cols)
        features_transformed = pipeline.fit_transform(features)

        return features_transformed
    
    def _get_graph_pieces(self, log, user_features = None, item_features = None):
        #Build the dense mappings
        user_ids = np.sort(user_features["user_idx"].unique())
        item_ids = np.sort(item_features["item_idx"].unique())

        uid2nid = {uid: n for n, uid in enumerate(user_ids)}            # 0…U-1
        iid2nid = {iid: n for n, iid in enumerate(item_ids, start=len(user_ids))}

        #Re-order feature frames to match 
        user_features = (
            user_features.copy()
            .assign(__nid__=lambda df: df["user_idx"].map(uid2nid))
            .sort_values("__nid__")
            .drop(columns="__nid__")
        )
        item_features = (
            item_features.copy()
            .assign(__nid__=lambda df: df["item_idx"].map(iid2nid))
            .sort_values("__nid__")
            .drop(columns="__nid__")
        )

        # masking
        mask = log["user_idx"].isin(uid2nid) & log["item_idx"].isin(iid2nid)
        if not mask.all():
            dropped = (~mask).sum()
        log = log[mask]

        log_dense_u = log["user_idx"].map(uid2nid).astype(np.int64).values
        log_dense_i = log["item_idx"].map(iid2nid).astype(np.int64).values

        #Create the edge index
        edge_index  = torch.tensor([log_dense_u, log_dense_i], dtype=torch.long)
        #Create the edge weight which simply connects by relevance values
        edge_weight = torch.tensor(log["relevance"].values, dtype=torch.float32)

        #Call the feature preprocessing
        user_feat_arr = self._preprocess_features(user_features)
        item_feat_arr = self._preprocess_features(item_features)

        user_tensor = torch.tensor(user_feat_arr, dtype=torch.float32)
        item_tensor = torch.tensor(item_feat_arr, dtype=torch.float32)

        return user_tensor, item_tensor, edge_index, edge_weight


    def fit(self, log, user_features = None, item_features = None):
        #Make them into pandas dataframe
        pd_log = log.toPandas()
        if hasattr(user_features, "toPandas"):
            user_features = user_features.toPandas()
        if hasattr(item_features, "toPandas"):
            item_features = item_features.toPandas()
        user_tensor, item_tensor, edge_index, edge_weight = self._get_graph_pieces(pd_log, user_features, item_features)
        #get necessary dimensions
        user_dim = user_tensor.shape[1]
        item_dim = item_tensor.shape[1]
        
        if self.model is None:
            self.model = RevenueGCN(user_dim=user_dim, item_dim=item_dim)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
            self.criterion = torch.nn.MSELoss()

        #Train the model:
        self.model.train()
        self.optimizer.zero_grad()

        # Get node embeddings
        node_embeddings = self.model(user_tensor, item_tensor, edge_index, edge_weight)

        # Split embeddings
        user_embs = node_embeddings[:user_tensor.shape[0]]
        item_embs = node_embeddings[user_tensor.shape[0]:]

        # For each edge in edge_index[0] (user_idx) and edge_index[1] (item_idx)
        # compute dot product between user embedding and item embedding → predicted relevance
        user_edge_emb = user_embs[edge_index[0]]
        item_edge_emb = item_embs[edge_index[1] - user_tensor.shape[0]]  # shift back offset

        # Predicted relevance
        preds = torch.sum(user_edge_emb * item_edge_emb, dim=1)  # [num_edges]
        loss = self.criterion(preds, edge_weight)

        # backward + step
        loss.backward()
        self.optimizer.step()
        

    def predict(self, log, k, users, items, user_features=None, item_features=None, filter_seen_items=True):
        pd_log = log.toPandas()
        users_pd = users.toPandas()
        items_pd = items.toPandas()
        user_feats_pd = user_features.toPandas()
        item_feats_pd = item_features.toPandas()

        price_map = (
            items_pd.set_index("item_idx")["price"].to_dict()
            if "price" in items_pd
            else {}
        )

        # Build graph and compute embeddings
        user_tensor, item_tensor, edge_index, edge_weight = self._get_graph_pieces(
            pd_log, user_feats_pd, item_feats_pd
        )

        #set model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            #get the node embeddings from the model's prediction
            node_embs = self.model(user_tensor, item_tensor, edge_index, edge_weight)

        # Split back into user/item blocks
        n_users = user_tensor.shape[0]
        user_embs = node_embs[:n_users]
        item_embs = node_embs[n_users:]

        # Position look-ups (because rows were sorted by user_idx and item_idx)
        user_order = user_feats_pd.sort_values("user_idx")["user_idx"].tolist()
        item_order = item_feats_pd.sort_values("item_idx")["item_idx"].tolist()
        u_pos = {u: i for i, u in enumerate(user_order)}
        i_pos = {i: j for j, i in enumerate(item_order)}

        # pre-gather past interactions
        hist_by_user = pd_log.groupby("user_idx")
        all_items = items_pd["item_idx"].tolist()

        recommendations = []

        #score and rank
        for uid in users_pd["user_idx"].unique():
            past_items = (
                hist_by_user.get_group(uid)["item_idx"].tolist()
                if uid in hist_by_user.groups
                else []
            )
            cand_items = (
                [it for it in all_items if it not in past_items]
                if filter_seen_items
                else all_items
            )

            u_vec = user_embs[u_pos[uid]]
            scores = []
            for it in cand_items:
                if it not in i_pos:  #safety check
                    continue
                i_vec = item_embs[i_pos[it]]
                pred = torch.dot(u_vec, i_vec).item()  # dot-product relevance
                rev = pred * price_map.get(it, 1.0)   # expected revenue
                scores.append((it, rev))


            #sort the top k
            top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
            for rank, (it, sc) in enumerate(top_k, 1):
                recommendations.append(
                    {"user_idx": uid, "item_idx": it, "relevance": sc, "rank": rank}
                )

        rec_pd  = pd.DataFrame(recommendations)
        rec_spark = spark.createDataFrame(rec_pd)
        return rec_spark