# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 17:49:40 2025

@author: Asus
"""

# AI-Powered Personal Health Analytics and Recommendation System
# Using Step and Sleep Data
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

# Machine Learning imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, mean_squared_error
import xgboost as xgb
from statsmodels.tsa.seasonal import seasonal_decompose
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

# Dashboard libraries
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# For model explainability (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

class HealthAnalyticsSystem:
    """
    AI-Powered Personal Health Analytics and Recommendation System using 
    Step and Sleep Data
    """
    
    def __init__(self):
        self.data = None
        self.preprocessed_data = None
        self.features = None
        self.labels = None
        self.models = {}
        self.cluster_model = None
        self.recommendation_engine = None
        self.scaler = None
    
    def load_data(self, file_path, data_source="fitbit"):
        """
        Load data from different sources (Fitbit, Apple Health, MESA)
        
        Parameters:
        -----------
        file_path : str
            Path to the data file
        data_source : str
            Source of the data ('fitbit', 'apple_health', 'mesa')
        """
        if data_source.lower() == "fitbit":
            self.data = pd.read_csv(file_path)
            # Ensure datetime formatting
            if 'date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['date'])
                self.data.set_index('date', inplace=True)
            elif 'datetime' in self.data.columns:
                self.data['datetime'] = pd.to_datetime(self.data['datetime'])
                self.data.set_index('datetime', inplace=True)
                
        elif data_source.lower() == "apple_health":
            self.data = pd.read_csv(file_path)
            # Apple Health specific transformations
            if 'start' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['start'])
                self.data.set_index('date', inplace=True)
                
        elif data_source.lower() == "mesa":
            self.data = pd.read_csv(file_path)
            # MESA dataset specific transformations
            if 'sleep_date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['sleep_date'])
                self.data.set_index('date', inplace=True)
                
        else:
            raise ValueError("Unsupported data source. Use 'fitbit', 'apple_health', or 'mesa'.")
        
        print(f"Data loaded successfully with shape: {self.data.shape}")
        return self.data.head()
    
    def clean_data(self):
        """Clean and preprocess the data"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")
        
        # Make a copy of the data to avoid modifying the original
        df = self.data.copy()
        
        # Handle missing values
        # For numerical columns: fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # For categorical columns: fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        
        # Handle outliers using IQR method for step and sleep data
        for col in ['steps', 'sleep_duration'] if 'steps' in df.columns and 'sleep_duration' in df.columns else []:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
        
        # Ensure the index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
                df.set_index(date_cols[0], inplace=True)
            else:
                raise ValueError("No date column found for time-series structuring.")
        
        # Sort by date
        df = df.sort_index()
        
        self.preprocessed_data = df
        print("Data cleaned and preprocessed successfully.")
        return df.head()
    
    def perform_eda(self, save_plots=False, plot_dir='./plots'):
        """
        Perform Exploratory Data Analysis (EDA)
        
        Parameters:
        -----------
        save_plots : bool
            Whether to save the plots to disk
        plot_dir : str
            Directory to save the plots
        """
        if self.preprocessed_data is None:
            raise ValueError("No preprocessed data available. Please run clean_data() first.")
        
        import os
        if save_plots and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        df = self.preprocessed_data
        
        # 1. Summary statistics
        summary = df.describe()
        print("Summary Statistics:")
        print(summary)
        
        # 2. Correlation matrix and heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')
        if save_plots:
            plt.savefig(f"{plot_dir}/correlation_heatmap.png")
        plt.close()
        
        # 3. Distribution of key metrics
        for col in ['steps', 'sleep_duration', 'calories'] if all(c in df.columns for c in ['steps', 'sleep_duration', 'calories']) else df.select_dtypes(include=[np.number]).columns[:3]:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
            if save_plots:
                plt.savefig(f"{plot_dir}/{col}_distribution.png")
            plt.close()
        
        # 4. Time series plot for steps and sleep
        if all(col in df.columns for col in ['steps', 'sleep_duration']):
            plt.figure(figsize=(14, 7))
            plt.subplot(2, 1, 1)
            plt.plot(df.index, df['steps'], label='Steps')
            plt.title('Steps Over Time')
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(df.index, df['sleep_duration'], label='Sleep Duration', color='orange')
            plt.title('Sleep Duration Over Time')
            plt.legend()
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(f"{plot_dir}/time_series_plot.png")
            plt.close()
        
        # 5. Day of week analysis
        df['day_of_week'] = df.index.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Steps by day of week
        if 'steps' in df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='day_of_week', y='steps', data=df, order=day_order)
            plt.title('Steps by Day of Week')
            plt.xticks(rotation=45)
            if save_plots:
                plt.savefig(f"{plot_dir}/steps_by_day.png")
            plt.close()
        
        # Sleep by day of week
        if 'sleep_duration' in df.columns:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='day_of_week', y='sleep_duration', data=df, order=day_order)
            plt.title('Sleep Duration by Day of Week')
            plt.xticks(rotation=45)
            if save_plots:
                plt.savefig(f"{plot_dir}/sleep_by_day.png")
            plt.close()
        
        # 6. Time series decomposition if enough data
        for col in ['steps', 'sleep_duration'] if all(c in df.columns for c in ['steps', 'sleep_duration']) else []:
            # Need enough data points for decomposition
            if len(df) >= 14:  # At least 2 weeks of data
                try:
                    # Resample to ensure regular time series
                    ts = df[col].resample('D').mean()
                    # Fill missing values for decomposition
                    ts = ts.fillna(ts.median())
                    
                    decomposition = seasonal_decompose(ts, model='additive', period=7)
                    
                    plt.figure(figsize=(14, 10))
                    plt.subplot(4, 1, 1)
                    plt.plot(ts)
                    plt.title(f'{col} - Original')
                    
                    plt.subplot(4, 1, 2)
                    plt.plot(decomposition.trend)
                    plt.title('Trend')
                    
                    plt.subplot(4, 1, 3)
                    plt.plot(decomposition.seasonal)
                    plt.title('Seasonality')
                    
                    plt.subplot(4, 1, 4)
                    plt.plot(decomposition.resid)
                    plt.title('Residuals')
                    
                    plt.tight_layout()
                    if save_plots:
                        plt.savefig(f"{plot_dir}/{col}_decomposition.png")
                    plt.close()
                except Exception as e:
                    print(f"Could not perform decomposition for {col}: {e}")
        
        print("EDA completed successfully.")
        return correlation_matrix
    
    def engineer_features(self):
        """
        Engineer features for modeling
        """
        if self.preprocessed_data is None:
            raise ValueError("No preprocessed data available. Please run clean_data() first.")
        
        df = self.preprocessed_data.copy()
        
        # 1. Date-based features
        df['day_of_week'] = df.index.dayofweek  # 0=Monday, 6=Sunday
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df['month'] = df.index.month
        df['day'] = df.index.day
        
        # 2. Rolling window features (7-day)
        if 'steps' in df.columns:
            df['steps_7d_avg'] = df['steps'].rolling(window=7, min_periods=1).mean()
            df['steps_7d_std'] = df['steps'].rolling(window=7, min_periods=1).std()
        
        if 'sleep_duration' in df.columns:
            df['sleep_7d_avg'] = df['sleep_duration'].rolling(window=7, min_periods=1).mean()
            df['sleep_7d_std'] = df['sleep_duration'].rolling(window=7, min_periods=1).std()
        
        # 3. Lag features
        if 'steps' in df.columns:
            df['steps_1d_lag'] = df['steps'].shift(1)
            df['steps_2d_lag'] = df['steps'].shift(2)
            df['steps_7d_lag'] = df['steps'].shift(7)
        
        if 'sleep_duration' in df.columns:
            df['sleep_1d_lag'] = df['sleep_duration'].shift(1)
            df['sleep_2d_lag'] = df['sleep_duration'].shift(2)
            df['sleep_7d_lag'] = df['sleep_duration'].shift(7)
        
        # 4. Interaction features
        if all(col in df.columns for col in ['steps', 'sleep_duration']):
            # Previous day's steps and current sleep
            df['steps_sleep_ratio'] = df['steps'] / (df['sleep_duration'] + 1)  # Add 1 to avoid division by zero
            df['steps_1d_sleep_impact'] = df['steps_1d_lag'] / (df['sleep_duration'] + 1)
        
        # 5. Sleep efficiency (if relevant fields exist)
        if all(col in df.columns for col in ['sleep_duration', 'time_in_bed']):
            df['sleep_efficiency'] = df['sleep_duration'] / df['time_in_bed']
        
        # 6. Sleep quality label (binary)
        if 'sleep_duration' in df.columns:
            # Good sleep: >= 7 hours
            df['good_sleep'] = (df['sleep_duration'] >= 7).astype(int)
            
            # If heart rate data is available, refine the definition
            if 'resting_heart_rate' in df.columns:
                # Good sleep: >= 7 hours AND low heart rate
                rhr_median = df['resting_heart_rate'].median()
                df['good_sleep'] = ((df['sleep_duration'] >= 7) & 
                                    (df['resting_heart_rate'] <= rhr_median)).astype(int)
        
        # 7. Fill missing values created by lag features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].median())
        
        self.features = df
        print(f"Feature engineering completed. Dataset now has {df.shape[1]} features.")
        return df.head()
    
    def prepare_for_modeling(self, target_variable='good_sleep', test_size=0.2, time_series_split=True):
        """
        Prepare data for modeling
        
        Parameters:
        -----------
        target_variable : str
            Target variable for prediction
        test_size : float
            Proportion of data to use for testing
        time_series_split : bool
            Whether to use time series split instead of random split
        """
        if self.features is None:
            raise ValueError("No feature data available. Please run engineer_features() first.")
        
        # Check if target variable exists
        if target_variable not in self.features.columns:
            raise ValueError(f"Target variable '{target_variable}' not found in features.")
        
        # Get the data
        df = self.features.copy()
        
        # Define X and y
        y = df[target_variable]
        
        # Drop the target and any other unwanted columns
        cols_to_drop = [target_variable, 'day_of_week'] if 'day_of_week' in df.columns else [target_variable]
        X = df.drop(columns=cols_to_drop)
        
        # Handle any non-numeric columns
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X.drop(columns=[col], inplace=True)
        
        # Normalize/standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Split data
        if time_series_split:
            # Use TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            for train_index, test_index in tscv.split(X_scaled):
                X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            print("Data split using TimeSeriesSplit")
        else:
            # Random split (less appropriate for time series)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42
            )
            print(f"Data split randomly with test_size={test_size}")
            
        self.model_features = X.columns.tolist()
        return X_train, X_test, y_train, y_test, X.columns
    
    def train_sleep_prediction_model(self, X_train, y_train, model_type='random_forest'):
        """
        Train a sleep quality prediction model
        
        Parameters:
        -----------
        X_train : DataFrame
            Training features
        y_train : Series
            Training target
        model_type : str
            Type of model to train ('random_forest', 'xgboost')
        """
        print(f"Training {model_type} model for sleep prediction...")
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            raise ValueError("Unsupported model type. Use 'random_forest' or 'xgboost'.")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Store the model
        self.models['sleep_prediction'] = model
        
        print("Sleep prediction model trained successfully.")
        return model
    
    def train_step_prediction_model(self, X_train, y_train, model_type='random_forest'):
        """
        Train a step count prediction model
        
        Parameters:
        -----------
        X_train : DataFrame
            Training features
        y_train : Series
            Training target (step count)
        model_type : str
            Type of model to train ('random_forest', 'xgboost')
        """
        print(f"Training {model_type} model for step prediction...")
        
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        elif model_type == 'xgboost':
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                objective='reg:squarederror'
            )
        else:
            raise ValueError("Unsupported model type. Use 'random_forest' or 'xgboost'.")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Store the model
        self.models['step_prediction'] = model
        
        print("Step prediction model trained successfully.")
        return model
    
    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a trained model
        
        Parameters:
        -----------
        model_name : str
            Name of the model to evaluate ('sleep_prediction', 'step_prediction')
        X_test : DataFrame
            Test features
        y_test : Series
            Test target
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Please train the model first.")
        
        model = self.models[model_name]
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate based on model type
        if model_name == 'sleep_prediction':
            # Classification metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            # ROC-AUC if the model supports predict_proba
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob)
            else:
                roc_auc = None
            
            # Feature importances
            if hasattr(model, 'feature_importances_'):
                feature_imp = pd.DataFrame({
                    'Feature': X_test.columns,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
            else:
                feature_imp = None
            
            print(f"Model: {model_name}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            if roc_auc is not None:
                print(f"ROC-AUC: {roc_auc:.4f}")
            
            # SHAP values for model explainability
            if SHAP_AVAILABLE:
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test)
                    
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_test, plot_type="bar")
                    plt.title(f"SHAP Feature Importance for {model_name}")
                    plt.tight_layout()
                    plt.savefig(f"{model_name}_shap_importance.png")
                    plt.close()
                except Exception as e:
                    print(f"Could not generate SHAP values: {e}")
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'feature_importance': feature_imp
            }
        
        elif model_name == 'step_prediction':
            # Regression metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # R-squared
            r2 = model.score(X_test, y_test)
            
            # Feature importances
            if hasattr(model, 'feature_importances_'):
                feature_imp = pd.DataFrame({
                    'Feature': X_test.columns,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
            else:
                feature_imp = None
            
            print(f"Model: {model_name}")
            print(f"MSE: {mse:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"R² Score: {r2:.4f}")
            
            # SHAP values for model explainability
            if SHAP_AVAILABLE:
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test)
                    
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_test, plot_type="bar")
                    plt.title(f"SHAP Feature Importance for {model_name}")
                    plt.tight_layout()
                    plt.savefig(f"{model_name}_shap_importance.png")
                    plt.close()
                except Exception as e:
                    print(f"Could not generate SHAP values: {e}")
            
            return {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'feature_importance': feature_imp
            }
        
        else:
            raise ValueError(f"Unknown model type for evaluation: {model_name}")

    #Modified
    def perform_clustering(self, n_clusters=3, clustering_method='kmeans'):
        
        """
        Perform clustering on the feature data for personalization
    
        Parameters:
        -----------
        n_clusters : int
            Number of clusters for KMeans
        clustering_method : str
            Method for clustering ('kmeans', 'dbscan')
        """
        if self.features is None:
            raise ValueError("No feature data available. Please run engineer_features() first.")
    
        # Get relevant features for clustering
        df = self.features.copy()
    
        # Select features for clustering
        feature_cols = []
        for pattern in ['steps', 'sleep', 'heart', 'calories', 'avg', 'std']:
            feature_cols.extend([col for col in df.columns if pattern in col.lower()])
    
        # Ensure we have the core features
        core_features = ['steps', 'sleep_duration']
        feature_cols.extend([col for col in core_features if col in df.columns])
    
        # Remove duplicates and non-existent columns
        feature_cols = list(set([col for col in feature_cols if col in df.columns]))
    
        # Handle missing values
        cluster_data = df[feature_cols].fillna(df[feature_cols].median())
    
        # Scale the data
        scaler = StandardScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)
    
        # Perform clustering
        if clustering_method == 'kmeans':
            self.cluster_model = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
            cluster_labels = self.cluster_model.fit_predict(cluster_data_scaled)
    
        elif clustering_method == 'dbscan':
            self.cluster_model = DBSCAN(
                eps=0.5,
                min_samples=5
            )
            cluster_labels = self.cluster_model.fit_predict(cluster_data_scaled)
    
        else:
            raise ValueError("Unsupported clustering method. Use 'kmeans' or 'dbscan'.")
    
        # Add cluster labels to the data
        df['cluster'] = cluster_labels
    
        # Save clustering metadata for later use
        self.cluster_features = cluster_data.columns.tolist()
        self.cluster_scaler = scaler
    
        # Analyze the clusters
        cluster_summary = df.groupby('cluster').agg({
            col: 'mean' for col in feature_cols
        })
    
        print(f"Clustering completed using {clustering_method}.")
        print(f"Number of clusters: {len(cluster_summary)}")
        print("\nCluster Summaries:")
        print(cluster_summary)
    
        # Visualize clusters (2D projection)
        from sklearn.decomposition import PCA
    
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(cluster_data_scaled)
    
        plt.figure(figsize=(10, 8))
        for cluster_id in range(n_clusters if clustering_method == 'kmeans' else len(set(cluster_labels))):
            if cluster_id == -1 and clustering_method == 'dbscan':
                plt.scatter(
                    pca_result[cluster_labels == cluster_id, 0],
                    pca_result[cluster_labels == cluster_id, 1],
                    s=50, c='black', label='Noise'
                )
            else:
                plt.scatter(
                    pca_result[cluster_labels == cluster_id, 0],
                    pca_result[cluster_labels == cluster_id, 1],
                    s=50, label=f'Cluster {cluster_id}'
                )
    
        plt.title(f'Cluster Visualization (PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.tight_layout()
        plt.savefig("cluster_visualization.png")
        plt.close()
    
        # Define cluster profiles for recommendations
        cluster_profiles = {}
    
        for cluster_id, data in cluster_summary.iterrows():
            profile = {}
    
            if 'steps' in data:
                if data['steps'] < 5000:
                    profile['activity_level'] = "Low"
                elif data['steps'] < 10000:
                    profile['activity_level'] = "Moderate"
                else:
                    profile['activity_level'] = "High"
    
            if 'sleep_duration' in data:
                if data['sleep_duration'] < 6:
                    profile['sleep_level'] = "Poor"
                elif data['sleep_duration'] < 7.5:
                    profile['sleep_level'] = "Adequate"
                else:
                    profile['sleep_level'] = "Good"
    
            cluster_profiles[cluster_id] = profile
    
        self.cluster_profiles = cluster_profiles
        print("\nCluster Profiles for Recommendations:")
        for cluster_id, profile in cluster_profiles.items():
            print(f"Cluster {cluster_id}: {profile}")
    
        return df['cluster'], cluster_summary

    #
    
    '''def perform_clustering(self, n_clusters=3, clustering_method='kmeans'):
        """
        Perform clustering on the feature data for personalization
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters for KMeans
        clustering_method : str
            Method for clustering ('kmeans', 'dbscan')
        """
        if self.features is None:
            raise ValueError("No feature data available. Please run engineer_features() first.")
        
        # Get relevant features for clustering
        df = self.features.copy()
        
        # Select features for clustering
        feature_cols = []
        for pattern in ['steps', 'sleep', 'heart', 'calories', 'avg', 'std']:
            feature_cols.extend([col for col in df.columns if pattern in col.lower()])
        
        # Ensure we have the core features
        core_features = ['steps', 'sleep_duration']
        feature_cols.extend([col for col in core_features if col in df.columns])
        
        # Remove duplicates and non-existent columns
        feature_cols = list(set([col for col in feature_cols if col in df.columns]))
        
        # Handle missing values
        cluster_data = df[feature_cols].fillna(df[feature_cols].median())
        
        # Scale the data
        scaler = StandardScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)
        
        # Perform clustering
        if clustering_method == 'kmeans':
            # KMeans clustering
            self.cluster_model = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
            cluster_labels = self.cluster_model.fit_predict(cluster_data_scaled)
            
        elif clustering_method == 'dbscan':
            # DBSCAN clustering
            self.cluster_model = DBSCAN(
                eps=0.5,
                min_samples=5
            )
            cluster_labels = self.cluster_model.fit_predict(cluster_data_scaled)
            
        else:
            raise ValueError("Unsupported clustering method. Use 'kmeans' or 'dbscan'.")
        
        # Add cluster labels to the data
        df['cluster'] = cluster_labels
        
        # Analyze the clusters
        cluster_summary = df.groupby('cluster').agg({
            col: 'mean' for col in feature_cols
        })
        
        print(f"Clustering completed using {clustering_method}.")
        print(f"Number of clusters: {len(cluster_summary)}")
        print("\nCluster Summaries:")
        print(cluster_summary)
        
        # Visualize clusters (2D projection)
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(cluster_data_scaled)
        
        plt.figure(figsize=(10, 8))
        for cluster_id in range(n_clusters if clustering_method == 'kmeans' else len(set(cluster_labels))):
            if cluster_id == -1 and clustering_method == 'dbscan':
                # Noise points in DBSCAN
                plt.scatter(
                    pca_result[cluster_labels == cluster_id, 0],
                    pca_result[cluster_labels == cluster_id, 1],
                    s=50, c='black', label='Noise'
                )
            else:
                plt.scatter(
                    pca_result[cluster_labels == cluster_id, 0],
                    pca_result[cluster_labels == cluster_id, 1],
                    s=50, label=f'Cluster {cluster_id}'
                )
        
        plt.title(f'Cluster Visualization (PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.tight_layout()
        plt.savefig("cluster_visualization.png")
        plt.close()
        
        # Define cluster profiles for recommendations
        cluster_profiles = {}
        
        for cluster_id, data in cluster_summary.iterrows():
            profile = {}
            
            if 'steps' in data:
                if data['steps'] < 5000:
                    profile['activity_level'] = "Low"
                elif data['steps'] < 10000:
                    profile['activity_level'] = "Moderate"
                else:
                    profile['activity_level'] = "High"
            
            if 'sleep_duration' in data:
                if data['sleep_duration'] < 6:
                    profile['sleep_level'] = "Poor"
                elif data['sleep_duration'] < 7.5:
                    profile['sleep_level'] = "Adequate"
                else:
                    profile['sleep_level'] = "Good"
            
            cluster_profiles[cluster_id] = profile
        
        self.cluster_profiles = cluster_profiles
        print("\nCluster Profiles for Recommendations:")
        for cluster_id, profile in cluster_profiles.items():
            print(f"Cluster {cluster_id}: {profile}")
        
        return df['cluster'], cluster_summary'''
    
    def build_recommendation_engine(self):
        """Build the recommendation engine based on model predictions and clusters"""
        if not self.models or 'sleep_prediction' not in self.models:
            raise ValueError("Sleep prediction model not found. Please train models first.")
        
        if self.cluster_profiles is None:
            raise ValueError("Cluster profiles not available. Please run perform_clustering() first.")
        
        # Create recommendation rules
        recommendation_rules = {
            # Rules based on sleep quality prediction
            'sleep_prediction': {
                0: [  # Predicted poor sleep
                    "Try to maintain a consistent sleep schedule by going to bed and waking up at the same time every day.",
                    "Create a relaxing bedtime routine to signal your body it's time to wind down.",
                    "Limit screen time at least 1 hour before bed to reduce blue light exposure.",
                    "Make sure your bedroom is cool, dark, and quiet for optimal sleep.",
                    "Avoid caffeine and alcohol close to bedtime."
                ],
                1: [  # Predicted good sleep
                    "Your sleep habits are good! Maintain your current sleep routine.",
                    "Consider adding a brief mindfulness session before bed to further improve sleep quality.",
                    "Track any changes to your routine that might affect this positive sleep pattern."
                ]
            },
            
            # Rules based on cluster profiles
            'clusters': {}
        }
        
        # Add cluster-specific recommendations
        for cluster_id, profile in self.cluster_profiles.items():
            recommendations = []
            
            # Activity level recommendations
            if 'activity_level' in profile:
                if profile['activity_level'] == "Low":
                    recommendations.extend([
                        "Try to increase your daily step count by taking short walks during breaks.",
                        "Consider setting a reminder to stand up and move every hour.",
                        "Start with a goal of 5,000 steps and gradually increase it."
                    ])
                elif profile['activity_level'] == "Moderate":
                    recommendations.extend([
                        "You're doing well with activity! Try to add one more active day per week.",
                        "Consider adding variety to your routine with different types of activities.",
                        "Work towards reaching 10,000 steps consistently."
                    ])
                elif profile['activity_level'] == "High":
                    recommendations.extend([
                        "Great job staying active! Make sure to include rest days for recovery.",
                        "Consider adding strength training to complement your high step count.",
                        "Focus on maintaining consistency rather than increasing intensity."
                    ])
            
            # Sleep level recommendations
            if 'sleep_level' in profile:
                if profile['sleep_level'] == "Poor":
                    recommendations.extend([
                        "Your sleep duration is below recommended levels. Try to add 30 minutes to your sleep time.",
                        "Establish a consistent bedtime routine to improve sleep quality.",
                        "Limit caffeine after noon and avoid late meals."
                    ])
                elif profile['sleep_level'] == "Adequate":
                    recommendations.extend([
                        "Your sleep is adequate. Try to improve consistency in sleep and wake times.",
                        "Consider tracking your sleep stages to better understand your sleep quality.",
                        "Aim for 7-8 hours of sleep consistently."
                    ])
                elif profile['sleep_level'] == "Good":
                    recommendations.extend([
                        "You have good sleep habits! Maintain your current sleep schedule.",
                        "Continue to prioritize your sleep as part of your health routine.",
                        "Monitor how your daily activities affect your high-quality sleep."
                    ])
            
            recommendation_rules['clusters'][cluster_id] = recommendations
        
        self.recommendation_engine = recommendation_rules
        print("Recommendation engine built successfully.")
        return recommendation_rules
    
    def get_personalized_recommendations(self, user_data):
        """
        Get personalized health recommendations based on user data
    
        Parameters:
        -----------
        user_data : dict or Series
            User data with necessary features for prediction
    
        Returns:
        --------
        dict
            Personalized recommendations
        """
        if self.models.get('sleep_prediction') is None or self.recommendation_engine is None:
            raise ValueError("Models or recommendation engine not available. Please train models and build recommendation engine first.")
        
        if self.scaler is None:
            raise ValueError("Scaler not available. Ensure prepare_for_modeling() has been called.")
    
        # Convert user_data to DataFrame if it's a dict
        if isinstance(user_data, dict):
            user_data = pd.Series(user_data).to_frame().T
    
        # Ensure all necessary features are available for prediction
        model = self.models['sleep_prediction']
        #required_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else self.features.columns
        
        required_features = self.model_features
        missing_features = [feat for feat in required_features if feat not in user_data.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            for feat in missing_features:
                user_data[feat] = self.features[feat].median()
    
        # Align order and drop extras
        user_data = user_data[required_features]
    
        # Scale for prediction
        user_data_scaled = self.scaler.transform(user_data)
    
        # Predict sleep quality
        sleep_prediction = self.models['sleep_prediction'].predict(user_data_scaled)[0]
    
        # Predict steps if step model exists
        if 'step_prediction' in self.models:
            step_prediction = self.models['step_prediction'].predict(user_data_scaled)[0]
        else:
            step_prediction = None
    
        # Determine cluster
        cluster = None
        if self.cluster_model is not None and self.cluster_features is not None and self.cluster_scaler is not None:
            missing_cluster_feats = [f for f in self.cluster_features if f not in user_data.columns]
            for feat in missing_cluster_feats:
                user_data[feat] = self.features[feat].median()
    
            clustering_data = user_data[self.cluster_features]
            cluster_scaled = self.cluster_scaler.transform(clustering_data)
    
            if hasattr(self.cluster_model, 'predict'):
                cluster = self.cluster_model.predict(cluster_scaled)[0]
            else:
                cluster = self.cluster_model.fit_predict(cluster_scaled)[0]
    
        # Construct recommendations
        recommendations = {
            'sleep_recommendations': self.recommendation_engine['sleep_prediction'][sleep_prediction],
            'predicted_sleep_quality': 'Good' if sleep_prediction == 1 else 'Poor'
        }
    
        if step_prediction is not None:
            recommendations['predicted_steps'] = int(step_prediction)
    
        if cluster is not None and cluster in self.recommendation_engine['clusters']:
            recommendations['cluster_recommendations'] = self.recommendation_engine['clusters'][cluster]
            recommendations['user_cluster'] = int(cluster)
    
        return recommendations

    

    '''def get_personalized_recommendations(self, user_data):
        """
        Get personalized health recommendations based on user data
        
        Parameters:
        -----------
        user_data : dict or Series
            User data with necessary features for prediction
        
        Returns:
        --------
        dict
            Personalized recommendations
        """
        if self.models.get('sleep_prediction') is None or self.recommendation_engine is None:
            raise ValueError("Models or recommendation engine not available. Please train models and build recommendation engine first.")
        
        # Convert user_data to DataFrame if it's a dict
        if isinstance(user_data, dict):
            user_data = pd.Series(user_data).to_frame().T
        
        # Ensure all necessary features are available
        model = self.models['sleep_prediction']
        required_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        
        if required_features is not None:
            missing_features = [feat for feat in required_features if feat not in user_data.columns]
            if missing_features:
                print(f"Warning: Missing features: {missing_features}")
                # Fill missing features with median values
                for feat in missing_features:
                    user_data[feat] = self.features[feat].median()
                    
        user_data = user_data[required_features]

        # Scale the features
        user_data_scaled = self.scaler.transform(user_data)
        
        # Predict sleep quality
        sleep_prediction = self.models['sleep_prediction'].predict(user_data_scaled)[0]
        
        # Predict steps if model exists
        if 'step_prediction' in self.models:
            step_prediction = self.models['step_prediction'].predict(user_data_scaled)[0]
        else:
            step_prediction = None
        
        # Determine cluster
        #if self.cluster_model is not None:
            # For clustering, ensure the same feature subset used during training
        if self.cluster_model is not None:
            cluster_features = self.cluster_model.n_features_in_  # Number of expected features
            # Reconstruct the feature list used for clustering
            clustering_feature_cols = []
            for pattern in ['steps', 'sleep', 'heart', 'calories', 'avg', 'std']:
                clustering_feature_cols.extend([col for col in user_data.columns if pattern in col.lower()])
            clustering_feature_cols = list(set(clustering_feature_cols))  # Remove duplicates
            # Ensure only features used in clustering are used
            clustering_data = user_data[clustering_feature_cols].copy()
            clustering_data = clustering_data[self.cluster_model.feature_names_in_]
            cluster_scaled = StandardScaler().fit(self.features[self.cluster_model.feature_names_in_]).transform(clustering_data)
            
            if hasattr(self.cluster_model, 'predict'):
                cluster = self.cluster_model.predict(cluster_scaled)[0]
            else:
                cluster = self.cluster_model.fit_predict(cluster_scaled)[0]


            
            #if hasattr(self.cluster_model, 'predict'):
            #    cluster = self.cluster_model.predict(user_data_scaled)[0]
            #else:
            #    # For DBSCAN
            #    cluster = self.cluster_model.fit_predict(user_data_scaled)[0]
        else:
            cluster = None
        
        # Get recommendations based on predictions and cluster
        recommendations = {
            'sleep_recommendations': self.recommendation_engine['sleep_prediction'][sleep_prediction],
            'predicted_sleep_quality': 'Good' if sleep_prediction == 1 else 'Poor'
        }
        
        if step_prediction is not None:
            recommendations['predicted_steps'] = int(step_prediction)
        
        if cluster is not None and cluster in self.recommendation_engine['clusters']:
            recommendations['cluster_recommendations'] = self.recommendation_engine['clusters'][cluster]
            recommendations['user_cluster'] = int(cluster)
        
        return recommendations'''
    
    def create_streamlit_dashboard(self):
        """
        Create a Streamlit dashboard for visualization and interaction
        Note: This method should be run separately as a Streamlit app
        """
        pass


def streamlit_app():
    """
    Standalone function to run the Streamlit dashboard
    """
    st.set_page_config(
        page_title="Health Analytics Dashboard",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 AI-Powered Personal Health Analytics")
    st.subheader("Step and Sleep Data Analysis Dashboard")
    
    # Initialize session state for health system
    if 'health_system' not in st.session_state:
        st.session_state.health_system = HealthAnalyticsSystem()
        st.session_state.data_loaded = False
        st.session_state.models_trained = False
        st.session_state.clustering_done = False
    
    # Sidebar for navigation and controls
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a Page",
        ["Upload Data", "Data Exploration", "Feature Analysis", "Predictive Models", "Recommendations"]
    )
    
    # Page 1: Upload Data
    if page == "Upload Data":
        st.header("Upload Your Health Data")
        
        data_source = st.selectbox(
            "Select Data Source",
            ["Fitbit", "Apple Health", "MESA", "Sample Data"]
        )
        
        if data_source != "Sample Data":
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    # Save the uploaded file to disk temporarily
                    with open("temp_upload.csv", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Load the data
                    health_system = st.session_state.health_system
                    data = health_system.load_data("temp_upload.csv", data_source.lower().replace(" ", "_"))
                    
                    # Clean the data
                    cleaned_data = health_system.clean_data()
                    
                    st.success("Data loaded and preprocessed successfully!")
                    st.session_state.data_loaded = True
                    
                    # Show a sample of the data
                    st.subheader("Sample of Loaded Data")
                    st.dataframe(cleaned_data.head())
                    
                    # Show basic statistics
                    st.subheader("Basic Statistics")
                    st.dataframe(cleaned_data.describe())
                    
                except Exception as e:
                    st.error(f"Error loading data: {e}")
        else:
            # Use sample data
            if st.button("Load Sample Data"):
                try:
                    # Generate sample data
                    dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='D')
                    np.random.seed(42)
                    
                    data = pd.DataFrame({
                        'date': dates,
                        'steps': np.random.randint(2000, 15000, size=len(dates)),
                        'sleep_duration': np.random.uniform(5.0, 9.0, size=len(dates)),
                        'calories': np.random.randint(1500, 3000, size=len(dates)),
                        'resting_heart_rate': np.random.randint(55, 80, size=len(dates))
                    })
                    
                    # Save sample data
                    data.to_csv("sample_health_data.csv", index=False)
                    
                    # Load the data
                    health_system = st.session_state.health_system
                    health_system.data = data.copy()
                    health_system.data['date'] = pd.to_datetime(health_system.data['date'])
                    health_system.data.set_index('date', inplace=True)
                    
                    # Clean the data
                    cleaned_data = health_system.clean_data()
                    
                    st.success("Sample data loaded and preprocessed successfully!")
                    st.session_state.data_loaded = True
                    
                    # Show a sample of the data
                    st.subheader("Sample of Loaded Data")
                    st.dataframe(cleaned_data.head())
                    
                    # Show basic statistics
                    st.subheader("Basic Statistics")
                    st.dataframe(cleaned_data.describe())
                    
                except Exception as e:
                    st.error(f"Error loading sample data: {e}")
    
    # Page 2: Data Exploration
    elif page == "Data Exploration":
        if not st.session_state.data_loaded:
            st.warning("Please upload data first on the 'Upload Data' page.")
        else:
            st.header("Exploratory Data Analysis")
            
            # Run EDA if not already done
            if 'eda_done' not in st.session_state:
                with st.spinner("Performing exploratory data analysis..."):
                    health_system = st.session_state.health_system
                    correlation_matrix = health_system.perform_eda(save_plots=True)
                    st.session_state.eda_done = True
            
            # Show correlation heatmap
            st.subheader("Correlation Matrix")
            corr_data = st.session_state.health_system.preprocessed_data.select_dtypes(include=[np.number]).corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
            st.pyplot(fig)
            
            # Time series plots
            st.subheader("Time Series Analysis")
            data = st.session_state.health_system.preprocessed_data
            
            # Create tabs for different visualizations
            time_series_tab, distribution_tab, day_analysis_tab = st.tabs(["Time Series", "Distributions", "Day Analysis"])
            
            with time_series_tab:
                # Steps and sleep over time
                if all(col in data.columns for col in ['steps', 'sleep_duration']):
                    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
                    
                    ax[0].plot(data.index, data['steps'], label='Steps')
                    ax[0].set_title('Steps Over Time')
                    ax[0].legend()
                    
                    ax[1].plot(data.index, data['sleep_duration'], label='Sleep Duration', color='orange')
                    ax[1].set_title('Sleep Duration Over Time')
                    ax[1].legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            
            with distribution_tab:
                # Distribution plots
                for col in ['steps', 'sleep_duration', 'calories'] if all(c in data.columns for c in ['steps', 'sleep_duration', 'calories']) else data.select_dtypes(include=[np.number]).columns[:3]:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(data[col], kde=True, ax=ax)
                    ax.set_title(f'Distribution of {col}')
                    st.pyplot(fig)
            
            with day_analysis_tab:
                # Day of week analysis
                data['day_of_week'] = data.index.day_name()
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                # Steps by day of week
                if 'steps' in data.columns:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.boxplot(x='day_of_week', y='steps', data=data, order=day_order, ax=ax)
                    ax.set_title('Steps by Day of Week')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                
                # Sleep by day of week
                if 'sleep_duration' in data.columns:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.boxplot(x='day_of_week', y='sleep_duration', data=data, order=day_order, ax=ax)
                    ax.set_title('Sleep Duration by Day of Week')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
    
    # Page 3: Feature Analysis
    elif page == "Feature Analysis":
        if not st.session_state.data_loaded:
            st.warning("Please upload data first on the 'Upload Data' page.")
        else:
            st.header("Feature Engineering and Analysis")
            
            # Run feature engineering if not already done
            if 'features_done' not in st.session_state:
                with st.spinner("Performing feature engineering..."):
                    health_system = st.session_state.health_system
                    features = health_system.engineer_features()
                    st.session_state.features_done = True
            
            # Show engineered features
            features = st.session_state.health_system.features
            st.subheader("Engineered Features")
            st.write(f"Total features: {features.shape[1]}")
            st.dataframe(features.head())
            
            # Feature correlations
            st.subheader("Feature Correlations with Sleep Quality")
            if 'good_sleep' in features.columns:
                corr_with_sleep = features.corr()['good_sleep'].sort_values(ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_with_sleep = corr_with_sleep[corr_with_sleep.index != 'good_sleep']  # Remove self-correlation
                corr_with_sleep = corr_with_sleep.head(15)  # Top 15 correlations
                
                sns.barplot(x=corr_with_sleep.values, y=corr_with_sleep.index, ax=ax)
                ax.set_title('Top Features Correlated with Sleep Quality')
                ax.set_xlabel('Correlation Coefficient')
                st.pyplot(fig)
            
            # Feature correlations with steps
            st.subheader("Feature Correlations with Step Count")
            if 'steps' in features.columns:
                corr_with_steps = features.corr()['steps'].sort_values(ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_with_steps = corr_with_steps[corr_with_steps.index != 'steps']  # Remove self-correlation
                corr_with_steps = corr_with_steps.head(15)  # Top 15 correlations
                
                sns.barplot(x=corr_with_steps.values, y=corr_with_steps.index, ax=ax)
                ax.set_title('Top Features Correlated with Step Count')
                ax.set_xlabel('Correlation Coefficient')
                st.pyplot(fig)
    
    # Page 4: Predictive Models
    elif page == "Predictive Models":
        if not st.session_state.data_loaded or not hasattr(st.session_state.health_system, 'features'):
            st.warning("Please upload data and generate features first.")
        else:
            st.header("Predictive Models")
            
            tab1, tab2 = st.tabs(["Sleep Quality Prediction", "Step Count Prediction"])
            
            with tab1:
                st.subheader("Sleep Quality Prediction Model")
                
                if st.button("Train Sleep Prediction Model"):
                    with st.spinner("Training sleep prediction model..."):
                        try:
                            health_system = st.session_state.health_system
                            
                            # Prepare for modeling
                            X_train, X_test, y_train, y_test, feature_names = health_system.prepare_for_modeling(
                                target_variable='good_sleep',
                                time_series_split=True
                            )
                            
                            # Train model
                            model = health_system.train_sleep_prediction_model(X_train, y_train, model_type='random_forest')
                            
                            # Evaluate model
                            eval_results = health_system.evaluate_model('sleep_prediction', X_test, y_test)
                            
                            st.session_state.sleep_model_trained = True
                            st.session_state.sleep_eval_results = eval_results
                            st.session_state.models_trained = True
                            
                            st.success("Sleep prediction model trained successfully!")
                            
                        except Exception as e:
                            st.error(f"Error training model: {e}")
                
                # Show evaluation results if available
                if hasattr(st.session_state, 'sleep_model_trained') and st.session_state.sleep_model_trained:
                    eval_results = st.session_state.sleep_eval_results
                    
                    # Metrics
                    metrics_cols = st.columns(4)
                    metrics_cols[0].metric("Accuracy", f"{eval_results['accuracy']:.4f}")
                    metrics_cols[1].metric("Precision", f"{eval_results['precision']:.4f}")
                    metrics_cols[2].metric("Recall", f"{eval_results['recall']:.4f}")
                    if eval_results['roc_auc'] is not None:
                        metrics_cols[3].metric("ROC-AUC", f"{eval_results['roc_auc']:.4f}")
                    
                    # Feature importances
                    if eval_results['feature_importance'] is not None:
                        st.subheader("Feature Importance")
                        feature_imp = eval_results['feature_importance'].head(10)
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.barplot(x='Importance', y='Feature', data=feature_imp, ax=ax)
                        ax.set_title('Top 10 Important Features for Sleep Prediction')
                        st.pyplot(fig)
            
            with tab2:
                st.subheader("Step Count Prediction Model")
                
                if st.button("Train Step Prediction Model"):
                    with st.spinner("Training step prediction model..."):
                        try:
                            health_system = st.session_state.health_system
                            
                            # Prepare for modeling
                            X_train, X_test, y_train, y_test, feature_names = health_system.prepare_for_modeling(
                                target_variable='steps',
                                time_series_split=True
                            )
                            
                            # Train model
                            model = health_system.train_step_prediction_model(X_train, y_train, model_type='random_forest')
                            
                            # Evaluate model
                            eval_results = health_system.evaluate_model('step_prediction', X_test, y_test)
                            
                            st.session_state.step_model_trained = True
                            st.session_state.step_eval_results = eval_results
                            st.session_state.models_trained = True
                            
                            st.success("Step prediction model trained successfully!")
                            
                        except Exception as e:
                            st.error(f"Error training model: {e}")
                
                # Show evaluation results if available
                if hasattr(st.session_state, 'step_model_trained') and st.session_state.step_model_trained:
                    eval_results = st.session_state.step_eval_results
                    
                    # Metrics
                    metrics_cols = st.columns(3)
                    metrics_cols[0].metric("MSE", f"{eval_results['mse']:.2f}")
                    metrics_cols[1].metric("RMSE", f"{eval_results['rmse']:.2f}")
                    metrics_cols[2].metric("R² Score", f"{eval_results['r2']:.4f}")
                    
                    # Feature importances
                    if eval_results['feature_importance'] is not None:
                        st.subheader("Feature Importance")
                        feature_imp = eval_results['feature_importance'].head(10)
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.barplot(x='Importance', y='Feature', data=feature_imp, ax=ax)
                        ax.set_title('Top 10 Important Features for Step Prediction')
                        st.pyplot(fig)
    
    # Page 5: Recommendations
    elif page == "Recommendations":
        if not st.session_state.data_loaded or not st.session_state.models_trained:
            st.warning("Please upload data and train models first.")
        else:
            st.header("Personalized Health Recommendations")
            
            # Perform clustering if not already done
            if not st.session_state.clustering_done:
                with st.spinner("Performing user clustering for personalization..."):
                    try:
                        health_system = st.session_state.health_system
                        cluster_labels, cluster_summary = health_system.perform_clustering(n_clusters=3)
                        
                        # Build recommendation engine
                        recommendation_rules = health_system.build_recommendation_engine()
                        
                        st.session_state.clustering_done = True
                        st.success("Clustering and recommendation engine built successfully!")
                    except Exception as e:
                        st.error(f"Error performing clustering: {e}")
            
            # User clustering visualization
            if st.session_state.clustering_done:
                st.subheader("User Clusters")
                
                # Show cluster visualization
                img = plt.imread("cluster_visualization.png")
                st.image(img, caption="Cluster Visualization (PCA)", use_column_width=True)
                
                # Get cluster profiles
                cluster_profiles = st.session_state.health_system.cluster_profiles
                
                # Display cluster profiles
                st.subheader("Cluster Profiles")
                for cluster_id, profile in cluster_profiles.items():
                    with st.expander(f"Cluster {cluster_id}"):
                        st.write(profile)
            
            # Get personalized recommendations
            st.subheader("Get Your Personalized Recommendations")
            
            # Create input form for user data
            with st.form("user_data_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    steps = st.number_input("Today's Steps", min_value=0, max_value=30000, value=8000)
                    sleep = st.number_input("Last Night's Sleep (hours)", min_value=0.0, max_value=12.0, value=7.0)
                
                with col2:
                    heart_rate = st.number_input("Resting Heart Rate (bpm)", min_value=40, max_value=120, value=65)
                    calories = st.number_input("Calories Burned", min_value=0, max_value=5000, value=2000)
                
                submit_button = st.form_submit_button("Get Recommendations")
            
            if submit_button:
                try:
                    # Create user data
                    user_data = {
                        'steps': steps,
                        'sleep_duration': sleep,
                        'resting_heart_rate': heart_rate,
                        'calories': calories,
                        'day_of_week': datetime.now().weekday(),
                        'is_weekend': 1 if datetime.now().weekday() >= 5 else 0
                    }
                    
                    # Get recommendations
                    health_system = st.session_state.health_system
                    recommendations = health_system.get_personalized_recommendations(user_data)
                    
                    # Display recommendations
                    st.subheader("Your Health Insights")
                    
                    # Create columns for metrics
                    metric_cols = st.columns(3)
                    
                    # Show predicted sleep quality
                    metric_cols[0].metric(
                        "Predicted Sleep Quality", 
                        recommendations['predicted_sleep_quality'],
                        delta=None
                    )
                    
                    # Show user cluster if available
                    if 'user_cluster' in recommendations:
                        metric_cols[1].metric(
                            "Your User Profile", 
                            f"Cluster {recommendations['user_cluster']}",
                            delta=None
                        )
                    
                    # Show step prediction if available
                    if 'predicted_steps' in recommendations:
                        metric_cols[2].metric(
                            "Tomorrow's Step Prediction", 
                            f"{recommendations['predicted_steps']} steps",
                            delta=None
                        )
                    
                    # Show recommendations
                    st.subheader("Your Personalized Recommendations")
                    
                    # Sleep recommendations
                    with st.expander("Sleep Recommendations", expanded=True):
                        for rec in recommendations['sleep_recommendations'][:3]:  # Show top 3
                            st.markdown(f"• {rec}")
                    
                    # Cluster-specific recommendations
                    if 'cluster_recommendations' in recommendations:
                        with st.expander("Activity & Lifestyle Recommendations", expanded=True):
                            for rec in recommendations['cluster_recommendations'][:3]:  # Show top
                                st.markdown(f"• {rec}")
                    
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")


if __name__ == '__main__':
    # Main entry point
    health_system = HealthAnalyticsSystem()
    
    # If running this as a script, start with a demo
    print("Starting AI-Powered Personal Health Analytics Demo...")
    
    # Option 1: Run as a standalone script
    if 'streamlit_app' in sys.argv:
        streamlit_app()
    else:
        # Option 2: Run as a demonstration script
        # For this demo, we'll generate some sample data
        print("Generating sample data...")
        
        # Generate sample dates
        dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='D')
        np.random.seed(42)
        
        # Create sample dataframe
        data = pd.DataFrame({
            'date': dates,
            'steps': np.random.randint(2000, 15000, size=len(dates)),
            'sleep_duration': np.random.uniform(5.0, 9.0, size=len(dates)),
            'calories': np.random.randint(1500, 3000, size=len(dates)),
            'resting_heart_rate': np.random.randint(55, 80, size=len(dates))
        })
        
        # Save sample data
        data.to_csv("H:/ai_work/FITBIT_SLeepTracker/sample_health_data.csv", index=False)
        print("Sample data saved to 'sample_health_data.csv'")
        
        # Load data
        print("\n1. Loading data...")
        health_system.data = data.copy()
        health_system.data['date'] = pd.to_datetime(health_system.data['date'])
        health_system.data.set_index('date', inplace=True)
        
        # Clean data
        print("\n2. Cleaning data...")
        cleaned_data = health_system.clean_data()
        print(cleaned_data.head())
        
        # Perform EDA
        print("\n3. Performing EDA...")
        health_system.perform_eda(save_plots=True)
        
        # Feature engineering
        print("\n4. Engineering features...")
        features = health_system.engineer_features()
        print(f"Created {features.shape[1]} features")
        
        # Train sleep model
        print("\n5. Training sleep prediction model...")
        X_train, X_test, y_train, y_test, feature_names = health_system.prepare_for_modeling(
            target_variable='good_sleep',
            time_series_split=True
        )
        sleep_model = health_system.train_sleep_prediction_model(X_train, y_train)
        
        # Evaluate model
        print("\n6. Evaluating sleep prediction model...")
        sleep_eval = health_system.evaluate_model('sleep_prediction', X_test, y_test)
        
        # Train step model
        print("\n7. Training step prediction model...")
        X_train, X_test, y_train, y_test, feature_names = health_system.prepare_for_modeling(
            target_variable='steps',
            time_series_split=True
        )
        step_model = health_system.train_step_prediction_model(X_train, y_train)
        
        # Evaluate model
        print("\n8. Evaluating step prediction model...")
        step_eval = health_system.evaluate_model('step_prediction', X_test, y_test)
        
        # Perform clustering
        print("\n9. Performing user clustering...")
        cluster_labels, cluster_summary = health_system.perform_clustering(n_clusters=3)
        
        # Build recommendation engine
        print("\n10. Building recommendation engine...")
        recommendation_rules = health_system.build_recommendation_engine()
        
        # Get sample recommendations
        print("\n11. Getting sample recommendations...")
        sample_user = {
            'steps': 8000,
            'sleep_duration': 6.5,
            'resting_heart_rate': 68,
            'calories': 2200,
            'day_of_week': 3,  # Thursday
            'is_weekend': 0
        }
        
        recommendations = health_system.get_personalized_recommendations(sample_user)
        
        print("\nSample Personalized Recommendations:")
        print(f"Predicted Sleep Quality: {recommendations['predicted_sleep_quality']}")
        if 'predicted_steps' in recommendations:
            print(f"Predicted Steps: {recommendations['predicted_steps']}")
        if 'user_cluster' in recommendations:
            print(f"User Cluster: {recommendations['user_cluster']}")
        
        print("\nSleep Recommendations:")
        for i, rec in enumerate(recommendations['sleep_recommendations'][:3], 1):
            print(f"{i}. {rec}")
        
        if 'cluster_recommendations' in recommendations:
            print("\nActivity & Lifestyle Recommendations:")
            for i, rec in enumerate(recommendations['cluster_recommendations'][:3], 1):
                print(f"{i}. {rec}")
        
        print("\nDemo completed! To run the interactive dashboard, use:")
        print("streamlit run app.py")
        