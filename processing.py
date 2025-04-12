import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from causalml.inference.meta import BaseTLearner, BaseSLearner, BaseXLearner
from causalml.metrics.visualize import plot_qini
from causalml.metrics import qini_score
import lightgbm as lgb
import shap

def load_criteo_data(file_path):
    """Load the Criteo uplift dataset"""
    column_names = [f'f{i}' for i in range(12)] + ['treatment', 'conversion', 'visit', 'exposure']
    data = pd.read_csv(file_path, header=None, names=column_names, low_memory=False)
    
    for col in ['treatment', 'conversion', 'visit', 'exposure']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    #Fill NaN values with 0
    data = data.fillna(0)
    
    return data

#Feature engineering
def preprocess_data(df, sample_size=None):
    """
    Args:
        df: Input dataframe
        sample_size: Optional sample size to reduce memory usage
    """
    if sample_size is not None:
        df = df.sample(sample_size, random_state=42)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    #Instead of creating all possible interactions, select only the most important ones
    base_features = [f'f{i}' for i in range(12)]
    
    #Creating a smaller set of interaction features
    #For example, only create interactions for the first 6 features
    important_features = base_features[:6]
    interaction_features = []
    
    for i in range(len(important_features)):
        for j in range(i+1, len(important_features)):
            feature_name = f'{important_features[i]}_{important_features[j]}_interaction'
            df[feature_name] = df[important_features[i]] * df[important_features[j]]
            interaction_features.append(feature_name)
    
    #Combine base features and interaction features
    all_feature_cols = base_features + interaction_features
    
    #Reducin memory usage
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    scaler = StandardScaler()
    sample_for_scaling = df[all_feature_cols].sample(min(100000, len(df)), random_state=42)
    scaler.fit(sample_for_scaling)
    
    batch_size = 100000
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        df.iloc[start:end, df.columns.get_indexer(all_feature_cols)] = scaler.transform(df.iloc[start:end][all_feature_cols])
    
    return df, all_feature_cols, scaler

#Splitting
def split_data(df, feature_cols):
    X = df[feature_cols]
    y = df['conversion']
    t = df['treatment']
    
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, t, test_size=0.2, random_state=42)
    
    X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(
        X_train, y_train, t_train, test_size=0.25, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, t_train, t_val, t_test

#EDA
def explore_data(df):
    """Perform exploratory data analysis on the dataset"""
    print(f"Dataset shape: {df.shape}")
    print("\nBasic statistics:")
    print(df.describe())

    treatment_counts = df['treatment'].astype(int).value_counts(normalize=True) * 100
    print(f"\nTreatment distribution: {treatment_counts.to_dict()}")

    df['conversion'] = pd.to_numeric(df['conversion'], errors='coerce').fillna(0)
    df['visit'] = pd.to_numeric(df['visit'], errors='coerce').fillna(0)
    
    overall_conv_rate = df['conversion'].mean() * 100
    treated_conv_rate = df[df['treatment'] == 1]['conversion'].mean() * 100
    control_conv_rate = df[df['treatment'] == 0]['conversion'].mean() * 100
    
    print(f"\nOverall conversion rate: {overall_conv_rate:.2f}%")
    print(f"Treated group conversion rate: {treated_conv_rate:.2f}%")
    print(f"Control group conversion rate: {control_conv_rate:.2f}%")
    print(f"Uplift: {treated_conv_rate - control_conv_rate:.2f}%")
    
    #Feature correlations
    feature_cols = [col for col in df.columns if col.startswith('f')]
    
    #Convert feature columns to numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    corr = df[feature_cols].corr()
    
    #correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    return plt.gcf()

def save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test, 
                          t_train, t_val, t_test, feature_cols, scaler, output_dir='models'):
    """Save preprocessed data and scaler for later use"""
    import os
    import joblib

    os.makedirs(output_dir, exist_ok=True)

    data_dict = {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        't_train': t_train, 't_val': t_val, 't_test': t_test,
        'feature_cols': feature_cols
    }
    
    joblib.dump(data_dict, f'{output_dir}/preprocessed_data.pkl')
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    
    print(f"Preprocessed data and scaler saved to {output_dir}/")