import pandas as pd
import numpy as np
import joblib
import os

def generate_feature_stats():
    """
    Generate feature statistics for the conversion simulator
    """
    print("Loading preprocessed data...")
    try:
        #Load preprocessed data from the combined file instead of individual files
        preprocessed_data = joblib.load('models/preprocessed_data.pkl')
        feature_cols = joblib.load('models/feature_cols.pkl')
        
        X_train = preprocessed_data['X_train']
        
        #Calculate statistics for each feature
        feature_stats = {}
        for col in feature_cols:
            feature_stats[col] = {
                'min': X_train[col].min(),
                'max': X_train[col].max(),
                'mean': X_train[col].mean(),
                'std': X_train[col].std()
            }
        
        #Save feature statistics
        print("Saving feature statistics...")
        joblib.dump(feature_stats, 'models/feature_stats.pkl')
        print("Feature statistics saved to models/feature_stats.pkl")
        
    except Exception as e:
        print(f"Error generating feature statistics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_feature_stats()