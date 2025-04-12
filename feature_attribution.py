import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

def analyze_feature_importance(two_model, X_train, feature_cols):
    """
    Analyze feature importance from the two-model approach
    
    Args:
        two_model: Dictionary with treatment and control models
        X_train: Training features
        feature_cols: Feature column names
    """
    treated_model = two_model['treatment_model']
    control_model = two_model['control_model']
    
    #Get feature importances
    treated_importance = treated_model.feature_importances_
    control_importance = control_model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Treatment_Importance': treated_importance,
        'Control_Importance': control_importance
    })
    
    importance_df['Importance_Difference'] = importance_df['Treatment_Importance'] - importance_df['Control_Importance']
    
    #Sort by absolute difference
    importance_df = importance_df.sort_values(by='Importance_Difference', key=abs, ascending=False)
    
    return importance_df

def shap_analysis(two_model, X_train, feature_cols):
    """
    Perform SHAP analysis for detailed feature attribution
    """
    treated_model = two_model['treatment_model']
    control_model = two_model['control_model']
    
    treated_explainer = shap.TreeExplainer(treated_model)
    control_explainer = shap.TreeExplainer(control_model)

    X_sample = X_train.sample(1000, random_state=42)
    
    #Calculate SHAP values
    treated_shap = treated_explainer.shap_values(X_sample)
    control_shap = control_explainer.shap_values(X_sample)
    
    #For classification, we take the positive class
    if isinstance(treated_shap, list):
        treated_shap = treated_shap[1]
        control_shap = control_shap[1]
    
    #Calculate uplift SHAP
    uplift_shap = np.abs(treated_shap - control_shap)
    

    feature_shap_values = pd.DataFrame({
        'Feature': feature_cols,
        'Uplift_SHAP': np.mean(uplift_shap, axis=0)
    }).sort_values('Uplift_SHAP', ascending=False)
    
    return feature_shap_values, treated_shap, control_shap, uplift_shap, X_sample

def plot_feature_importance(importance_df, title="Feature Importance", top_n=20):
    """
    Plot feature importance
    """
    plt.figure(figsize=(12, 8))
    
    #Get top N features
    top_features = importance_df.head(top_n)
    
    sns.barplot(x='Uplift_Importance', y='Feature', data=top_features)
    plt.title(title)
    plt.tight_layout()
    
    return plt.gcf()

def plot_shap_summary(shap_values, X_sample, feature_names, title="SHAP Summary Plot"):
    """
    Create SHAP summary plot
    
    Args:
        shap_values: SHAP values
        X_sample: Sample data
        feature_names: Feature names
        title: Plot title
        
    """
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title(title)
    plt.tight_layout()
    
    return plt.gcf()

def save_attribution_results(feature_importance, feature_shap_values, output_dir='models'):
    """Save attribution results for later use"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    #Save feature importance
    feature_importance.to_csv(f'{output_dir}/feature_importance.csv', index=False)
    
    #Save SHAP values
    feature_shap_values.to_csv(f'{output_dir}/feature_shap_values.csv', index=False)
    
    print(f"Attribution results saved to {output_dir}/")