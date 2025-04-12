import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

def load_simulator_models():
    treated_model = joblib.load('models/treated_model.pkl')
    control_model = joblib.load('models/control_model.pkl')
    feature_cols = joblib.load('models/feature_cols.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_stats = joblib.load('models/feature_stats.pkl')  #min, max, mean, std for each feature
    return treated_model, control_model, feature_cols, scaler, feature_stats

def run_simulator():
    st.title("Conversion Uplift Simulator")
    
    treated_model, control_model, feature_cols, scaler, feature_stats = load_simulator_models()
    
    #Create base features without interactions
    base_features = [col for col in feature_cols if '_interaction' not in col]
    
    #Create sidebar for user inputs
    st.sidebar.header("User Profile")
    
    #Create user profile with sliders
    user_profile = {}
    for feature in base_features:

        if feature in feature_stats:
            stats = feature_stats[feature]
            min_val = float(stats['min'])
            max_val = float(stats['max'])
            mean_val = float(stats['mean'])

            user_profile[feature] = st.sidebar.slider(
                f"{feature}", 
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )

    for i, feat1 in enumerate(base_features):
        for feat2 in base_features[i+1:]:
            interaction_name = f"{feat1}_{feat2}_interaction"
            if interaction_name in feature_cols:
                user_profile[interaction_name] = user_profile[feat1] * user_profile[feat2]

    user_df = pd.DataFrame([user_profile])
    

    treated_prob = treated_model.predict_proba(user_df)[0, 1]
    control_prob = control_model.predict_proba(user_df)[0, 1]
    uplift = treated_prob - control_prob
    

    st.header("Conversion Probability")
    col1, col2, col3 = st.columns(3)
    col1.metric("If Treated", f"{treated_prob:.2%}")
    col2.metric("If Not Treated", f"{control_prob:.2%}")
    col3.metric("Uplift", f"{uplift:.2%}")

    try:
        st.header("Feature Importance")
        explainer = shap.TreeExplainer(treated_model)
        shap_values = explainer.shap_values(user_df)
        
        #For classification models, we want the positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        #Create SHAP force plot
        st.write("SHAP Force Plot (Treatment Model)")
        shap_html = shap.force_plot(explainer.expected_value, 
                                    shap_values, 
                                    user_df,
                                    feature_names=feature_cols,
                                    matplotlib=True,
                                    show=False)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()
    except Exception as e:
        st.warning(f"Could not generate SHAP explanation: {str(e)}")
        st.info("Feature importance visualization is not available, but the model predictions are still valid.")
    
    # Recommendation
    st.header("Recommendation")
    if uplift > 0.01:  #1% uplift threshold
        st.success("✅ TREAT this user - Significant positive uplift expected")
    elif uplift > 0:
        st.info("🟡 CONSIDER treating this user - Small positive uplift expected")
    else:
        st.error("❌ DO NOT treat this user - No positive uplift expected")

if __name__ == "__main__":
    run_simulator()