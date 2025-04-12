import os
import pandas as pd
import numpy as np
import joblib
from processing import load_criteo_data, preprocess_data, split_data, save_preprocessed_data, explore_data
from uplift_model import train_two_model_approach, train_metalearners, evaluate_models, save_models
from feature_attribution import analyze_feature_importance, shap_analysis, save_attribution_results

def run_pipeline(data_path, output_dir='models', sample_size=1000000, start_step=1):

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_file = f"{output_dir}/pipeline_checkpoint.pkl"
    
    #Load checkpoint data if it exists
    checkpoint_data = {}
    if os.path.exists(checkpoint_file):
        try:
            checkpoint_data = joblib.load(checkpoint_file)
            print(f"Loaded checkpoint data. Available steps: {list(checkpoint_data.keys())}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    #Step 1: Loading data
    if start_step <= 1:
        print("Step 1: Loading data...")
        data = load_criteo_data(data_path)
        checkpoint_data['data'] = data
        joblib.dump(checkpoint_data, checkpoint_file)
    else:
        print("Step 1: Loading data... [SKIPPED]")
        data = checkpoint_data.get('data')
        if data is None:
            print("Data not found in checkpoint, loading from file...")
            data = load_criteo_data(data_path)
            checkpoint_data['data'] = data
            joblib.dump(checkpoint_data, checkpoint_file)
    
    #Step 2: Exploring data
    if start_step <= 2:
        print("Step 2: Exploring data...")
        explore_plot = explore_data(data)
        explore_plot.savefig(f'{output_dir}/data_exploration.png')
    else:
        print("Step 2: Exploring data... [SKIPPED]")
    
    #Step 3: Preprocessing data
    if start_step <= 3:
        print(f"Step 3: Preprocessing data (using {sample_size if sample_size else 'all'} samples)...")
        processed_data, feature_cols, scaler = preprocess_data(data, sample_size=sample_size)
        checkpoint_data['processed_data'] = processed_data
        checkpoint_data['feature_cols'] = feature_cols
        checkpoint_data['scaler'] = scaler
        joblib.dump(checkpoint_data, checkpoint_file)
    else:
        print("Step 3: Preprocessing data... [SKIPPED]")
        processed_data = checkpoint_data.get('processed_data')
        feature_cols = checkpoint_data.get('feature_cols')
        scaler = checkpoint_data.get('scaler')
        if processed_data is None or feature_cols is None:
            print("Processed data not found in checkpoint, reprocessing...")
            processed_data, feature_cols, scaler = preprocess_data(data, sample_size=sample_size)
            checkpoint_data['processed_data'] = processed_data
            checkpoint_data['feature_cols'] = feature_cols
            checkpoint_data['scaler'] = scaler
            joblib.dump(checkpoint_data, checkpoint_file)
    
    #Step 4: Splitting data
    if start_step <= 4:
        print("Step 4: Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test, t_train, t_val, t_test = split_data(processed_data, feature_cols)
        checkpoint_data['split_data'] = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            't_train': t_train, 't_val': t_val, 't_test': t_test
        }
        joblib.dump(checkpoint_data, checkpoint_file)
    else:
        print("Step 4: Splitting data... [SKIPPED]")
        split_data_dict = checkpoint_data.get('split_data', {})
        X_train = split_data_dict.get('X_train')
        X_val = split_data_dict.get('X_val')
        X_test = split_data_dict.get('X_test')
        y_train = split_data_dict.get('y_train')
        y_val = split_data_dict.get('y_val')
        y_test = split_data_dict.get('y_test')
        t_train = split_data_dict.get('t_train')
        t_val = split_data_dict.get('t_val')
        t_test = split_data_dict.get('t_test')
        
        if X_train is None:
            print("Split data not found in checkpoint, resplitting...")
            X_train, X_val, X_test, y_train, y_val, y_test, t_train, t_val, t_test = split_data(processed_data, feature_cols)
            checkpoint_data['split_data'] = {
                'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
                't_train': t_train, 't_val': t_val, 't_test': t_test
            }
            joblib.dump(checkpoint_data, checkpoint_file)
    
    #Step 5: Saving preprocessed data
    if start_step <= 5:
        print("Step 5: Saving preprocessed data...")
        save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test, 
                              t_train, t_val, t_test, feature_cols, scaler, output_dir)
    else:
        print("Step 5: Saving preprocessed data... [SKIPPED]")
    
    #Step 6: Training two-model approach
    if start_step <= 6:
        print("Step 6: Training two-model approach...")
        two_model = train_two_model_approach(X_train, y_train, t_train)
        checkpoint_data['two_model'] = two_model
        joblib.dump(checkpoint_data, checkpoint_file)
    else:
        print("Step 6: Training two-model approach... [SKIPPED]")
        two_model = checkpoint_data.get('two_model')
        if two_model is None:
            print("Two-model not found in checkpoint, retraining...")
            two_model = train_two_model_approach(X_train, y_train, t_train)
            checkpoint_data['two_model'] = two_model
            joblib.dump(checkpoint_data, checkpoint_file)
    
    #Step 7: Training meta-learners
    if start_step <= 7:
        print("Step 7: Training meta-learners...")
        metalearners = train_metalearners(X_train, y_train, t_train)
        checkpoint_data['metalearners'] = metalearners
        joblib.dump(checkpoint_data, checkpoint_file)
    else:
        print("Step 7: Training meta-learners... [SKIPPED]")
        metalearners = checkpoint_data.get('metalearners')
        if metalearners is None:
            print("Meta-learners not found in checkpoint, retraining...")
            metalearners = train_metalearners(X_train, y_train, t_train)
            checkpoint_data['metalearners'] = metalearners
            joblib.dump(checkpoint_data, checkpoint_file)
    
    #Step 8: Evaluating models
    if start_step <= 8:
        print("Step 8: Evaluating models...")
        results, uplift_scores = evaluate_models(two_model, metalearners, X_val, y_val, t_val)
        checkpoint_data['evaluation'] = {'results': results, 'uplift_scores': uplift_scores}
        joblib.dump(checkpoint_data, checkpoint_file)
        
        print("Model evaluation results:")
        for model, score in results.items():
            print(f"{model}: {score:.4f}")
    else:
        print("Step 8: Evaluating models... [SKIPPED]")
        evaluation = checkpoint_data.get('evaluation', {})
        results = evaluation.get('results')
        uplift_scores = evaluation.get('uplift_scores')
        
        if results is not None:
            print("Model evaluation results:")
            for model, score in results.items():
                print(f"{model}: {score:.4f}")
        else:
            print("Evaluation results not found in checkpoint, reevaluating...")
            results, uplift_scores = evaluate_models(two_model, metalearners, X_val, y_val, t_val)
            checkpoint_data['evaluation'] = {'results': results, 'uplift_scores': uplift_scores}
            joblib.dump(checkpoint_data, checkpoint_file)
            
            print("Model evaluation results:")
            for model, score in results.items():
                print(f"{model}: {score:.4f}")
    
    #Step 9: Analyzing feature importance
    if start_step <= 9:
        print("Step 9: Analyzing feature importance...")
        importance_df = analyze_feature_importance(two_model, X_train, feature_cols)
        feature_shap_values, _, _, _, _ = shap_analysis(two_model, X_train, feature_cols)
        checkpoint_data['feature_importance'] = {
            'importance_df': importance_df,
            'feature_shap_values': feature_shap_values
        }
        joblib.dump(checkpoint_data, checkpoint_file)
    else:
        print("Step 9: Analyzing feature importance... [SKIPPED]")
        feature_importance = checkpoint_data.get('feature_importance', {})
        importance_df = feature_importance.get('importance_df')
        feature_shap_values = feature_importance.get('feature_shap_values')
        
        if importance_df is None or feature_shap_values is None:
            print("Feature importance not found in checkpoint, reanalyzing...")
            importance_df = analyze_feature_importance(two_model, X_train, feature_cols)
            feature_shap_values, _, _, _, _ = shap_analysis(two_model, X_train, feature_cols)
            checkpoint_data['feature_importance'] = {
                'importance_df': importance_df,
                'feature_shap_values': feature_shap_values
            }
            joblib.dump(checkpoint_data, checkpoint_file)
    
    #Step 10: Saving attribution results
    if start_step <= 10:
        print("Step 10: Saving attribution results...")
        save_attribution_results(importance_df, feature_shap_values, output_dir)
    else:
        print("Step 10: Saving attribution results... [SKIPPED]")
    
    #Step 11: Saving models
    if start_step <= 11:
        print("Step 11: Saving models...")
        save_models(two_model, metalearners, feature_cols, output_dir)
    else:
        print("Step 11: Saving models... [SKIPPED]")
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    #op dir
    os.makedirs('models', exist_ok=True)
    
    #Run pipeline with a sample of 1 million rows
    #Using GPU acceleration for model training
    #Start from step 6 to retrain models with GPU
    run_pipeline("criteo-uplift-v2.1.csv", sample_size=1000000, start_step=1)