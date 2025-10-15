# -*- coding: utf-8 -*-
"""Main script to run the complete pipeline"""

import warnings
warnings.filterwarnings('ignore')

from preprocessing import preprocess_sigfox_data, prepare_classification_data, prepare_anomaly_data
from training import train_dqn, train_random_forest, train_isolation_forest
from evaluation import evaluate_dqn_agent, evaluate_classifier, evaluate_anomaly_detector, calculate_roc_metrics
from visualization import (plot_training_progress, plot_confusion_matrix, plot_roc_curve, 
                         plot_anomaly_map, plot_feature_importance, plot_combined_roc_curves)
from utils import load_data, normalize_features, split_data, read_training_log
from models import SimpleEnv

def main():
    print("=== Semantic-Aware Reinforcement Learning for Resource-Efficient IoT Communication ===\n")
    
    # Step 1: Preprocess data
    print("Step 1: Preprocessing data...")
    data = preprocess_sigfox_data(
        sigfox_path="sigfox_dataset_antwerp.csv",
        bs_mapping_path="sigfox_bs_mapping.csv",
        output_path="semantic_features_antwerp.csv"
    )
    
    # Step 2: DQN Training
    print("\nStep 2: DQN Training...")
    features = ["mean_rssi", "num_active_bs", "Latitude", "Longitude", "hour"]
    states, _ = normalize_features(data, features)
    
    policy_net, target_net, losses, total_rewards = train_dqn(states, num_episodes=2)
    
    # Evaluate DQN
    env = SimpleEnv(states, lambda state, action: state[0] * 10 + state[1] * 5 - action)
    dqn_rewards = evaluate_dqn_agent(env, policy_net, episodes=50)
    
    # Step 3: Classification with Random Forest
    print("\nStep 3: Random Forest Classification...")
    X_scaled, y, feature_names, scaler = prepare_classification_data(data)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y, test_size=0.5)
    
    rf_clf = train_random_forest(X_train, y_train)
    y_pred, y_prob, accuracy = evaluate_classifier(rf_clf, X_test, y_test)
    
    # Step 4: Anomaly Detection with Isolation Forest
    print("\nStep 4: Anomaly Detection...")
    X_scaled_anomaly, feature_names, scaler_anomaly = prepare_anomaly_data(data)
    iso_forest, anomaly_labels = train_isolation_forest(X_scaled_anomaly)
    
    data_with_anomalies = data.copy()
    data_with_anomalies["anomaly"] = anomaly_labels
    y_true_anomaly, anomaly_scores, auc_anomaly = evaluate_anomaly_detector(
        iso_forest, X_scaled_anomaly, anomaly_labels
    )
    
    # Step 5: Visualization
    print("\nStep 5: Generating visualizations...")
    
    # Plot training progress (if log file exists)
    try:
        episodes, rewards, epsilons = read_training_log('k.txt')
        plot_training_progress(episodes, rewards, epsilons, 'training_progress.pdf')
    except FileNotFoundError:
        print("Training log file not found, skipping training progress plot")
    
    # Classification plots
    plot_confusion_matrix(y_test, y_pred, 'confusion_matrix.pdf')
    plot_feature_importance(rf_clf, feature_names, 'feature_importance.pdf')
    
    # Anomaly detection plots
    plot_anomaly_map(data_with_anomalies, 'anomaly_map.pdf')
    
    # ROC curves
    fpr_rf, tpr_rf, auc_rf = calculate_roc_metrics(y_test, y_prob)
    plot_roc_curve(fpr_rf, tpr_rf, auc_rf, save_path='RFroc_curve.pdf')
    
    fpr_if, tpr_if, auc_if = calculate_roc_metrics(y_true_anomaly, anomaly_scores)
    plot_roc_curve(fpr_if, tpr_if, auc_if, save_path='IFroc_curve.pdf')
    
    # Combined ROC curves
    plot_combined_roc_curves(fpr_rf, tpr_rf, auc_rf, fpr_if, tpr_if, auc_if, 
                           'combined_ROC_curves_with_labels.pdf')
    
    print("\n=== Pipeline completed successfully! ===")

if __name__ == "__main__":
    main()