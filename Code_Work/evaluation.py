import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

def evaluate_dqn_agent(env, policy_net, episodes=50, device=None):
    """Evaluate trained DQN agent"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nEvaluating trained DQN agent...")
    rewards = []
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state).unsqueeze(0).to(device)
                action = policy_net(state_tensor).argmax().item()
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
        print(f"Evaluation Episode {ep+1}: Reward = {total_reward:.2f}")
    
    print(f"\nAverage Evaluation Reward: {np.mean(rewards):.2f}")
    return rewards

def evaluate_classifier(clf, X_test, y_test):
    """Evaluate classifier performance"""
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    return y_pred, y_prob, accuracy

def evaluate_anomaly_detector(iso_forest, X_scaled, anomaly_labels):
    """Evaluate anomaly detection performance"""
    y_true = (anomaly_labels == -1).astype(int)
    anomaly_scores = -iso_forest.decision_function(X_scaled)
    auc = roc_auc_score(y_true, anomaly_scores)
    
    print(f"Anomaly distribution: {np.unique(anomaly_labels, return_counts=True)}")
    print(f"AUC Score: {auc:.4f}")
    
    return y_true, anomaly_scores, auc

def calculate_roc_metrics(y_true, y_scores):
    """Calculate ROC curve metrics"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    return fpr, tpr, auc