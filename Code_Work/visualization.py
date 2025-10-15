import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def setup_plot_style():
    """Set up consistent plot style"""
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

def plot_training_progress(episodes, rewards, epsilons, save_path=None):
    """Plot DQN training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot total reward
    ax1.plot(episodes, rewards, label='Total Reward', color='tab:blue', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    
    # Plot epsilon
    ax2.plot(episodes, epsilons, label='Epsilon', color='tab:red', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_roc_curve(fpr, tpr, auc, title="ROC Curve", save_path=None):
    """Plot ROC curve"""
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_anomaly_map(df, save_path=None):
    """Plot anomaly detection results on map"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="Longitude", y="Latitude",
        hue="anomaly", palette={1: "blue", -1: "red"},
        data=df, alpha=0.7
    )
    plt.legend(title="Point Type", labels=["Normal", "Anomaly"])
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_feature_importance(clf, feature_names, save_path=None):
    """Plot feature importance from Random Forest"""
    importances = clf.feature_importances_
    
    plt.figure(figsize=(8, 4))
    sns.barplot(x=importances, y=feature_names)
    plt.title("Random Forest Feature Importances")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_combined_roc_curves(fpr_rf, tpr_rf, auc_rf, fpr_if, tpr_if, auc_if, save_path=None):
    """Plot combined ROC curves for comparison"""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve - Random Forest
    axs[0].plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'AUC = {auc_rf:.2f}')
    axs[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    axs[0].legend(loc="lower right")
    
    # ROC Curve - Isolation Forest
    axs[1].plot(fpr_if, tpr_if, color='blue', lw=2, label=f'AUC = {auc_if:.2f}')
    axs[1].plot([0, 1], [0, 1], linestyle="--", color="navy")
    axs[1].set_xlabel("False Positive Rate")
    axs[1].set_ylabel("True Positive Rate")
    axs[1].legend(loc="lower right")
    
    # Add subplot labels
    fig.text(0.23, 0.02, '(a) ROC Curve - Random Forest', ha='center', va='center', fontsize=12, fontweight='bold')
    fig.text(0.77, 0.02, '(b) ROC Curve - Isolation Forest', ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if save_path:
        plt.savefig(save_path)
    plt.show()