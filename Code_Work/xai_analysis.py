# -*- coding: utf-8 -*-
"""Explainable AI (XAI) Analysis for IoT Communication Models"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')

class XAIAnalyzer:
    def __init__(self, data_path="semantic_features_antwerp.csv"):
        """Initialize XAI Analyzer"""
        self.data_path = data_path
        self.data = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = ["mean_rssi", "num_active_bs", "Latitude", "Longitude", "hour"]
        
    def load_and_prepare_data(self):
        """Load and prepare data for analysis"""
        print("Loading and preparing data...")
        self.data = pd.read_csv(self.data_path)
        
        # Create classification label
        self.data['label'] = ((self.data['mean_rssi'] > -110) & (self.data['num_active_bs'] >= 3)).astype(int)
        print(f"Label distribution:\n{self.data['label'].value_counts()}")
        
        # Prepare features
        X = self.data[self.feature_names]
        y = self.data['label']
        
        # Normalize features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
    def train_model(self):
        """Train Random Forest model"""
        print("\nTraining Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(self.X_train, self.y_train)
        
        # Calculate accuracy
        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test)
        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        
    def plot_feature_importance(self, save_path="xai_feature_importance.pdf"):
        """Plot feature importance from Random Forest"""
        print("\nPlotting feature importance...")
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [self.feature_names[i] for i in indices], rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title('Random Forest Feature Importance')
        
        # Add value labels on bars
        for bar, importance in zip(bars, importances[indices]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{importance:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def permutation_importance_analysis(self, save_path="xai_permutation_importance.pdf"):
        """Calculate and plot permutation importance"""
        print("\nCalculating permutation importance...")
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            self.model, self.X_test, self.y_test,
            n_repeats=10, random_state=42
        )
        
        # Sort features by importance
        sorted_idx = perm_importance.importances_mean.argsort()[::-1]
        
        plt.figure(figsize=(10, 6))
        boxes = plt.boxplot(
            [perm_importance.importances[i] for i in sorted_idx],
            labels=[self.feature_names[i] for i in sorted_idx],
            patch_artist=True
        )
        
        # Customize boxplot
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightyellow']
        for patch, color in zip(boxes['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.xlabel('Features')
        plt.ylabel('Permutation Importance')
        plt.title('Permutation Importance (Test Set)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def partial_dependence_analysis(self, save_path="xai_partial_dependence.pdf"):
        """Perform partial dependence analysis"""
        print("\nPerforming partial dependence analysis...")
        
        from sklearn.inspection import PartialDependenceDisplay
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        PartialDependenceDisplay.from_estimator(
            self.model, self.X_test, features=[0, 1, 2, 3, 4],
            feature_names=self.feature_names,
            ax=ax,
            grid_resolution=20
        )
        
        plt.suptitle('Partial Dependence Plots')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def lime_explanation(self, instance_idx=0, save_path="xai_lime_explanation.pdf"):
        """Generate LIME explanation for a specific instance"""
        print(f"\nGenerating LIME explanation for instance {instance_idx}...")
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train,
            feature_names=self.feature_names,
            class_names=['Poor Connection', 'Good Connection'],
            mode='classification',
            random_state=42
        )
        
        # Explain a specific instance
        exp = explainer.explain_instance(
            self.X_test[instance_idx],
            self.model.predict_proba,
            num_features=len(self.feature_names)
        )
        
        # Plot LIME explanation
        fig = exp.as_pyplot_figure()
        plt.title(f'LIME Explanation for Instance {instance_idx}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print explanation in console
        print("\nLIME Explanation:")
        print(f"Actual class: {self.y_test.iloc[instance_idx]}")
        print(f"Predicted probability for class 1: {self.model.predict_proba([self.X_test[instance_idx]])[0][1]:.3f}")
        
    def correlation_analysis(self, save_path="xai_correlation_heatmap.pdf"):
        """Plot correlation heatmap"""
        print("\nPlotting correlation heatmap...")
        
        # Create correlation matrix
        data_with_label = self.data[self.feature_names + ['label']]
        correlation_matrix = data_with_label.corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def feature_distribution_analysis(self, save_path="xai_feature_distributions.pdf"):
        """Plot feature distributions by class"""
        print("\nPlotting feature distributions by class...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, feature in enumerate(self.feature_names):
            for label in [0, 1]:
                subset = self.data[self.data['label'] == label]
                axes[i].hist(subset[feature], alpha=0.7, label=f'Class {label}', 
                           bins=20, density=True)
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplot
        axes[-1].set_visible(False)
        
        plt.suptitle('Feature Distributions by Class')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def run_complete_analysis(self):
        """Run complete XAI analysis"""
        print("=== Starting Complete XAI Analysis ===\n")
        
        self.load_and_prepare_data()
        self.train_model()
        
        # Generate all plots
        self.plot_feature_importance()
        self.permutation_importance_analysis()
        self.partial_dependence_analysis()
        self.lime_explanation(instance_idx=0)
        self.correlation_analysis()
        self.feature_distribution_analysis()
        
        print("\n=== XAI Analysis Completed ===")

def main():
    """Main function to run XAI analysis"""
    analyzer = XAIAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()