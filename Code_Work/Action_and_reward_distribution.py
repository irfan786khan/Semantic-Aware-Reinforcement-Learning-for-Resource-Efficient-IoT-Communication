# -*- coding: utf-8 -*-
"""Reward and Action Distribution Analysis on Test Data"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set publication quality style
plt.rcParams.update({
    'font.family': 'serif', 
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

class RewardActionAnalyzer:
    def __init__(self, data_path="semantic_features_antwerp.csv"):
        self.data_path = data_path
        self.data = None
        self.model = None
        self.feature_names = ["Mean RSSI", "Active Base Stations", "Latitude", "Longitude", "Hour"]
        self.original_features = ["mean_rssi", "num_active_bs", "Latitude", "Longitude", "hour"]
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
        self.action_names = ['Do Nothing', 'Move North', 'Move South', 'Move East', 'Move West']
    
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded successfully! Shape: {self.data.shape}")
            
            # Create target variable
            self.data['label'] = ((self.data['mean_rssi'] > -110) & 
                                (self.data['num_active_bs'] >= 3)).astype(int)
            
            print(f"Class distribution:\n{self.data['label'].value_counts()}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def train_model(self):
        """Train a model for action prediction"""
        print("\nTraining model...")
        
        # Prepare features
        X = self.data[self.original_features]
        y = self.data['label']
        
        # Normalize features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Store original test data for analysis
        self.test_data_original = self.data.iloc[self.y_test.index]
        self.X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        
        test_score = self.model.score(self.X_test, self.y_test)
        print(f"✅ Model trained. Test accuracy: {test_score:.4f}")
        
        return True
    
    def define_reward_function(self, state, action):
        """
        Define reward function based on state features and action
        Higher rewards for better connection quality and efficient actions
        """
        mean_rssi = state[0]  # Normalized value
        num_active_bs = state[1]  # Normalized value
        hour = state[4]  # Normalized value
        
        # Base reward from connection quality
        connection_reward = (mean_rssi * 50) + (num_active_bs * 20)
        
        # Time-based reward (prefer certain hours)
        time_reward = 10 if (hour > 0.3 and hour < 0.7) else 0  # Prefer 7AM-5PM
        
        # Action cost (penalize unnecessary movements)
        if action == 0:  # Do nothing - no cost
            action_cost = 0
        else:  # Movement actions have cost
            action_cost = 5
        
        # Combine rewards
        total_reward = connection_reward + time_reward - action_cost
        
        return total_reward
    
    def simulate_actions_on_test_data(self):
        """Simulate actions and calculate rewards for test data"""
        print("\nSimulating actions on test data...")
        
        n_samples = len(self.X_test)
        n_actions = 5
        
        # Initialize results storage
        self.rewards_matrix = np.zeros((n_samples, n_actions))
        self.optimal_actions = np.zeros(n_samples, dtype=int)
        self.optimal_rewards = np.zeros(n_samples)
        
        # Calculate rewards for each state-action pair
        for i in range(n_samples):
            state = self.X_test[i]
            rewards_for_state = []
            
            for action in range(n_actions):
                reward = self.define_reward_function(state, action)
                self.rewards_matrix[i, action] = reward
                rewards_for_state.append(reward)
            
            # Find optimal action for this state
            self.optimal_actions[i] = np.argmax(rewards_for_state)
            self.optimal_rewards[i] = np.max(rewards_for_state)
        
        print(f"✅ Reward simulation completed for {n_samples} test samples")
        print(f"Average optimal reward: {np.mean(self.optimal_rewards):.2f}")
        
        return True
    
    def plot_reward_distribution(self):
        """Plot distribution of rewards across test data"""
        print("\nCreating reward distribution plots...")
        
        fig = plt.figure(figsize=(16, 14))
        
        # Create subplots with adjusted spacing - more space at bottom for labels
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3, bottom=0.08, top=0.92)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Plot 1: Overall reward distribution
        all_rewards = self.rewards_matrix.flatten()
        ax1.hist(all_rewards, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Reward Value', fontsize=14)
        ax1.set_ylabel('Frequency', fontsize=14)
        ax1.grid(False)
        
        # Add statistics
        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        ax1.axvline(mean_reward, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_reward:.2f}')
        ax1.legend()
        
        # Plot 2: Optimal reward distribution
        ax2.hist(self.optimal_rewards, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Optimal Reward Value', fontsize=14)
        ax2.set_ylabel('Frequency', fontsize=14)
        ax2.grid(False)
        
        # Plot 3: Reward distribution by action
        action_rewards = [self.rewards_matrix[:, i] for i in range(5)]
        box_plot = ax3.boxplot(action_rewards, labels=self.action_names, patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax3.set_ylabel('Reward Value', fontsize=14)
        ax3.set_xlabel('Action', fontsize=14)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(False)
        
        # Plot 4: Cumulative reward distribution
        sorted_rewards = np.sort(all_rewards)
        cumulative_prob = np.arange(1, len(sorted_rewards) + 1) / len(sorted_rewards)
        
        ax4.plot(sorted_rewards, cumulative_prob, linewidth=2, color='purple')
        ax4.set_xlabel('Reward Value', fontsize=14)
        ax4.set_ylabel('Cumulative Probability', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        # Add subplot labels (a), (b), (c), (d) and titles below each plot
        fig.text(0.25, 0.46, '(a) Overall Reward Distribution', ha='center', va='center', 
                fontsize=16, fontweight='bold')
        fig.text(0.75, 0.46, '(b) Optimal Reward Distribution', ha='center', va='center', 
                fontsize=16, fontweight='bold')
        fig.text(0.25, 0.02, '(c) Reward Distribution by Action', ha='center', va='center', 
                fontsize=16, fontweight='bold')
        fig.text(0.75, 0.02, '(d) Cumulative Reward Distribution', ha='center', va='center', 
                fontsize=16, fontweight='bold')
        
        # Clean styling
        for ax in [ax1, ax2, ax3, ax4]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('Reward_Distribution_Analysis.pdf', bbox_inches='tight', dpi=300)
        plt.show()
        print("✅ Reward distribution analysis saved as 'Reward_Distribution_Analysis.pdf'")
    
    def plot_action_distribution(self):
        """Plot distribution of optimal actions - REMOVED UNWANTED PLOTS"""
        print("\nCreating action distribution plots...")
        
        fig = plt.figure(figsize=(12, 6))  # Reduced figure size for 2 plots
        
        # Create subplots with adjusted spacing - only 2 plots now
        gs = fig.add_gridspec(1, 2, hspace=0.4, wspace=0.3, bottom=0.15, top=0.85)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Plot 1: Average reward by action (KEPT)
        avg_rewards_by_action = [np.mean(self.rewards_matrix[:, i]) for i in range(5)]
        bars = ax1.bar(range(5), avg_rewards_by_action, color=self.colors, alpha=0.8)
        ax1.set_xlabel('Action', fontsize=14)
        ax1.set_ylabel('Average Reward', fontsize=14)
        ax1.set_xticks(range(5))
        ax1.set_xticklabels(self.action_names, rotation=45)
        
        # Add value labels on bars
        for bar, reward in zip(bars, avg_rewards_by_action):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{reward:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Action distribution by connection quality (KEPT)
        good_connection_mask = self.y_test == 1
        poor_connection_mask = self.y_test == 0
        
        good_actions = self.optimal_actions[good_connection_mask]
        poor_actions = self.optimal_actions[poor_connection_mask]
        
        good_counts = np.bincount(good_actions, minlength=5)
        poor_counts = np.bincount(poor_actions, minlength=5)
        
        x = np.arange(5)
        width = 0.35
        
        ax2.bar(x - width/2, good_counts, width, label='Good Connection', 
               color='lightgreen', alpha=0.8)
        ax2.bar(x + width/2, poor_counts, width, label='Poor Connection', 
               color='lightcoral', alpha=0.8)
        
        ax2.set_xlabel('Action', fontsize=14)
        ax2.set_ylabel('Count', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.action_names, rotation=45)
        ax2.legend()
        
        # Add subplot labels (a), (b) and titles below each plot
        fig.text(0.25, 0.02, '(a) Average Reward by Action', ha='center', va='center', 
                fontsize=16, fontweight='bold')
        fig.text(0.75, 0.02, '(b) Action Distribution by Connection Quality', ha='center', va='center', 
                fontsize=16, fontweight='bold')
        
        # Clean styling
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('Action_Distribution_Analysis.pdf', bbox_inches='tight', dpi=300)
        plt.show()
        print("✅ Action distribution analysis saved as 'Action_Distribution_Analysis.pdf'")
    
    def plot_reward_vs_features(self):
        """Plot relationship between rewards and feature values - REMOVED UNWANTED PLOT"""
        print("\nCreating reward vs features analysis...")
        
        fig = plt.figure(figsize=(15, 10))  # Adjusted figure size for 5 plots
        
        # Create subplots with adjusted spacing - 5 plots in 2x3 grid (last one empty)
        gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3, bottom=0.08, top=0.92)
        
        axes = []
        for i in range(5):  # Only 5 plots now
            axes.append(fig.add_subplot(gs[i//3, i%3]))
        
        features_to_plot = self.original_features[:4]  # First 4 features
        
        for i, feature in enumerate(features_to_plot):
            feature_values = self.test_data_original[feature].values
            scatter = axes[i].scatter(feature_values, self.optimal_rewards, 
                                    alpha=0.6, s=30, c=feature_values, cmap='viridis')
            
            axes[i].set_xlabel(f'{self.feature_names[i]}', fontsize=14)
            axes[i].set_ylabel('Optimal Reward', fontsize=14)
            
            # Add correlation coefficient
            correlation = np.corrcoef(feature_values, self.optimal_rewards)[0, 1]
            axes[i].text(0.05, 0.95, f'ρ = {correlation:.3f}', 
                        transform=axes[i].transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add colorbar
            plt.colorbar(scatter, ax=axes[i])
        
        # Plot 5: Reward distribution by hour (KEPT)
        hours = self.test_data_original['hour'].values
        unique_hours = np.unique(hours)
        hourly_rewards = [self.optimal_rewards[hours == h] for h in unique_hours]
        
        axes[4].boxplot(hourly_rewards, labels=unique_hours)
        axes[4].set_xlabel('Hour of Day', fontsize=14)
        axes[4].set_ylabel('Optimal Reward', fontsize=14)
        axes[4].tick_params(axis='x', rotation=45)
        
        # Hide the 6th subplot (REMOVED Effectiveness of Optimal Actions)
        axes.append(fig.add_subplot(gs[1, 2]))
        axes[5].set_visible(False)
        
        # Add subplot labels (a) to (e) and titles below each plot
        titles = [
            '(a) Reward vs Mean RSSI',
            '(b) Reward vs Active Base Stations', 
            '(c) Reward vs Latitude',
            '(d) Reward vs Longitude',
            '(e) Reward Distribution by Hour'
            # Removed: '(f) Effectiveness of Optimal Actions'
        ]
        
        positions = [
            (0.17, 0.46), (0.50, 0.46), (0.83, 0.46),
            (0.17, 0.02), (0.50, 0.02)
            # Removed: (0.83, 0.02)
        ]
        
        for title, pos in zip(titles, positions):
            fig.text(pos[0], pos[1], title, ha='center', va='center', 
                    fontsize=16, fontweight='bold')
        
        # Clean styling
        for ax in axes:
            if ax.get_visible():
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('Reward_Feature_Relationship.pdf', bbox_inches='tight', dpi=300)
        plt.show()
        print("✅ Reward vs features analysis saved as 'Reward_Feature_Relationship.pdf'")
    
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        
        # Reward statistics
        print("\n📊 REWARD STATISTICS:")
        print(f"Total test samples: {len(self.optimal_rewards)}")
        print(f"Mean optimal reward: {np.mean(self.optimal_rewards):.2f}")
        print(f"Std optimal reward: {np.std(self.optimal_rewards):.2f}")
        print(f"Min optimal reward: {np.min(self.optimal_rewards):.2f}")
        print(f"Max optimal reward: {np.max(self.optimal_rewards):.2f}")
        print(f"Median optimal reward: {np.median(self.optimal_rewards):.2f}")
        
        # Action statistics
        print("\n🎯 ACTION STATISTICS:")
        action_counts = np.bincount(self.optimal_actions, minlength=5)
        for action, count in enumerate(action_counts):
            percentage = (count / len(self.optimal_actions)) * 100
            print(f"{self.action_names[action]}: {count} samples ({percentage:.1f}%)")
        
        # Reward by action
        print("\n💰 REWARD BY ACTION:")
        for action in range(5):
            action_rewards = self.rewards_matrix[:, action]
            mean_reward = np.mean(action_rewards)
            std_reward = np.std(action_rewards)
            print(f"{self.action_names[action]}: Mean = {mean_reward:.2f}, Std = {std_reward:.2f}")
        
        # Connection quality analysis
        print("\n📶 CONNECTION QUALITY ANALYSIS:")
        good_connection_count = np.sum(self.y_test == 1)
        poor_connection_count = np.sum(self.y_test == 0)
        print(f"Good connections: {good_connection_count} samples")
        print(f"Poor connections: {poor_connection_count} samples")
        
        if good_connection_count > 0:
            good_rewards = self.optimal_rewards[self.y_test == 1]
            print(f"Mean reward (good connection): {np.mean(good_rewards):.2f}")
        
        if poor_connection_count > 0:
            poor_rewards = self.optimal_rewards[self.y_test == 0]
            print(f"Mean reward (poor connection): {np.mean(poor_rewards):.2f}")
    
    def run_complete_analysis(self):
        """Run complete reward and action distribution analysis"""
        print("="*60)
        print("REWARD AND ACTION DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Load data
        if not self.load_and_prepare_data():
            return
        
        # Train model
        if not self.train_model():
            return
        
        # Simulate actions and rewards
        if not self.simulate_actions_on_test_data():
            return
        
        # Generate plots
        print("\n" + "="*60)
        print("GENERATING ANALYSIS PLOTS")
        print("="*60)
        
        self.plot_reward_distribution()
        self.plot_action_distribution()
        self.plot_reward_vs_features()
        
        # Generate summary statistics
        self.generate_summary_statistics()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("Generated publication-ready plots:")
        print("1. Reward_Distribution_Analysis.pdf - 4 reward distribution plots")
        print("2. Action_Distribution_Analysis.pdf - 2 action analysis plots") 
        print("3. Reward_Feature_Relationship.pdf - 5 feature relationship plots")
        print("="*60)

# Run the analysis
if __name__ == "__main__":
    analyzer = RewardActionAnalyzer("semantic_features_antwerp.csv")
    analyzer.run_complete_analysis()