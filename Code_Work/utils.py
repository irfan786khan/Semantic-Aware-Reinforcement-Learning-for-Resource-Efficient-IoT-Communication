import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load dataset from CSV file"""
    data = pd.read_csv(file_path)
    print("Dataset Preview:")
    print(data.head())
    print("\nData Info:")
    print(data.info())
    return data

def normalize_features(data, features):
    """Normalize features using MinMaxScaler"""
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[features])
    return data_scaled.astype(np.float32), scaler

def split_data(X, y, test_size=0.3, random_state=42):
    """Split data into training and testing sets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def read_training_log(file_path):
    """Read training log file"""
    episodes = []
    rewards = []
    epsilons = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if "Episode" in line:
                parts = line.split(" - ")
                episode_num = int(parts[0].split("/")[0].replace("Episode ", ""))
                reward = float(parts[1].split(": ")[1])
                epsilon = float(parts[2].split(": ")[1])
                
                episodes.append(episode_num)
                rewards.append(reward)
                epsilons.append(epsilon)
    
    return episodes, rewards, epsilons