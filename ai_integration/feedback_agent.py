"""
AI-Driven Adaptive Modulation Controller
Bold (Unubold) - EEME30003 Major Project 1

This module implements a Deep Q-Learning agent that learns optimal modulation
selection policies for two-layer wireless image transmission.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import os


class DQNNetwork(nn.Module):
    """Deep Q-Network for modulation selection."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 64]):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[1], action_dim)
        )
        
    def forward(self, x):
        return self.network(x)


class ExperienceReplay:
    """Experience replay buffer for stable training."""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class AdaptiveModulationAgent:
    """Main RL agent for adaptive modulation control."""
    
    def __init__(self, state_dim=7, learning_rate=0.001, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        
        # Modulation options
        self.base_mods = [2, 4, 16]  # BPSK, QPSK, 16-QAM
        self.enh_mods = [4, 16, 64, 256]  # QPSK, 16-QAM, 64-QAM, 256-QAM
        
        # Generate all valid action combinations
        self.actions = []
        for base_mod in self.base_mods:
            for enh_mod in self.enh_mods:
                if enh_mod >= base_mod:
                    self.actions.append((base_mod, enh_mod))
        
        self.action_dim = len(self.actions)
        self.state_dim = state_dim
        
        # Neural networks
        self.policy_net = DQNNetwork(state_dim, self.action_dim)
        self.target_net = DQNNetwork(state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Training components
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ExperienceReplay(capacity=10000)
        self.gamma = gamma
        
        # Exploration strategy
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # State normalization
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
        # Training metrics
        self.training_history = {
            'loss': [],
            'avg_reward': [],
            'epsilon': []
        }
    
    def normalize_state(self, state, fit=False):
        """Normalize state features for stable learning."""
        state_reshaped = state.reshape(1, -1)
        
        if fit and not self.scaler_fitted:
            self.scaler.fit(state_reshaped)
            self.scaler_fitted = True
        
        if self.scaler_fitted:
            return self.scaler.transform(state_reshaped).flatten()
        else:
            return state
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            action_idx = random.randrange(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            action_idx = q_values.argmax().item()
        
        return action_idx, self.actions[action_idx]
    
    def calculate_reward(self, ssim, psnr, urgent_flag, prev_action, curr_action):
        """Multi-objective reward function."""
        # Quality component
        quality_reward = ssim + (psnr / 50.0)
        
        # Urgency penalty
        urgency_penalty = -1.0 if urgent_flag else 0.0
        
        # Stability component
        mod_levels = {2: 0, 4: 1, 16: 2, 64: 3, 256: 4}
        prev_base, prev_enh = prev_action
        curr_base, curr_enh = curr_action
        
        base_jump = abs(mod_levels.get(curr_base, 0) - mod_levels.get(prev_base, 0))
        enh_jump = abs(mod_levels.get(curr_enh, 0) - mod_levels.get(prev_enh, 0))
        
        stability_penalty = -0.1 * (base_jump + enh_jump)
        
        # Efficiency bonus
        efficiency_bonus = 0.0
        if ssim > 0.85:
            throughput_score = (mod_levels.get(curr_base, 0) + 
                              mod_levels.get(curr_enh, 0)) / 6.0
            efficiency_bonus = 0.3 * throughput_score
        
        total_reward = (quality_reward + urgency_penalty + 
                       stability_penalty + efficiency_bonus)
        
        return total_reward
    
    def train_step(self, batch_size=64):
        """Single training step using experience replay."""
        if len(self.memory) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Reduce exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath='feedback_agent_model.pth'):
        """Save trained model."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_history': self.training_history
        }, filepath)
        
        with open('feedback_agent_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"✓ Model saved to {filepath}")
    
    def load(self, filepath='feedback_agent_model.pth'):
        """Load trained model."""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_history = checkpoint['training_history']
        
        with open('feedback_agent_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
            self.scaler_fitted = True
        
        print(f"✓ Model loaded from {filepath}")


def train_agent(csv_path='ai_training_data.csv', episodes=100, save_path='feedback_agent_model.pth'):
    """Train the DQN agent on collected data."""
    
    print("="*70)
    print(" Training AI-Driven Adaptive Modulation Controller")
    print(" Bold (Unubold) - EEME30003 Major Project")
    print("="*70)
    
    # Load data
    print("\n[1/5] Loading training data...")
    if not os.path.exists(csv_path):
        print(f"  ✗ Error: {csv_path} not found!")
        print(f"  Please run generate_training_data.m in MATLAB first.")
        return None
    
    data = pd.read_csv(csv_path)
    print(f"  ✓ Loaded {len(data)} samples")
    print(f"  ✓ SNR range: {data['snr_db'].min():.1f} - {data['snr_db'].max():.1f} dB")
    print(f"  ✓ SSIM range: {data['ssim'].min():.3f} - {data['ssim'].max():.3f}")
    
    # Initialize agent
    print("\n[2/5] Initializing DQN agent...")
    agent = AdaptiveModulationAgent()
    print(f"  ✓ State dimension: {agent.state_dim}")
    print(f"  ✓ Action space: {agent.action_dim} modulation combinations")
    print(f"  ✓ Network: {agent.state_dim} → 128 → 64 → {agent.action_dim}")
    
    # Prepare episodes
    print("\n[3/5] Preparing training episodes...")
    episodes_data = []
    for (img_id, prof_id), group in data.groupby(['image_id', 'profile_id']):
        episodes_data.append(group.reset_index(drop=True))
    
    random.shuffle(episodes_data)
    
    split_idx = int(0.8 * len(episodes_data))
    train_episodes = episodes_data[:split_idx]
    val_episodes = episodes_data[split_idx:]
    
    print(f"  ✓ Training episodes: {len(train_episodes)}")
    print(f"  ✓ Validation episodes: {len(val_episodes)}")
    
    # Fit scaler
    all_train_states = []
    for episode in train_episodes:
        for _, row in episode.iterrows():
            state = np.array([
                row['snr_db'],
                row['base_evm'] if not pd.isna(row['base_evm']) else 0,
                row['enh_evm'] if not pd.isna(row['enh_evm']) else 0,
                row['base_mod'],
                row['enh_mod'],
                row['ssim'],
                row['psnr']
            ])
            all_train_states.append(state)
    
    agent.normalize_state(np.array(all_train_states[0]), fit=True)
    print("  ✓ State scaler fitted")
    
    # Training loop
    print("\n[4/5] Training agent...")
    target_update_freq = 10
    batch_size = 64
    
    for episode_idx in range(len(train_episodes)):
        episode = train_episodes[episode_idx]
        episode_reward = 0
        
        for i in range(len(episode) - 1):
            row = episode.iloc[i]
            state = np.array([
                row['snr_db'],
                row['base_evm'] if not pd.isna(row['base_evm']) else 0,
                row['enh_evm'] if not pd.isna(row['enh_evm']) else 0,
                row['base_mod'],
                row['enh_mod'],
                row['ssim'],
                row['psnr']
            ])
            state = agent.normalize_state(state)
            
            action_idx, action = agent.select_action(state, training=True)
            
            next_row = episode.iloc[i + 1]
            next_state = np.array([
                next_row['snr_db'],
                next_row['base_evm'] if not pd.isna(next_row['base_evm']) else 0,
                next_row['enh_evm'] if not pd.isna(next_row['enh_evm']) else 0,
                next_row['base_mod'],
                next_row['enh_mod'],
                next_row['ssim'],
                next_row['psnr']
            ])
            next_state = agent.normalize_state(next_state)
            
            prev_action = (int(row['base_mod']), int(row['enh_mod']))
            reward = agent.calculate_reward(
                next_row['ssim'],
                next_row['psnr'],
                next_row['urgent_flag'],
                prev_action,
                action
            )
            
            done = (i == len(episode) - 2)
            agent.memory.push(state, action_idx, reward, next_state, done)
            
            episode_reward += reward
            
            loss = agent.train_step(batch_size)
        
        if episode_idx % target_update_freq == 0:
            agent.update_target_network()
        
        agent.decay_epsilon()
        
        if episode_idx % 10 == 0:
            avg_reward = episode_reward / len(episode)
            agent.training_history['avg_reward'].append(avg_reward)
            agent.training_history['epsilon'].append(agent.epsilon)
            
            print(f"  Episode {episode_idx:3d}/{len(train_episodes)} | "
                  f"Avg Reward: {avg_reward:6.3f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    # Validation
    print("\n[5/5] Validating...")
    val_rewards = []
    
    for episode in val_episodes:
        episode_reward = 0
        for i in range(len(episode) - 1):
            row = episode.iloc[i]
            state = np.array([
                row['snr_db'],
                row['base_evm'] if not pd.isna(row['base_evm']) else 0,
                row['enh_evm'] if not pd.isna(row['enh_evm']) else 0,
                row['base_mod'],
                row['enh_mod'],
                row['ssim'],
                row['psnr']
            ])
            state = agent.normalize_state(state)
            
            action_idx, action = agent.select_action(state, training=False)
            
            next_row = episode.iloc[i + 1]
            prev_action = (int(row['base_mod']), int(row['enh_mod']))
            reward = agent.calculate_reward(
                next_row['ssim'],
                next_row['psnr'],
                next_row['urgent_flag'],
                prev_action,
                action
            )
            episode_reward += reward
        
        val_rewards.append(episode_reward / len(episode))
    
    print(f"  ✓ Validation avg reward: {np.mean(val_rewards):.3f} ± {np.std(val_rewards):.3f}")
    
    # Save
    print(f"\nSaving model to {save_path}...")
    agent.save(save_path)
    
    # Plot
    plot_training_history(agent.training_history)
    
    print("\n" + "="*70)
    print(" Training Complete!")
    print("="*70)
    
    return agent


def plot_training_history(history):
    """Visualize training progress."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    axes[0].plot(history['avg_reward'], label='Avg Episode Reward', linewidth=2)
    axes[0].set_xlabel('Training Step (×10 episodes)')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('DQN Training Progress - Reward')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(history['epsilon'], label='Epsilon', color='orange', linewidth=2)
    axes[1].set_xlabel('Training Step (×10 episodes)')
    axes[1].set_ylabel('Exploration Rate (ε)')
    axes[1].set_title('Exploration-Exploitation Tradeoff')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("  ✓ Training plots saved to training_history.png")
    plt.show()


if __name__ == "__main__":
    agent = train_agent(
        csv_path='ai_training_data.csv',
        episodes=100,
        save_path='feedback_agent_model.pth'
    )

    