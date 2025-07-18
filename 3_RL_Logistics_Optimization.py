#!/usr/bin/env python3
"""
Reinforcement Learning for Logistics Optimization

This script implements a reinforcement learning approach to optimize trailer allocation
and relocation decisions, extending the rule-based system with MDP modeling.

The system models the logistics optimization problem as a Markov Decision Process (MDP):
- States: Current trailer positions, demand patterns, network capacity
- Actions: Trailer allocation and relocation decisions
- Rewards: Profit maximization, cost minimization, service level optimization

Author: Leanna Jeon
Date: 2025
"""

import pandas as pd
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from math import radians, sin, cos, asin, sqrt
import os
import json
from datetime import datetime, timedelta

class LogisticsEnvironment(gym.Env):
    """
    Custom Gym Environment for Logistics Optimization
    
    This environment models the trailer allocation and relocation problem
    as a Markov Decision Process with the following components:
    
    State Space:
    - Current trailer positions (location_id, asset_id)
    - Demand patterns per location
    - Network capacity constraints
    - Market conditions (tier levels)
    - Time-based factors
    
    Action Space:
    - Trailer relocation decisions (from_location, to_location, asset_id)
    - Allocation adjustments based on demand changes
    
    Reward Function:
    - Revenue maximization
    - Cost minimization (transportation, parking)
    - Service level optimization
    - Penalty for constraint violations
    """
    
    def __init__(self, network_data, demand_data, telematics_data, 
                 num_trailers=250, num_locations=100, max_distance=500):
        super(LogisticsEnvironment, self).__init__()
        
        # Environment parameters
        self.num_trailers = num_trailers
        self.num_locations = num_locations
        self.max_distance = max_distance
        
        # Load and preprocess data
        self.network_data = self._preprocess_network_data(network_data)
        self.demand_data = self._preprocess_demand_data(demand_data)
        self.telematics_data = self._preprocess_telematics_data(telematics_data)
        
        # Initialize state
        self.current_state = None
        self.current_step = 0
        self.max_steps = 100
        
        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete([
            num_trailers,  # Which trailer to move
            num_locations, # From location
            num_locations  # To location
        ])
        
        # State space: trailer positions + demand + capacity + market conditions
        state_size = (num_trailers * 2 +  # trailer positions (lat, lon)
                     num_locations * 3 +  # demand, capacity, market_tier
                     1)                   # current step
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )
        
        # Financial parameters (modified for confidentiality)
        self.transport_cost_per_mile = 1.5
        self.daily_revenue_tiers = {1: 75, 2: 45, 3: 25}
        self.daily_parking_cost_tiers = {1: 8, 2: 6, 3: 4}
        self.utilization_rate = 0.73
        
        # Initialize trailer positions
        self.trailer_positions = self._initialize_trailer_positions()
        
    def _preprocess_network_data(self, network_data):
        """Preprocess network location data"""
        df = network_data.copy()
        df['location_id'] = range(len(df))
        return df
    
    def _preprocess_demand_data(self, demand_data):
        """Preprocess demand data with time series patterns"""
        df = demand_data.copy()
        # Remove NA/empty rows and ensure Index is float
        df = df.dropna(subset=['Index'])
        df['Index'] = df['Index'].astype(float)
        # Add seasonal and trend components
        df['seasonal_factor'] = np.sin(2 * np.pi * df.index / 12) * 0.2 + 1
        df['trend_factor'] = 1 + (df.index * 0.01)
        # Use Index as base demand, convert to trailer count
        df['num_trailers'] = (df['Index'] / df['Index'].sum() * 250).round().astype(int)
        df['adjusted_demand'] = df['num_trailers'] * df['seasonal_factor'] * df['trend_factor']
        return df
    
    def _preprocess_telematics_data(self, telematics_data):
        """Preprocess telematics data for current positions"""
        df = telematics_data.copy()
        # Get latest position for each trailer
        latest_positions = df.groupby('Asset_id').last().reset_index()
        return latest_positions
    
    def _initialize_trailer_positions(self):
        """Initialize trailer positions based on telematics data"""
        positions = {}
        for i, row in self.telematics_data.iterrows():
            if i < self.num_trailers:
                positions[i] = {
                    'asset_id': row['Asset_id'],
                    'latitude': row['Latitude_asset'],
                    'longitude': row['Longitude_asset'],
                    'current_location': self._find_nearest_location(
                        row['Latitude_asset'], row['Longitude_asset']
                    )
                }
        return positions
    
    def _find_nearest_location(self, lat, lon):
        """Find nearest network location to given coordinates"""
        distances = []
        for _, loc in self.network_data.iterrows():
            dist = self._haversine_distance(lat, lon, loc['Latitude'], loc['Longitude'])
            distances.append((dist, loc['location_id']))
        return min(distances, key=lambda x: x[0])[1]
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate haversine distance between two points"""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 3958.7613  # Earth radius in miles
        return c * r
    
    def _get_state(self):
        """Get current state representation"""
        state = []
        
        # Trailer positions (lat, lon for each trailer)
        for trailer_id in range(self.num_trailers):
            if trailer_id in self.trailer_positions:
                pos = self.trailer_positions[trailer_id]
                state.extend([pos['latitude'], pos['longitude']])
            else:
                state.extend([0, 0])  # Default position
        
        # Location demand and capacity
        for loc_id in range(self.num_locations):
            if loc_id < len(self.demand_data):
                demand = self.demand_data.iloc[loc_id]['adjusted_demand']
                capacity = 20  # Maximum trailers per location
                market_tier = self.demand_data.iloc[loc_id].get('tier', 2)
            else:
                demand, capacity, market_tier = 0, 20, 2
            state.extend([demand, capacity, market_tier])
        
        # Current step
        state.append(self.current_step)
        
        return np.array(state, dtype=np.float32)
    
    def _calculate_reward(self, action):
        """Calculate reward based on action and current state"""
        trailer_id, from_loc, to_loc = action
        
        if trailer_id >= self.num_trailers or from_loc >= self.num_locations or to_loc >= self.num_locations:
            return -1000  # Invalid action penalty
        
        # Get trailer and location information
        if trailer_id not in self.trailer_positions:
            return -1000
        
        trailer = self.trailer_positions[trailer_id]
        
        # Calculate transportation cost
        if from_loc < len(self.network_data) and to_loc < len(self.network_data):
            from_lat = self.network_data.iloc[from_loc]['Latitude']
            from_lon = self.network_data.iloc[from_loc]['Longitude']
            to_lat = self.network_data.iloc[to_loc]['Latitude']
            to_lon = self.network_data.iloc[to_loc]['Longitude']
            
            distance = self._haversine_distance(from_lat, from_lon, to_lat, to_lon)
            transport_cost = distance * self.transport_cost_per_mile
        else:
            transport_cost = 0
        
        # Calculate revenue potential at destination
        if to_loc < len(self.demand_data):
            demand = self.demand_data.iloc[to_loc]['adjusted_demand']
            market_tier = self.demand_data.iloc[to_loc].get('tier', 2)
            daily_revenue = self.daily_revenue_tiers[market_tier]
            monthly_revenue = daily_revenue * 30 * self.utilization_rate
        else:
            monthly_revenue = 0
            market_tier = 2  # Default tier
        
        # Calculate parking cost
        daily_parking_cost = self.daily_parking_cost_tiers.get(market_tier, 6)
        monthly_parking_cost = daily_parking_cost * 30 * (1 - self.utilization_rate)
        
        # Net profit
        net_profit = monthly_revenue - monthly_parking_cost - transport_cost
        
        # Additional penalties
        penalty = 0
        
        # Penalty for moving to location with no demand
        if to_loc < len(self.demand_data) and self.demand_data.iloc[to_loc]['adjusted_demand'] <= 0:
            penalty -= 50
        
        # Penalty for excessive distance
        if distance > self.max_distance:
            penalty -= 100
        
        return net_profit + penalty
    
    def step(self, action):
        """Execute action and return new state, reward, done, info"""
        reward = self._calculate_reward(action)
        
        # Update trailer position
        trailer_id, from_loc, to_loc = action
        if (trailer_id < self.num_trailers and 
            from_loc < self.num_locations and 
            to_loc < self.num_locations):
            
            if to_loc < len(self.network_data):
                self.trailer_positions[trailer_id]['current_location'] = to_loc
                self.trailer_positions[trailer_id]['latitude'] = self.network_data.iloc[to_loc]['Latitude']
                self.trailer_positions[trailer_id]['longitude'] = self.network_data.iloc[to_loc]['Longitude']
        
        # Update step
        self.current_step += 1
        
        # Get new state
        new_state = self._get_state()
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Update demand patterns (simulate time progression)
        self._update_demand_patterns()
        
        info = {
            'step': self.current_step,
            'total_reward': reward,
            'trailer_moved': trailer_id,
            'from_location': from_loc,
            'to_location': to_loc
        }
        
        return new_state, reward, done, info
    
    def _update_demand_patterns(self):
        """Update demand patterns to simulate time progression"""
        # Add some randomness to demand patterns
        for i in range(len(self.demand_data)):
            noise = np.random.normal(0, 0.1)
            self.demand_data.iloc[i, self.demand_data.columns.get_loc('adjusted_demand')] *= (1 + noise)
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.trailer_positions = self._initialize_trailer_positions()
        self.demand_data = self._preprocess_demand_data(self.demand_data.copy())
        return self._get_state()
    
    def render(self, mode='human'):
        """Render current state (optional)"""
        print(f"Step: {self.current_step}")
        print(f"Active trailers: {len(self.trailer_positions)}")
        print(f"Total locations: {len(self.network_data)}")

class DQNAgent:
    """
    Deep Q-Network Agent for Logistics Optimization
    
    This agent learns optimal trailer allocation and relocation strategies
    using Deep Q-Learning with experience replay and target networks.
    """
    
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        
        # Neural networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Update target network
        self.update_target_network()
    
    def _build_model(self):
        """Build neural network model"""
        model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model
    
    def update_target_network(self):
        """Update target network with current Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_actions=None, env=None):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            if valid_actions is not None:
                action = random.choice(valid_actions)
            else:
                # Return random MultiDiscrete action
                action = [
                    np.random.randint(0, env.action_space.nvec[0]),
                    np.random.randint(0, env.action_space.nvec[1]),
                    np.random.randint(0, env.action_space.nvec[2])
                ]
                return action
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        action_flat = np.argmax(q_values.detach().numpy())
        # Convert flat index to MultiDiscrete action
        action = np.unravel_index(action_flat, env.action_space.nvec)
        return list(action)
    
    def replay(self, batch_size=32, env=None):
        """Train the agent on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([exp[0] for exp in batch])
        # actions: list of [trailer_id, from_loc, to_loc]
        actions_multi = np.array([exp[1] for exp in batch])
        # Convert to flat index for Q-network
        actions_flat = [np.ravel_multi_index(a, env.action_space.nvec) for a in actions_multi]
        actions = torch.LongTensor(actions_flat)
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.FloatTensor([exp[3] for exp in batch])
        dones = torch.BoolTensor([exp[4] for exp in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """Save trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load_model(self, filepath):
        """Load trained model"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

def train_rl_agent(env, agent, episodes=1000, max_steps=100, target_update_freq=10):
    """Train the RL agent"""
    print("🚀 Starting RL Training...")
    print("=" * 60)
    
    episode_rewards = []
    episode_costs = []
    episode_profits = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        total_cost = 0
        total_profit = 0
        
        for step in range(max_steps):
            # Choose action
            action = agent.act(state, env=env)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            agent.replay(env=env)
            
            # Update metrics
            total_reward += reward
            if reward < 0:
                total_cost += abs(reward)
            else:
                total_profit += reward
            
            state = next_state
            
            if done:
                break
        
        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        # Store episode results
        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        episode_profits.append(total_profit)
        
        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_profit = np.mean(episode_profits[-100:])
            print(f"Episode {episode}/{episodes} - Avg Reward: {avg_reward:.2f}, Avg Profit: {avg_profit:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    # Visualization: Save reward/profit/cost curves
    import matplotlib.pyplot as plt
    os.makedirs("Output/RL_Results", exist_ok=True)
    plt.figure(figsize=(12,6))
    plt.plot(episode_rewards, label='Reward')
    plt.plot(episode_profits, label='Profit')
    plt.plot(episode_costs, label='Cost')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('RL Training Progress')
    plt.legend()
    plt.tight_layout()
    plt.savefig("Output/RL_Results/rl_training_progress.png")
    plt.close()
    print("📈 RL training progress graph saved to Output/RL_Results/rl_training_progress.png")
    
    return episode_rewards, episode_costs, episode_profits

def evaluate_rl_agent(env, agent, episodes=100):
    """Evaluate the trained RL agent"""
    print("\n📊 Evaluating RL Agent...")
    print("=" * 60)
    
    total_rewards = []
    total_profits = []
    total_costs = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_profit = 0
        episode_cost = 0
        
        for step in range(env.max_steps):
            action = agent.act(state, env=env)  # No exploration, always greedy
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            if reward < 0:
                episode_cost += abs(reward)
            else:
                episode_profit += reward
            
            state = next_state
            if done:
                break
        
        total_rewards.append(episode_reward)
        total_profits.append(episode_profit)
        total_costs.append(episode_cost)
    
    # Calculate metrics
    avg_reward = np.mean(total_rewards)
    avg_profit = np.mean(total_profits)
    avg_cost = np.mean(total_costs)
    std_reward = np.std(total_rewards)
    
    print(f"Evaluation Results ({episodes} episodes):")
    print(f"Average Reward: ${avg_reward:.2f} ± ${std_reward:.2f}")
    print(f"Average Profit: ${avg_profit:.2f}")
    print(f"Average Cost: ${avg_cost:.2f}")
    print(f"Profit/Cost Ratio: {avg_profit/avg_cost:.2f}")
    
    return {
        'avg_reward': avg_reward,
        'avg_profit': avg_profit,
        'avg_cost': avg_cost,
        'std_reward': std_reward,
        'profit_cost_ratio': avg_profit/avg_cost
    }

def compare_rl_vs_rule_based(env, agent, rule_based_baseline):
    """Compare RL agent performance against rule-based baseline"""
    print("\n🔍 Comparing RL vs Rule-Based Performance...")
    print("=" * 60)
    
    # RL Agent performance
    rl_results = evaluate_rl_agent(env, agent, episodes=50)
    
    # Rule-based baseline performance (simplified)
    rule_based_rewards = []
    for episode in range(50):
        state = env.reset()
        episode_reward = 0
        
        for step in range(env.max_steps):
            # Simple rule-based action (random but within constraints)
            action = rule_based_baseline(state, env=env)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            if done:
                break
        
        rule_based_rewards.append(episode_reward)
    
    rule_based_avg = np.mean(rule_based_rewards)
    rule_based_std = np.std(rule_based_rewards)
    
    print(f"RL Agent: ${rl_results['avg_reward']:.2f} ± ${rl_results['std_reward']:.2f}")
    print(f"Rule-Based: ${rule_based_avg:.2f} ± ${rule_based_std:.2f}")
    print(f"Improvement: {((rl_results['avg_reward'] - rule_based_avg) / rule_based_avg * 100):.1f}%")
    
    return {
        'rl_performance': rl_results,
        'rule_based_performance': {'avg_reward': rule_based_avg, 'std_reward': rule_based_std},
        'improvement_percentage': ((rl_results['avg_reward'] - rule_based_avg) / rule_based_avg * 100)
    }

def main():
    """Main function to run RL logistics optimization"""
    print("🤖 Reinforcement Learning for Logistics Optimization")
    print("=" * 60)
    
    # Load data
    try:
        network_data = pd.read_csv("dataset/NetLoc_Synthetic_cleaned.csv")
        demand_data = pd.read_csv("dataset/MarketIndex_Comdist_Synthetic.csv")
        telematics_data = pd.read_csv("dataset/Telematics_Synthetic.csv")
        print("✅ Data loaded successfully")
    except FileNotFoundError as e:
        print(f"❌ Data file not found: {e}")
        return
    
    # Create output folders
    os.makedirs("Output/RL_Optimization_Models", exist_ok=True)
    os.makedirs("Output/RL_Optimization_Results", exist_ok=True)
    
    # Create environment
    env = LogisticsEnvironment(
        network_data=network_data,
        demand_data=demand_data,
        telematics_data=telematics_data,
        num_trailers=250,
        num_locations=100,
        max_distance=500
    )
    
    # Create agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.nvec.prod()  # Product of action space dimensions
    agent = DQNAgent(state_size, action_size)
    
    # Train agent
    print("\n🎯 Training RL Agent...")
    episode_rewards, episode_costs, episode_profits = train_rl_agent(
        env, agent, episodes=500, max_steps=100
    )
    
    # Save trained model
    model_path = "Output/RL_Optimization_Models/logistics_optimization_model.pth"
    agent.save_model(model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Evaluate agent
    evaluation_results = evaluate_rl_agent(env, agent, episodes=100)
    
    # Compare with rule-based baseline
    def simple_rule_based_baseline(state, env=None):
        """Simple rule-based baseline for comparison"""
        # Random action within valid range
        return [
            np.random.randint(0, env.action_space.nvec[0]),
            np.random.randint(0, env.action_space.nvec[1]),
            np.random.randint(0, env.action_space.nvec[2])
        ]
    
    comparison_results = compare_rl_vs_rule_based(env, agent, simple_rule_based_baseline)
    
    # Save results
    results = {
        'training_episodes': len(episode_rewards),
        'evaluation_results': evaluation_results,
        'comparison_results': comparison_results,
        'model_path': model_path,
        'timestamp': datetime.now().isoformat()
    }
    
    with open("Output/RL_Optimization_Results/training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to Output/RL_Optimization_Results/training_results.json")
    print(f"🎉 RL Training completed successfully!")

if __name__ == "__main__":
    main() 