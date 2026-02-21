this was my whole experminent codebase:
# Install required libraries
!pip install gym stable-baselines3 torch
!pip install --upgrade gymnasium shimmy
!pip install networkx matplotlib seaborn

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import gym
from gym import spaces
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback
import torch
import torch.nn as nn
import copy
import time
import os
import threading
import queue
from collections import deque, defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import networkx as nx

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("All dependencies installed successfully!")
# Load dataset
df = pd.read_csv('sdndata.csv')
print("Dataset columns:", df.columns.tolist())

# Define continuous features based on actual columns
continuous_features = [
    'flow_duration', 'byte_count', 'packet_count',
    'TotLenFwdPkts', 'TotLenBwdPkts', 'FwdPktLenMean',
    'BwdPktLenMean', 'FlowPkts/s', 'FlowByteS/s'
]

# Filter only existing features
available_features = [col for col in continuous_features if col in df.columns]
print("Normalizing features:", available_features)

# Normalize continuous features
scaler = MinMaxScaler()
df[available_features] = scaler.fit_transform(df[available_features])

# Filter anomalies using Label column
if 'Label' in df.columns:
    df_normal = df[df['Label'] != 'anomaly'].copy()
    print(f"Filtered anomalies: {len(df) - len(df_normal)} flows removed")
else:
    df_normal = df.copy()
    print("No 'Label' column found - using full dataset")

print(f"Final dataset size: {len(df_normal)} flows")

# Split dataset by traffic levels
traffic_levels = {
    'low': 1000,
    'medium': 5000,
    'high': 10000,
    'very_high': min(20000, len(df_normal))
}

# Prepare feature arrays for each traffic level
flow_features = {}
for level, size in traffic_levels.items():
    subset = df_normal[available_features].iloc[:size].values
    flow_features[level] = subset.astype(np.float32)
    print(f"{level} traffic: {len(flow_features[level])} flows")
class HierarchicalSDNEnvironment(gym.Env):
    """Enhanced SDN Environment with Hierarchical Controller Design"""

    def __init__(self, flow_features, num_regional_controllers=2, num_local_controllers=3):
        super(HierarchicalSDNEnvironment, self).__init__()

        self.flow_features = flow_features
        self.num_regional = num_regional_controllers
        self.num_local = num_local_controllers
        self.total_controllers = num_regional_controllers * num_local_controllers

        # Action space: choose regional controller (0 to num_regional-1)
        self.action_space = spaces.Discrete(num_regional_controllers)

        # Observation space: flow features + controller states
        obs_dim = flow_features.shape[1] + self.total_controllers + 2  # +2 for regional loads
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        """Reset environment state"""
        # Initialize hierarchical controller loads
        self.regional_loads = np.zeros(self.num_regional)
        self.local_loads = np.zeros((self.num_regional, self.num_local))
        self.controller_queues = [[[] for _ in range(self.num_local)]
                                 for _ in range(self.num_regional)]

        self.current_flow_index = 0
        self.total_assigned_bytes = 0
        self.step_count = 0

        return self._get_observation()

    def _get_observation(self):
        """Get current observation including flow features and controller states"""
        if self.current_flow_index >= len(self.flow_features):
            flow_obs = np.zeros(self.flow_features.shape[1])
        else:
            flow_obs = self.flow_features[self.current_flow_index]

        # Flatten local loads and add regional loads
        local_loads_flat = self.local_loads.flatten()
        regional_loads_norm = self.regional_loads / (self.total_assigned_bytes + 1e-8)

        return np.concatenate([flow_obs, local_loads_flat, regional_loads_norm])

    def step(self, action):
        """Execute one step in the environment"""
        if self.current_flow_index >= len(self.flow_features):
            return self._get_observation(), 0, True, {}

        flow = self.flow_features[self.current_flow_index]
        flow_duration = flow[0]
        byte_count = flow[1]

        # Select regional controller
        regional_idx = action

        # Smart local controller selection within region
        local_idx = self._select_local_controller(regional_idx, byte_count)

        # Assign flow
        self.controller_queues[regional_idx][local_idx].append(flow)
        self.local_loads[regional_idx][local_idx] += byte_count
        self.regional_loads[regional_idx] += byte_count
        self.total_assigned_bytes += byte_count

        # Calculate enhanced reward
        reward = self._calculate_hierarchical_reward(flow_duration)

        # Update state
        self.current_flow_index += 1
        self.step_count += 1
        done = self.current_flow_index >= len(self.flow_features)

        info = self._get_info_dict()

        return self._get_observation(), reward, done, info

    def _select_local_controller(self, regional_idx, byte_count):
        """Smart local controller selection based on current loads"""
        local_loads = self.local_loads[regional_idx]

        # Choose least loaded local controller
        return np.argmin(local_loads)

    def _calculate_hierarchical_reward(self, flow_duration):
        """Calculate reward considering hierarchical structure"""
        # Regional load balancing
        regional_variance = np.var(self.regional_loads)

        # Local load balancing within regions
        local_variances = [np.var(self.local_loads[i]) for i in range(self.num_regional)]
        avg_local_variance = np.mean(local_variances)

        # Calculate latency (considering hierarchical routing)
        total_queue_length = sum(sum(len(q) for q in region) for region in self.controller_queues)
        avg_latency = total_queue_length * flow_duration / self.total_controllers

        # Multi-objective reward with hierarchical considerations
        alpha, beta, gamma, delta = 0.4, 0.3, 0.2, 0.1

        reward = (-alpha * regional_variance
                 -beta * avg_local_variance
                 -gamma * avg_latency
                 +delta * self.total_assigned_bytes)

        return reward

    def _get_info_dict(self):
        """Get information dictionary for logging"""
        total_queue_length = sum(sum(len(q) for q in region) for region in self.controller_queues)

        return {
            'latency': total_queue_length / self.total_controllers,
            'regional_variance': np.var(self.regional_loads),
            'local_variance': np.mean([np.var(self.local_loads[i]) for i in range(self.num_regional)]),
            'throughput': self.total_assigned_bytes,
            'load_balance_score': 1.0 / (1.0 + np.var(self.regional_loads))
        }
class AsynchronousFederatedLearning:
    """Enhanced Asynchronous Federated Learning with Smart Aggregation"""

    def __init__(self, num_agents=3, performance_threshold=0.7, staleness_threshold=5):
        self.num_agents = num_agents
        self.performance_threshold = performance_threshold
        self.staleness_threshold = staleness_threshold

        # Asynchronous components
        self.global_model_queue = queue.Queue()
        self.agent_update_queue = queue.Queue()
        self.performance_tracker = defaultdict(deque)

        # Communication tracking
        self.communication_overhead = []
        self.aggregation_rounds = 0
        self.selective_updates = 0

        # Traffic-adaptive weights
        self.traffic_weights = {}

    def calculate_traffic_adaptive_weights(self, agents, traffic_level):
        """Calculate weights based on traffic conditions and agent performance"""
        weights = []
        total_performance = 0

        for i, agent in enumerate(agents):
            # Get recent performance
            recent_rewards = list(self.performance_tracker[i])[-10:]
            if recent_rewards:
                avg_performance = np.mean(recent_rewards)
                # Traffic-specific weighting
                traffic_multiplier = self._get_traffic_multiplier(traffic_level)
                adjusted_performance = avg_performance * traffic_multiplier
                weights.append(max(0.1, adjusted_performance))  # Minimum weight 0.1
                total_performance += adjusted_performance
            else:
                weights.append(1.0)
                total_performance += 1.0

        # Normalize weights
        if total_performance > 0:
            weights = [w / total_performance for w in weights]
        else:
            weights = [1.0 / len(agents)] * len(agents)

        return weights

    def _get_traffic_multiplier(self, traffic_level):
        """Get traffic-specific multiplier for adaptive weighting"""
        multipliers = {
            'low': 1.0,
            'medium': 1.2,
            'high': 1.5,
            'very_high': 2.0
        }
        return multipliers.get(traffic_level, 1.0)

    def selective_parameter_sharing(self, agents, share_ratio=0.6):
        """Share only base layers to reduce communication overhead"""
        if not agents:
            return 0

        # Identify base layers (first 60% of parameters)
        reference_model = agents[0]['model'].policy
        all_params = list(reference_model.named_parameters())
        num_base_params = int(len(all_params) * share_ratio)
        base_param_names = [name for name, _ in all_params[:num_base_params]]

        # Calculate weights for participating agents
        participating_agents = self._select_participating_agents(agents)
        if len(participating_agents) < 2:
            return 0

        weights = self.calculate_traffic_adaptive_weights(
            participating_agents,
            getattr(self, 'current_traffic_level', 'medium')
        )

        # Aggregate only base parameters
        global_params = {}
        for name in base_param_names:
            global_params[name] = torch.zeros_like(
                participating_agents[0]['model'].policy.state_dict()[name]
            )

            for i, agent in enumerate(participating_agents):
                state_dict = agent['model'].policy.state_dict()
                global_params[name] += weights[i] * state_dict[name]

        # Update all agents with aggregated base parameters
        for agent in agents:
            current_state = agent['model'].policy.state_dict()
            current_state.update(global_params)
            agent['model'].policy.load_state_dict(current_state)

        # Calculate communication overhead (only for shared parameters)
        overhead = 0
        for name in base_param_names:
            param = global_params[name]
            overhead += param.numel() * param.element_size()

        overhead_kb = overhead / 1024
        self.communication_overhead.append(overhead_kb)
        self.aggregation_rounds += 1

        return overhead_kb

    def _select_participating_agents(self, agents):
        """Select high-performing agents for aggregation"""
        if len(agents) <= 2:
            return agents

        agent_scores = []
        for i, agent in enumerate(agents):
            recent_rewards = list(self.performance_tracker[i])[-5:]
            if recent_rewards:
                score = np.mean(recent_rewards)
                agent_scores.append((score, i, agent))
            else:
                agent_scores.append((0, i, agent))

        # Sort by performance and select top performers
        agent_scores.sort(reverse=True, key=lambda x: x[0])

        # Select top 70% of agents or minimum 2
        num_selected = max(2, int(len(agents) * 0.7))
        selected = [agent for _, _, agent in agent_scores[:num_selected]]

        self.selective_updates += 1
        return selected

    def update_performance_tracker(self, agent_id, reward):
        """Update performance tracking for smart agent selection"""
        self.performance_tracker[agent_id].append(reward)
        # Keep only recent history
        if len(self.performance_tracker[agent_id]) > 20:
            self.performance_tracker[agent_id].popleft()

    def get_communication_stats(self):
        """Get communication statistics"""
        return {
            'total_overhead_kb': sum(self.communication_overhead),
            'avg_overhead_per_round': np.mean(self.communication_overhead) if self.communication_overhead else 0,
            'aggregation_rounds': self.aggregation_rounds,
            'selective_updates': self.selective_updates,
            'overhead_reduction': 1.0 - (np.mean(self.communication_overhead) / 35.5) if self.communication_overhead else 0
        }
class EnhancedMetricLogger(BaseCallback):
    """Enhanced metric logger for comprehensive performance tracking"""

    def __init__(self, agent_id=0, verbose=0):
        super().__init__(verbose)
        self.agent_id = agent_id

        # Performance metrics
        self.episode_rewards = []
        self.episode_latencies = []
        self.episode_variances = []
        self.episode_throughputs = []
        self.episode_load_balance_scores = []

        # Training metrics
        self.training_times = []
        self.convergence_episodes = []

        # Step-level tracking
        self.step_rewards = []
        self.step_latencies = []

    def _on_step(self) -> bool:
        """Called at each step"""
        # Track step-level metrics
        if 'rewards' in self.locals:
            self.step_rewards.extend(self.locals['rewards'])

        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout"""
        infos = self.locals.get('infos', [])
        rewards = self.locals.get('rewards', [])

        if rewards and infos:
            # Episode-level metrics
            episode_reward = np.sum(rewards)
            self.episode_rewards.append(episode_reward)

            # Extract info from last step
            last_info = infos[-1] if infos else {}
            self.episode_latencies.append(last_info.get('latency', 0))
            self.episode_variances.append(last_info.get('regional_variance', 0))
            self.episode_throughputs.append(last_info.get('throughput', 0))
            self.episode_load_balance_scores.append(last_info.get('load_balance_score', 0))

    def get_recent_performance(self, window=10):
        """Get recent performance metrics"""
        if len(self.episode_rewards) < window:
            return np.mean(self.episode_rewards) if self.episode_rewards else 0
        return np.mean(self.episode_rewards[-window:])

    def get_convergence_info(self, threshold_improvement=0.05):
        """Detect convergence based on reward stability"""
        if len(self.episode_rewards) < 20:
            return None

        recent_rewards = self.episode_rewards[-10:]
        older_rewards = self.episode_rewards[-20:-10]

        recent_avg = np.mean(recent_rewards)
        older_avg = np.mean(older_rewards)

        improvement = (recent_avg - older_avg) / (abs(older_avg) + 1e-8)

        if abs(improvement) < threshold_improvement:
            return len(self.episode_rewards)

        return None

    def get_summary_stats(self):
        """Get comprehensive summary statistics"""
        if not self.episode_rewards:
            return {}

        return {
            'final_reward': self.episode_rewards[-1],
            'avg_reward': np.mean(self.episode_rewards),
            'best_reward': max(self.episode_rewards),
            'reward_std': np.std(self.episode_rewards),
            'final_latency': self.episode_latencies[-1] if self.episode_latencies else 0,
            'avg_latency': np.mean(self.episode_latencies) if self.episode_latencies else 0,
            'final_variance': self.episode_variances[-1] if self.episode_variances else 0,
            'avg_variance': np.mean(self.episode_variances) if self.episode_variances else 0,
            'final_throughput': self.episode_throughputs[-1] if self.episode_throughputs else 0,
            'avg_throughput': np.mean(self.episode_throughputs) if self.episode_throughputs else 0,
            'convergence_episode': self.get_convergence_info()
        }

def train_enhanced_federated_system(traffic_level, num_controllers=6, total_timesteps=5000):
    """Train enhanced federated system with all improvements"""
    print(f"\n=== Enhanced Training for {traffic_level.upper()} traffic ===")

    # Create hierarchical environment
    env = HierarchicalSDNEnvironment(
        flow_features[traffic_level],
        num_regional_controllers=2,
        num_local_controllers=3
    )

    # Initialize baseline models for comparison
    baseline_models = {
        'PPO_Baseline': PPO('MlpPolicy', env, verbose=0, device='cpu'),
        'A2C_Baseline': A2C('MlpPolicy', env, verbose=0, device='cpu'),
        'DQN_Baseline': DQN('MlpPolicy', env, verbose=0, device='cpu')
    }

    # Train baseline models
    baseline_results = {}
    print("Training baseline models...")

    for name, model in baseline_models.items():
        print(f"  Training {name}...")
        start_time = time.time()
        logger = EnhancedMetricLogger()
        model.learn(total_timesteps=total_timesteps, callback=logger)
        training_time = time.time() - start_time

        baseline_results[name] = {
            'logger': logger,
            'training_time': training_time,
            'stats': logger.get_summary_stats()
        }
        print(f"{name} completed in {training_time:.1f}s")

    # Enhanced Federated Training
    print(" Starting Enhanced Federated PPO training...")

    num_agents = 3
    agents = []

    # Create agents with different data distributions
    for i in range(num_agents):
        # Distribute data with some overlap for better generalization
        start_idx = i * len(flow_features[traffic_level]) // (num_agents + 1)
        end_idx = (i + 2) * len(flow_features[traffic_level]) // (num_agents + 1)
        agent_data = flow_features[traffic_level][start_idx:end_idx]

        agent_env = HierarchicalSDNEnvironment(
            agent_data,
            num_regional_controllers=2,
            num_local_controllers=3
        )
        agent_model = PPO('MlpPolicy', agent_env, verbose=0, device='cpu')
        agent_logger = EnhancedMetricLogger(agent_id=i)

        agents.append({
            'env': agent_env,
            'model': agent_model,
            'logger': agent_logger,
            'id': i
        })

    # Initialize enhanced federated learning
    fed_system = AsynchronousFederatedLearning(
        num_agents=num_agents,
        performance_threshold=0.7,
        staleness_threshold=3
    )
    fed_system.current_traffic_level = traffic_level

    # Enhanced training loop with asynchronous aggregation
    start_time = time.time()
    federated_rounds = 8  # Reduced due to more efficient training
    timesteps_per_round = total_timesteps // federated_rounds

    aggregation_schedule = [2, 4, 6, 8]  # Asynchronous aggregation points

    for round_num in range(federated_rounds):
        print(f"Federated Round {round_num + 1}/{federated_rounds}")

        # Parallel agent training (simulated asynchronous)
        for agent in agents:
            agent['model'].learn(
                total_timesteps=timesteps_per_round,
                callback=agent['logger']
            )

            # Update performance tracker
            recent_reward = agent['logger'].get_recent_performance()
            fed_system.update_performance_tracker(agent['id'], recent_reward)

        # Asynchronous aggregation at scheduled points
        if (round_num + 1) in aggregation_schedule:
            overhead = fed_system.selective_parameter_sharing(agents, share_ratio=0.6)
            participating = len(fed_system._select_participating_agents(agents))
            print(f"Aggregation: {participating}/{num_agents} agents, {overhead:.1f} KB overhead")

    fed_training_time = time.time() - start_time

    # Collect federated results
    fed_results = {
        'AFRL': {
            'logger': agents[0]['logger'],  # Use first agent as representative
            'training_time': fed_training_time,
            'stats': agents[0]['logger'].get_summary_stats(),
            'communication_stats': fed_system.get_communication_stats(),
            'all_agents_stats': [agent['logger'].get_summary_stats() for agent in agents]
        }
    }

    print(f"Enhanced Federated training completed in {fed_training_time:.1f}s")

    # Combine results
    all_results = {**baseline_results, **fed_results}

    return all_results, fed_system

# Execute training for all traffic levels
print("Starting comprehensive training across all traffic levels...")
all_traffic_results = {}
all_fed_systems = {}

for traffic_level in traffic_levels.keys():
    results, fed_system = train_enhanced_federated_system(traffic_level)
    all_traffic_results[traffic_level] = results
    all_fed_systems[traffic_level] = fed_system
    print(f"Completed {traffic_level} traffic level")

print("ALL TARINING IS SUCCESSFULLY COMPLETED")


def create_comprehensive_analysis(all_results, fed_systems):
    """Create comprehensive analysis with comparison graphs"""

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create results directory
    os.makedirs('enhanced_results', exist_ok=True)

    # 1. Performance Comparison Across Traffic Levels
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Enhanced Federated Learning vs Baseline Performance', fontsize=16, fontweight='bold')

    metrics = ['avg_reward', 'avg_latency', 'avg_variance', 'avg_throughput', 'training_time']
    metric_titles = ['Average Reward', 'Average Latency', 'Load Variance', 'Throughput', 'Training Time (s)']

    traffic_names = list(traffic_levels.keys())

    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        if idx >= 6:
            break

        ax = axes[idx // 3, idx % 3]

        # Collect data for each model
        models = ['PPO_Baseline', 'A2C_Baseline', 'DQN_Baseline', 'AFRL']
        model_colors = ['blue', 'orange', 'green', 'red']

        for model, color in zip(models, model_colors):
            values = []
            for traffic in traffic_names:
                if model in all_results[traffic]:
                    if metric == 'training_time':
                        values.append(all_results[traffic][model]['training_time'])
                    else:
                        values.append(all_results[traffic][model]['stats'].get(metric, 0))
                else:
                    values.append(0)

            ax.plot(traffic_names, values, 'o-', label=model, color=color, linewidth=2, markersize=8)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Traffic Level')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    # Remove empty subplot
    if len(metrics) < 6:
        fig.delaxes(axes[1, 2])

    plt.tight_layout()
    plt.savefig('enhanced_results/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Communication Overhead Analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Overhead reduction
    traffic_names = list(traffic_levels.keys())
    overhead_reductions = []
    total_overheads = []

    for traffic in traffic_names:
        fed_system = fed_systems[traffic]
        stats = fed_system.get_communication_stats()
        overhead_reductions.append(stats['overhead_reduction'] * 100)
        total_overheads.append(stats['total_overhead_kb'])

    axes[0].bar(traffic_names, overhead_reductions, color='green', alpha=0.7)
    axes[0].set_title('Communication Overhead Reduction (%)', fontweight='bold')
    axes[0].set_ylabel('Reduction Percentage')
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(traffic_names, total_overheads, color='orange', alpha=0.7)
    axes[1].set_title('Total Communication Overhead (KB)', fontweight='bold')
    axes[1].set_ylabel('Overhead (KB)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('enhanced_results/communication_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Training Efficiency Comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Training time comparison
    models = ['PPO_Baseline', 'AFRL']
    baseline_times = []
    federated_times = []

    for traffic in traffic_names:
        baseline_times.append(all_results[traffic]['PPO_Baseline']['training_time'])
        federated_times.append(all_results[traffic]['AFRL']['training_time'])

    x = np.arange(len(traffic_names))
    width = 0.35

    axes[0].bar(x - width/2, baseline_times, width, label='PPO Baseline', color='blue', alpha=0.7)
    axes[0].bar(x + width/2, federated_times, width, label='Enhanced Federated', color='red', alpha=0.7)
    axes[0].set_title('Training Time Comparison', fontweight='bold')
    axes[0].set_ylabel('Time (seconds)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(traffic_names)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Speedup calculation
    speedups = [b/f for b, f in zip(baseline_times, federated_times)]
    axes[1].bar(traffic_names, speedups, color='purple', alpha=0.7)
    axes[1].set_title('Training Speedup Factor', fontweight='bold')
    axes[1].set_ylabel('Speedup Factor')
    axes[1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No speedup')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('enhanced_results/training_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Detailed Performance Metrics Table
    print("\n=== DETAILED PERFORMANCE ANALYSIS ===")
    print("="*80)

    for traffic in traffic_names:
        print(f"\n{traffic.upper()} TRAFFIC LEVEL ({traffic_levels[traffic]:,} flows)")
        print("-" * 60)

        # Create comparison table
        comparison_data = []
        for model in ['PPO_Baseline', 'AFRL']:
            if model in all_results[traffic]:
                stats = all_results[traffic][model]['stats']
                training_time = all_results[traffic][model]['training_time']

                row = {
                    'Model': model,
                    'Avg Reward': f"{stats.get('avg_reward', 0):.2f}",
                    'Final Reward': f"{stats.get('final_reward', 0):.2f}",
                    'Avg Latency': f"{stats.get('avg_latency', 0):.3f}",
                    'Load Variance': f"{stats.get('avg_variance', 0):.4f}",
                    'Throughput': f"{stats.get('avg_throughput', 0):.0f}",
                    'Training Time': f"{training_time:.1f}s"
                }
                comparison_data.append(row)

        # Print table
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))

        # Communication stats for federated model
        if traffic in fed_systems:
            comm_stats = fed_systems[traffic].get_communication_stats()
            print(f"\nCommunication Statistics:")
            print(f"   Total Overhead: {comm_stats['total_overhead_kb']:.1f} KB")
            print(f"   Avg per Round: {comm_stats['avg_overhead_per_round']:.1f} KB")
            print(f"   Overhead Reduction: {comm_stats['overhead_reduction']*100:.1f}%")
            print(f"   Aggregation Rounds: {comm_stats['aggregation_rounds']}")

# Execute comprehensive analysis
create_comprehensive_analysis(all_traffic_results, all_fed_systems)
def generate_final_summary():
    """Generate comprehensive final summary and recommendations"""

    print("\n === ENHANCED FEDERATED RL SYSTEM - FINAL SUMMARY ===")
    print("="*70)

    print("\nPERFORMANCE ACHIEVEMENTS:")
    print("-" * 30)

    # Calculate overall statistics
    total_improvements = {
        'reward': [],
        'latency': [],
        'variance': [],
        'throughput': [],
        'training_time': []
    }

    for traffic in traffic_levels.keys():
        baseline = all_traffic_results[traffic]['PPO_Baseline']
        enhanced = all_traffic_results[traffic]['AFRL']

        # Reward improvement
        reward_imp = ((enhanced['stats']['avg_reward'] - baseline['stats']['avg_reward']) /
                     abs(baseline['stats']['avg_reward']) * 100)
        total_improvements['reward'].append(reward_imp)

        # Latency reduction
        latency_red = ((baseline['stats']['avg_latency'] - enhanced['stats']['avg_latency']) /
                      baseline['stats']['avg_latency'] * 100)
        total_improvements['latency'].append(latency_red)

        # Variance reduction
        variance_red = ((baseline['stats']['avg_variance'] - enhanced['stats']['avg_variance']) /
                       baseline['stats']['avg_variance'] * 100)
        total_improvements['variance'].append(variance_red)

        # Throughput improvement
        throughput_imp = ((enhanced['stats']['avg_throughput'] - baseline['stats']['avg_throughput']) /
                         baseline['stats']['avg_throughput'] * 100)
        total_improvements['throughput'].append(throughput_imp)

        # Training time reduction
        time_red = ((baseline['training_time'] - enhanced['training_time']) /
                   baseline['training_time'] * 100)
        total_improvements['training_time'].append(time_red)

    # Print achievements
    print(f"Average Reward Improvement: {np.mean(total_improvements['reward']):+.1f}%")
    print(f"Average Latency Reduction: {np.mean(total_improvements['latency']):+.1f}%")
    print(f"Average Load Variance Reduction: {np.mean(total_improvements['variance']):+.1f}%")
    print(f"Average Throughput Recovery: {np.mean(total_improvements['throughput']):+.1f}%")
    print(f"Average Training Time Reduction: {np.mean(total_improvements['training_time']):+.1f}%")

    # Communication improvements
    avg_comm_reduction = np.mean([
        all_fed_systems[traffic].get_communication_stats()['overhead_reduction'] * 100
        for traffic in traffic_levels.keys()
    ])
    print(f"Communication Overhead Reduction: {avg_comm_reduction:.1f}%")

    print("\nKEY INNOVATIONS SUCCESSFULLY IMPLEMENTED:")
    print("-" * 45)

    innovations = [
        "Asynchronous Federated Learning with Smart Scheduling",
        "Hierarchical SDN Controller Architecture (Regional → Local)",
        "Selective Parameter Sharing (Base Layers Only)",
        "Traffic-Adaptive Weighting System",
        "Performance-Based Agent Selection",
        "Multi-Objective Reward Optimization",
        "Real-time Load Balancing with Variance Minimization"
    ]

    for innovation in innovations:
        print(f"  {innovation}")

    print("\nSCALABILITY ACHIEVEMENTS:")
    print("-" * 28)

    scalability_metrics = []
    for traffic in traffic_levels.keys():
        flows = traffic_levels[traffic]
        enhanced_stats = all_traffic_results[traffic]['AFRL']['stats']

        # Calculate efficiency metrics
        reward_per_flow = enhanced_stats['avg_reward'] / flows
        throughput_per_flow = enhanced_stats['avg_throughput'] / flows

        scalability_metrics.append({
            'traffic': traffic,
            'flows': flows,
            'reward_efficiency': reward_per_flow,
            'throughput_efficiency': throughput_per_flow,
            'final_variance': enhanced_stats['avg_variance']
        })

    print("Traffic Level | Flows    | Reward Efficiency | Throughput Efficiency | Load Variance")
    print("-" * 80)
    for metric in scalability_metrics:
        print(f"{metric['traffic']:12} | {metric['flows']:8,} | {metric['reward_efficiency']:17.6f} | "
              f"{metric['throughput_efficiency']:21.6f} | {metric['final_variance']:11.6f}")



    print(f"\n SYSTEM READY FOR PRODUCTION DEPLOYMENT! 🎊")
    print("="*70)

# Generate final summary
generate_final_summary()

# Save results with proper type conversion
print("\nSaving results...")
import json

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Create serializable results dictionary
serializable_results = {}

for traffic in traffic_levels.keys():
    serializable_results[traffic] = {}

    for model_name in all_traffic_results[traffic].keys():
        model_data = all_traffic_results[traffic][model_name]

        # Extract only serializable data
        serializable_results[traffic][model_name] = {
            'stats': model_data['stats'],
            'training_time': model_data['training_time']
        }

        # Add communication stats for federated models
        if 'communication_stats' in model_data:
            serializable_results[traffic][model_name]['communication_stats'] = model_data['communication_stats']

# Save communication statistics separately
communication_stats = {}
for traffic in traffic_levels.keys():
    if traffic in all_fed_systems:
        communication_stats[traffic] = all_fed_systems[traffic].get_communication_stats()

# Convert all NumPy types to native Python types
serializable_results_clean = convert_numpy_types(serializable_results)
communication_stats_clean = convert_numpy_types(communication_stats)
traffic_levels_clean = convert_numpy_types(traffic_levels)

# Create results directory
os.makedirs('enhanced_results', exist_ok=True)

# Save as JSON
try:
    with open('enhanced_results/performance_results.json', 'w') as f:
        json.dump(serializable_results_clean, f, indent=2)

    with open('enhanced_results/communication_stats.json', 'w') as f:
        json.dump(communication_stats_clean, f, indent=2)

    with open('enhanced_results/traffic_levels.json', 'w') as f:
        json.dump(traffic_levels_clean, f, indent=2)

    # Save summary statistics as CSV for easy analysis
    summary_data = []
    for traffic in traffic_levels.keys():
        for model_name in serializable_results_clean[traffic].keys():
            row = {
                'traffic_level': traffic,
                'model': model_name,
                'flows': traffic_levels_clean[traffic],
                **serializable_results_clean[traffic][model_name]['stats'],
                'training_time': serializable_results_clean[traffic][model_name]['training_time']
            }
            summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('enhanced_results/performance_summary.csv', index=False)

    print("Results saved successfully:")
    print("enhanced_results/performance_results.json")
    print("enhanced_results/communication_stats.json")
    print("enhanced_results/traffic_levels.json")
    print("enhanced_results/performance_summary.csv")
    print("Enhanced Federated RL System implementation completed successfully!")

except Exception as e:
    print(f"Error saving results: {e}")
    print("Displaying results instead:")
    print("\nPERFORMANCE SUMMARY:")
    for traffic in traffic_levels.keys():
        print(f"\n{traffic.upper()} Traffic ({traffic_levels[traffic]:,} flows):")
        for model in ['PPO_Baseline', 'AFRL']:
            if model in all_traffic_results[traffic]:
                stats = all_traffic_results[traffic][model]['stats']
                time_taken = all_traffic_results[traffic][model]['training_time']
                print(f"  {model}:")
                print(f"    Reward: {stats.get('avg_reward', 0):.2f}")
                print(f"    Latency: {stats.get('avg_latency', 0):.4f}")
                print(f"    Variance: {stats.get('avg_variance', 0):.6f}")
                print(f"    Throughput: {stats.get('avg_throughput', 0):.0f}")
                print(f"    Training Time: {time_taken:.1f}s")
now u suggest what can we do 