#!/usr/bin/env python3
"""
Record expert demonstrations from MetaDrive using the built-in IDM policy.

Usage:
    python record_metadrive_expert.py --num_episodes 50 --output expert_demos.pkl
    python record_metadrive_expert.py --num_episodes 10 --visualize
"""

import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path


def record_expert_demos(num_episodes=50, env_config=None, policy_type='idm'):
    """
    Record expert demonstrations using MetaDrive's expert policy.
    
    Args:
        num_episodes: Number of episodes to record
        env_config: MetaDrive environment configuration dict
        policy_type: Type of expert policy ('idm' or 'expert_obs')
    
    Returns:
        List of trajectory dictionaries
    """
    from metadrive import MetaDriveEnv
    
    if policy_type == 'idm':
        from metadrive.policy.idm_policy import IDMPolicy as ExpertPolicy
    else:
        from metadrive.policy.expert_policy import ExpertPolicy
    
    if env_config is None:
        env_config = dict(
            use_render=False,
            traffic_density=0.1,
            start_seed=0,
            num_scenarios=1000,
            # Make episodes more consistent
            decision_repeat=5,
            physics_world_step_size=0.02,
        )
    
    print("Initializing MetaDrive environment...")
    env = MetaDriveEnv(env_config)
    expert_policy = ExpertPolicy(env, 0)
    
    trajectories = []
    success_count = 0
    
    print(f"\nRecording {num_episodes} expert demonstrations using {policy_type.upper()} policy...")
    print("="*60)
    
    for episode in tqdm(range(num_episodes), desc="Recording episodes"):
        obs, info = env.reset()
        expert_policy.reset()
        
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'infos': []
        }
        
        done = False
        truncated = False
        step_count = 0
        episode_reward = 0
        
        while not (done or truncated):
            # Get expert action
            action = expert_policy.act(obs)
            
            # Store current step
            trajectory['observations'].append(obs.copy())
            trajectory['actions'].append(action.copy() if isinstance(action, np.ndarray) else np.array(action))
            
            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            
            trajectory['rewards'].append(reward)
            trajectory['dones'].append(done or truncated)
            trajectory['infos'].append(info.copy())
            
            episode_reward += reward
            step_count += 1
            
            # Safety limit to prevent infinite episodes
            if step_count > 2000:
                print(f"\nWarning: Episode {episode+1} exceeded 2000 steps, truncating...")
                truncated = True
                break
        
        # Convert lists to numpy arrays for efficient storage
        trajectory['observations'] = np.array(trajectory['observations'], dtype=np.float32)
        trajectory['actions'] = np.array(trajectory['actions'], dtype=np.float32)
        trajectory['rewards'] = np.array(trajectory['rewards'], dtype=np.float32)
        trajectory['dones'] = np.array(trajectory['dones'], dtype=bool)
        
        trajectories.append(trajectory)
        
        # Track success
        if info.get('arrive_dest', False):
            success_count += 1
        
        # Periodic progress report
        if (episode + 1) % 10 == 0:
            print(f"\nEpisode {episode+1}/{num_episodes}:")
            print(f"  Steps: {step_count}, Reward: {episode_reward:.2f}, Success: {info.get('arrive_dest', False)}")
            print(f"  Success rate so far: {success_count}/{episode+1} ({100*success_count/(episode+1):.1f}%)")
    
    env.close()
    
    print("\n" + "="*60)
    print("Recording complete!")
    print(f"Success rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    
    return trajectories


def save_trajectories(trajectories, filename='metadrive_expert_demos.pkl'):
    """Save trajectories to disk."""
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(trajectories, f)
    
    print(f"\n✓ Saved {len(trajectories)} trajectories to {filename}")
    
    # Print file size
    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")


def load_trajectories(filename='metadrive_expert_demos.pkl'):
    """Load trajectories from disk."""
    with open(filename, 'rb') as f:
        trajectories = pickle.load(f)
    print(f"Loaded {len(trajectories)} trajectories from {filename}")
    return trajectories


def print_statistics(trajectories):
    """Print statistics about the recorded trajectories."""
    total_steps = sum(len(traj['observations']) for traj in trajectories)
    avg_steps = total_steps / len(trajectories)
    avg_reward = np.mean([sum(traj['rewards']) for traj in trajectories])
    std_reward = np.std([sum(traj['rewards']) for traj in trajectories])
    
    # Check success rate
    success_count = sum(
        1 for traj in trajectories 
        if traj['infos'][-1].get('arrive_dest', False)
    )
    
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total episodes:           {len(trajectories)}")
    print(f"Total steps:              {total_steps}")
    print(f"Average steps/episode:    {avg_steps:.1f}")
    print(f"Average reward:           {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Success rate:             {success_count}/{len(trajectories)} ({100*success_count/len(trajectories):.1f}%)")
    print(f"Observation shape:        {trajectories[0]['observations'][0].shape}")
    print(f"Action shape:             {trajectories[0]['actions'][0].shape}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Record MetaDrive expert demonstrations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--num_episodes', type=int, default=50, 
                        help='Number of episodes to record')
    parser.add_argument('--output', type=str, default='data/metadrive_expert_demos.pkl', 
                        help='Output filename')
    parser.add_argument('--visualize', action='store_true', 
                        help='Show visualization while recording')
    parser.add_argument('--policy', type=str, default='idm', choices=['idm', 'expert_obs'],
                        help='Expert policy type to use')
    parser.add_argument('--traffic_density', type=float, default=0.1,
                        help='Traffic density (0.0 to 1.0)')
    
    args = parser.parse_args()
    
    env_config = dict(
        use_render=args.visualize,
        traffic_density=args.traffic_density,
        start_seed=0,
        num_scenarios=1000,
        decision_repeat=5,
        physics_world_step_size=0.02,
    )
    
    # Record demonstrations
    trajectories = record_expert_demos(
        num_episodes=args.num_episodes, 
        env_config=env_config,
        policy_type=args.policy
    )
    
    # Print statistics
    print_statistics(trajectories)
    
    # Save to disk
    save_trajectories(trajectories, args.output)
    
    print("\n✓ Done! Next step:")
    print(f"  python convert_metadrive_demos.py --input {args.output}")

