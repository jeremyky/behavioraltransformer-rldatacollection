#!/usr/bin/env python3
"""
Convert MetaDrive expert demonstrations to BeT-AIL compatible format.

Usage:
    python convert_metadrive_demos.py --input data/metadrive_expert_demos.pkl --output data/metadrive_betail_format.pkl
"""

import pickle
import numpy as np
from pathlib import Path


def convert_to_betail_format(trajectories, verbose=True):
    """
    Convert MetaDrive trajectories to BeT-AIL/imitation library format.
    
    BeT-AIL expects trajectories with:
    - observations: (T, obs_dim)
    - actions: (T, act_dim)  
    - rewards: (T, 1) - reshaped to column vector
    - dones/terminals: (T,) - boolean array
    
    Args:
        trajectories: List of MetaDrive trajectory dicts
        verbose: Print conversion info
        
    Returns:
        List of converted trajectory dicts
    """
    
    converted = []
    
    if verbose:
        print("\nConverting trajectories to BeT-AIL format...")
        print("="*60)
    
    for i, traj in enumerate(trajectories):
        # Ensure observations and actions are 2D
        obs = traj['observations']
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)
        
        acts = traj['actions']
        if acts.ndim == 1:
            acts = acts.reshape(-1, 1)
        
        # Rewards must be (T, 1) shape
        rewards = traj['rewards']
        if rewards.ndim == 1:
            rewards = rewards.reshape(-1, 1)
        
        # Dones can stay 1D boolean
        dones = traj['dones'].astype(bool)
        
        # Create converted trajectory
        converted_traj = {
            'observations': obs.astype(np.float32),
            'actions': acts.astype(np.float32),
            'rewards': rewards.astype(np.float32),
            'terminals': dones,  # BeT-AIL uses 'terminals' sometimes
            'dones': dones,      # Keep both for compatibility
        }
        
        converted.append(converted_traj)
        
        # Validation
        assert len(obs) == len(acts) == len(rewards) == len(dones), \
            f"Trajectory {i}: Length mismatch!"
        
        if verbose and i == 0:
            print(f"Example trajectory (first one):")
            print(f"  Observations shape: {obs.shape}")
            print(f"  Actions shape:      {acts.shape}")
            print(f"  Rewards shape:      {rewards.shape}")
            print(f"  Dones shape:        {dones.shape}")
            print(f"  Episode length:     {len(obs)} steps")
            print(f"  Total reward:       {rewards.sum():.2f}")
    
    if verbose:
        total_steps = sum(len(t['observations']) for t in converted)
        print(f"\n✓ Converted {len(converted)} trajectories ({total_steps} total steps)")
        print("="*60)
    
    return converted


def validate_conversion(trajectories):
    """
    Validate that converted trajectories are in correct format.
    
    Returns:
        True if valid, raises AssertionError otherwise
    """
    print("\nValidating converted trajectories...")
    
    for i, traj in enumerate(trajectories):
        # Check required keys
        required_keys = ['observations', 'actions', 'rewards', 'dones']
        for key in required_keys:
            assert key in traj, f"Trajectory {i} missing key: {key}"
        
        # Check shapes
        T = len(traj['observations'])
        assert traj['actions'].shape[0] == T, f"Trajectory {i}: Action length mismatch"
        assert traj['rewards'].shape[0] == T, f"Trajectory {i}: Reward length mismatch"
        assert traj['dones'].shape[0] == T, f"Trajectory {i}: Done length mismatch"
        
        # Check dimensions
        assert traj['observations'].ndim == 2, f"Trajectory {i}: Observations must be 2D"
        assert traj['actions'].ndim == 2, f"Trajectory {i}: Actions must be 2D"
        assert traj['rewards'].ndim == 2, f"Trajectory {i}: Rewards must be 2D"
        assert traj['rewards'].shape[1] == 1, f"Trajectory {i}: Rewards must be (T, 1)"
        
        # Check types
        assert traj['observations'].dtype == np.float32, f"Trajectory {i}: Observations must be float32"
        assert traj['actions'].dtype == np.float32, f"Trajectory {i}: Actions must be float32"
        assert traj['rewards'].dtype == np.float32, f"Trajectory {i}: Rewards must be float32"
        assert traj['dones'].dtype == bool, f"Trajectory {i}: Dones must be boolean"
    
    print("✓ All trajectories valid!")
    return True


def save_converted(trajectories, filename):
    """Save converted trajectories to disk."""
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(trajectories, f)
    
    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"\n✓ Saved converted trajectories to {filename}")
    print(f"  File size: {file_size_mb:.2f} MB")


def print_dataset_info(trajectories):
    """Print information about the converted dataset."""
    total_steps = sum(len(t['observations']) for t in trajectories)
    avg_steps = total_steps / len(trajectories)
    total_reward = sum(t['rewards'].sum() for t in trajectories)
    avg_reward = total_reward / len(trajectories)
    
    obs_dim = trajectories[0]['observations'].shape[1]
    act_dim = trajectories[0]['actions'].shape[1]
    
    print(f"\n{'='*60}")
    print("CONVERTED DATASET INFO")
    print(f"{'='*60}")
    print(f"Number of trajectories:   {len(trajectories)}")
    print(f"Total steps:              {total_steps}")
    print(f"Average episode length:   {avg_steps:.1f}")
    print(f"Average episode reward:   {avg_reward:.2f}")
    print(f"Observation dimension:    {obs_dim}")
    print(f"Action dimension:         {act_dim}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert MetaDrive demonstrations to BeT-AIL format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input pickle file with MetaDrive trajectories')
    parser.add_argument('--output', type=str, default=None,
                        help='Output pickle file (default: input_betail.pkl)')
    parser.add_argument('--validate', action='store_true', default=True,
                        help='Validate converted trajectories')
    
    args = parser.parse_args()
    
    # Set default output filename if not provided
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_betail.pkl")
    
    print(f"\nInput:  {args.input}")
    print(f"Output: {args.output}")
    
    # Load original trajectories
    print("\nLoading trajectories...")
    with open(args.input, 'rb') as f:
        trajectories = pickle.load(f)
    print(f"✓ Loaded {len(trajectories)} trajectories")
    
    # Convert to BeT-AIL format
    converted = convert_to_betail_format(trajectories, verbose=True)
    
    # Validate if requested
    if args.validate:
        validate_conversion(converted)
    
    # Print dataset info
    print_dataset_info(converted)
    
    # Save converted trajectories
    save_converted(converted, args.output)
    
    print("\n✓ Conversion complete!")
    print(f"\nNext steps:")
    print(f"  1. Adapt main.py to use MetaDrive environment")
    print(f"  2. Load these trajectories: pickle.load(open('{args.output}', 'rb'))")
    print(f"  3. Train BeT on MetaDrive data\n")

