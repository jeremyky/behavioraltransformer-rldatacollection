#!/usr/bin/env python3
"""
Filter trajectories by scenario bucket for train/test splits.

Usage:
    # Create train set from bucket 0
    python filter_by_bucket.py --trajectories data/metadrive_expert_demos.pkl \
                                --buckets data/scenario_buckets.json \
                                --bucket_ids 0,1,2 \
                                --output data/train_bucket_012.pkl
    
    # Create test set from bucket 5
    python filter_by_bucket.py --trajectories data/metadrive_expert_demos.pkl \
                                --buckets data/scenario_buckets.json \
                                --bucket_ids 5 \
                                --output data/test_bucket_5.pkl
"""

import pickle
import json
import numpy as np
from pathlib import Path


def filter_trajectories_by_bucket(trajectories, bucketing, bucket_ids):
    """
    Filter trajectories to only include those in specified buckets.
    
    Args:
        trajectories: List of trajectory dicts
        bucketing: Bucketing results from scenario_bucketing.py
        bucket_ids: List of bucket IDs to include
        
    Returns:
        Filtered list of trajectories
    """
    # Get scenario IDs for selected buckets
    selected_ids = set()
    for bucket_id in bucket_ids:
        bucket_id = int(bucket_id)
        if bucket_id in bucketing['buckets']:
            selected_ids.update(bucketing['buckets'][bucket_id]['scenario_ids'])
    
    print(f"Selected {len(selected_ids)} scenarios from buckets {bucket_ids}")
    
    # Filter trajectories
    # Scenario IDs are in format "scenario_XXXX"
    filtered = []
    for i, traj in enumerate(trajectories):
        scenario_id = f"scenario_{i:04d}"
        if scenario_id in selected_ids:
            filtered.append(traj)
    
    return filtered


def print_split_stats(train_trajs, test_trajs):
    """Print statistics about the train/test split."""
    def stats(trajs, name):
        total_steps = sum(len(t['observations']) for t in trajs)
        avg_steps = total_steps / len(trajs) if trajs else 0
        avg_reward = np.mean([sum(t['rewards']) for t in trajs]) if trajs else 0
        
        print(f"\n{name}:")
        print(f"  Episodes: {len(trajs)}")
        print(f"  Total steps: {total_steps}")
        print(f"  Avg steps/episode: {avg_steps:.1f}")
        print(f"  Avg reward: {avg_reward:.2f}")
    
    print("\n" + "="*60)
    print("TRAIN/TEST SPLIT STATISTICS")
    print("="*60)
    stats(train_trajs, "TRAIN SET")
    stats(test_trajs, "TEST SET")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Filter trajectories by scenario bucket',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--trajectories', type=str, required=True,
                        help='Input trajectory pickle file')
    parser.add_argument('--buckets', type=str, required=True,
                        help='Bucketing JSON file from scenario_bucketing.py')
    parser.add_argument('--bucket_ids', type=str, required=True,
                        help='Comma-separated bucket IDs (e.g., "0,1,2")')
    parser.add_argument('--output', type=str, required=True,
                        help='Output pickle file for filtered trajectories')
    
    args = parser.parse_args()
    
    # Parse bucket IDs
    bucket_ids = [int(x.strip()) for x in args.bucket_ids.split(',')]
    
    # Load data
    print(f"Loading trajectories from {args.trajectories}...")
    with open(args.trajectories, 'rb') as f:
        trajectories = pickle.load(f)
    print(f"✓ Loaded {len(trajectories)} trajectories")
    
    print(f"\nLoading bucketing from {args.buckets}...")
    with open(args.buckets, 'r') as f:
        bucketing = json.load(f)['clustering']
    print(f"✓ Loaded bucketing with {len(bucketing['buckets'])} buckets")
    
    # Filter
    print(f"\nFiltering to buckets: {bucket_ids}")
    filtered = filter_trajectories_by_bucket(trajectories, bucketing, bucket_ids)
    
    # Print stats
    total_steps = sum(len(t['observations']) for t in filtered)
    avg_reward = np.mean([sum(t['rewards']) for t in filtered])
    
    print(f"\n✓ Filtered to {len(filtered)} trajectories")
    print(f"  Total steps: {total_steps}")
    print(f"  Avg reward: {avg_reward:.2f}")
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(filtered, f)
    
    file_size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"\n✓ Saved filtered trajectories to {args.output}")
    print(f"  File size: {file_size_mb:.2f} MB")
    
    print("\nNext step:")
    print(f"  Use this filtered dataset for training:")
    print(f"  python main_metadrive.py --expert_data {args.output}")

