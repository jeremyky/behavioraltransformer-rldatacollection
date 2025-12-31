#!/usr/bin/env python3
"""
Validate scenario buckets by visualizing scenarios from each bucket.

This script lets you run scenarios from the same bucket in MetaDrive
to visually confirm they are similar.

Usage:
    # Validate bucket 0 (run 3 random scenarios from bucket 0)
    python validate_buckets.py --buckets data/waymo_buckets.json \
                                 --scenario_dir ../drive-rig/datasets/waymo_converted_test/waymo_converted_test_0/ \
                                 --bucket_id 0 \
                                 --num_scenarios 3
    
    # Compare two buckets side-by-side
    python validate_buckets.py --buckets data/waymo_buckets.json \
                                 --scenario_dir ../drive-rig/datasets/waymo_converted_test/waymo_converted_test_0/ \
                                 --bucket_id 0,5 \
                                 --num_scenarios 2
"""

import json
import pickle
import random
from pathlib import Path
import numpy as np


def load_bucketing(json_path: str):
    """Load bucketing results."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_scenarios_from_bucket(bucketing: dict, bucket_id: int, num_scenarios: int = 3):
    """Get random scenarios from a bucket."""
    bucket_data = bucketing['clustering']['buckets'][str(bucket_id)]
    scenarios = bucket_data['scenarios']
    
    if num_scenarios > len(scenarios):
        print(f"Warning: Bucket {bucket_id} only has {len(scenarios)} scenarios, using all")
        return scenarios
    
    return random.sample(scenarios, num_scenarios)


def load_and_run_scenario(scenario_path: str, render: bool = True):
    """
    Load a Waymo scenario and run it in MetaDrive.
    
    Args:
        scenario_path: Path to .pkl file
        render: Whether to render visually
    """
    try:
        from metadrive import MetaDriveEnv
        from metadrive.policy.idm_policy import IDMPolicy
        
        # Load scenario
        with open(scenario_path, 'rb') as f:
            scenario = pickle.load(f)
        
        print(f"\n{'='*80}")
        print(f"Running: {Path(scenario_path).name}")
        print(f"{'='*80}")
        
        # Create MetaDrive env with this scenario
        config = {
            'use_render': render,
            'manual_control': False,
            'traffic_density': 0.0,  # Will use scenario traffic
            'start_seed': 0,
            'image_observation': False,
        }
        
        # If scenario is a ScenarioNet-compatible format
        if 'map_features' in scenario or 'tracks' in scenario:
            # Use MetaDrive's scenario replay
            config['data_directory'] = str(Path(scenario_path).parent)
            config['case_num'] = 1
            env = MetaDriveEnv(config)
        else:
            # Fall back to standard env
            env = MetaDriveEnv(config)
        
        # Run with IDM policy
        policy = IDMPolicy(env, 0)
        obs, info = env.reset()
        policy.reset()
        
        done = False
        truncated = False
        step = 0
        total_reward = 0
        
        print(f"Running scenario...")
        while not (done or truncated) and step < 1000:
            action = policy.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            if render and step % 10 == 0:
                print(f"  Step {step}, Reward: {total_reward:.2f}", end='\r')
        
        success = info.get('arrive_dest', False)
        print(f"\n{'='*80}")
        print(f"Finished: {step} steps, Reward: {total_reward:.2f}, Success: {success}")
        print(f"{'='*80}\n")
        
        env.close()
        
        return {
            'steps': step,
            'reward': total_reward,
            'success': success,
        }
        
    except ImportError:
        print("❌ MetaDrive not installed!")
        print("   Install: pip install metadrive-simulator")
        return None
    except Exception as e:
        print(f"❌ Error running scenario: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_bucket_info(bucketing: dict, bucket_id: int):
    """Print information about a bucket."""
    bucket_data = bucketing['clustering']['buckets'][str(bucket_id)]
    
    print(f"\n{'='*80}")
    print(f"BUCKET {bucket_id} INFORMATION")
    print(f"{'='*80}")
    print(f"Number of scenarios: {bucket_data['size']}")
    print(f"Average route length: {bucket_data['avg_length_m']:.1f}m")
    print(f"Average curvature: {bucket_data['avg_curvature']:.4f}")
    print(f"Average vehicles: {bucket_data['avg_num_vehicles']:.1f}")
    print(f"Average density: {bucket_data['avg_density']:.2f} vehicles/100m")
    print(f"Common maneuvers: {', '.join([f'{tok}({cnt})' for tok, cnt in bucket_data['common_tokens']])}")
    print(f"{'='*80}\n")


def compare_buckets_stats(bucketing: dict, bucket_ids: list):
    """Print comparison of multiple buckets."""
    print(f"\n{'='*80}")
    print(f"BUCKET COMPARISON")
    print(f"{'='*80}\n")
    
    stats = []
    for bid in bucket_ids:
        bucket_data = bucketing['clustering']['buckets'][str(bid)]
        stats.append({
            'id': bid,
            'size': bucket_data['size'],
            'length': bucket_data['avg_length_m'],
            'curvature': bucket_data['avg_curvature'],
            'vehicles': bucket_data['avg_num_vehicles'],
            'density': bucket_data['avg_density'],
        })
    
    # Print table
    print(f"{'Bucket':<8} {'Size':<8} {'Length(m)':<12} {'Curvature':<12} {'Vehicles':<10} {'Density':<10}")
    print("-" * 80)
    for s in stats:
        print(f"{s['id']:<8} {s['size']:<8} {s['length']:<12.1f} {s['curvature']:<12.4f} {s['vehicles']:<10.1f} {s['density']:<10.2f}")
    
    print("\n" + "="*80 + "\n")


def interactive_validation(bucketing: dict, scenario_dir: str):
    """Interactive mode to explore buckets."""
    print("\n" + "="*80)
    print("INTERACTIVE BUCKET VALIDATION")
    print("="*80)
    print("\nCommands:")
    print("  info <bucket_id>        - Show bucket information")
    print("  compare <id1>,<id2>     - Compare two buckets")
    print("  run <bucket_id> <n>     - Run n scenarios from bucket")
    print("  list                     - List all buckets")
    print("  quit                     - Exit")
    print("="*80)
    
    while True:
        try:
            cmd = input("\n> ").strip().split()
            if not cmd:
                continue
            
            action = cmd[0].lower()
            
            if action == 'quit':
                break
            
            elif action == 'list':
                for bucket_id in sorted(bucketing['clustering']['buckets'].keys()):
                    print_bucket_info(bucketing, int(bucket_id))
            
            elif action == 'info' and len(cmd) > 1:
                bucket_id = int(cmd[1])
                print_bucket_info(bucketing, bucket_id)
            
            elif action == 'compare' and len(cmd) > 1:
                bucket_ids = [int(x.strip()) for x in cmd[1].split(',')]
                compare_buckets_stats(bucketing, bucket_ids)
            
            elif action == 'run' and len(cmd) > 2:
                bucket_id = int(cmd[1])
                num_scenarios = int(cmd[2])
                
                print_bucket_info(bucketing, bucket_id)
                scenarios = get_scenarios_from_bucket(bucketing, bucket_id, num_scenarios)
                
                for i, scenario_info in enumerate(scenarios, 1):
                    print(f"\n--- Scenario {i}/{len(scenarios)} ---")
                    scenario_path = scenario_info['file_path']
                    if not Path(scenario_path).exists():
                        # Try relative to scenario_dir
                        scenario_path = str(Path(scenario_dir) / Path(scenario_path).name)
                    
                    result = load_and_run_scenario(scenario_path, render=True)
                    
                    if i < len(scenarios):
                        input("Press Enter to run next scenario...")
            
            else:
                print("Unknown command. Type 'quit' to exit.")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate scenario buckets by running scenarios',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--buckets', type=str, required=True,
                        help='Bucketing JSON file')
    parser.add_argument('--scenario_dir', type=str, required=True,
                        help='Directory with scenario .pkl files')
    parser.add_argument('--bucket_id', type=str, default=None,
                        help='Bucket ID(s) to validate (comma-separated for comparison)')
    parser.add_argument('--num_scenarios', type=int, default=3,
                        help='Number of scenarios to run per bucket')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode')
    
    args = parser.parse_args()
    
    # Load bucketing
    print(f"Loading bucketing from {args.buckets}...")
    bucketing = load_bucketing(args.buckets)
    print(f"✓ Loaded {bucketing['metadata']['n_buckets']} buckets with {bucketing['metadata']['n_scenarios']} scenarios")
    
    if args.interactive:
        interactive_validation(bucketing, args.scenario_dir)
    
    elif args.bucket_id:
        bucket_ids = [int(x.strip()) for x in args.bucket_id.split(',')]
        
        if len(bucket_ids) > 1:
            # Compare mode
            compare_buckets_stats(bucketing, bucket_ids)
            
            for bucket_id in bucket_ids:
                print(f"\n{'#'*80}")
                print(f"# Running scenarios from Bucket {bucket_id}")
                print(f"{'#'*80}")
                
                print_bucket_info(bucketing, bucket_id)
                scenarios = get_scenarios_from_bucket(bucketing, bucket_id, args.num_scenarios)
                
                for i, scenario_info in enumerate(scenarios, 1):
                    print(f"\n--- Bucket {bucket_id}, Scenario {i}/{len(scenarios)} ---")
                    scenario_path = scenario_info['file_path']
                    if not Path(scenario_path).exists():
                        scenario_path = str(Path(args.scenario_dir) / Path(scenario_path).name)
                    
                    result = load_and_run_scenario(scenario_path, render=True)
                    
                    if i < len(scenarios) or bucket_id != bucket_ids[-1]:
                        input("\nPress Enter to continue...")
        
        else:
            # Single bucket mode
            bucket_id = bucket_ids[0]
            print_bucket_info(bucketing, bucket_id)
            scenarios = get_scenarios_from_bucket(bucketing, bucket_id, args.num_scenarios)
            
            print(f"\nRunning {len(scenarios)} scenarios from bucket {bucket_id}...")
            print("Watch for similarities in:")
            print("  - Road geometry (straight, curved, intersections)")
            print("  - Traffic density")
            print("  - Maneuver types")
            print()
            
            for i, scenario_info in enumerate(scenarios, 1):
                print(f"\n--- Scenario {i}/{len(scenarios)} ---")
                scenario_path = scenario_info['file_path']
                if not Path(scenario_path).exists():
                    scenario_path = str(Path(args.scenario_dir) / Path(scenario_path).name)
                
                result = load_and_run_scenario(scenario_path, render=True)
                
                if i < len(scenarios):
                    input("\nPress Enter to run next scenario...")
            
            print("\n✓ Validation complete!")
            print("\nDid the scenarios look similar? If not:")
            print("  - Try different n_clusters")
            print("  - Check if you have enough scenarios")
            print("  - Adjust feature weights in bucketing code")
    
    else:
        print("\nNo bucket specified. Available buckets:")
        for bucket_id in sorted(bucketing['clustering']['buckets'].keys()):
            stats = bucketing['clustering']['buckets'][bucket_id]
            print(f"  Bucket {bucket_id}: {stats['size']} scenarios")
        print("\nUsage:")
        print(f"  python validate_buckets.py --buckets {args.buckets} --scenario_dir {args.scenario_dir} --bucket_id 0")
        print(f"  python validate_buckets.py --buckets {args.buckets} --scenario_dir {args.scenario_dir} --interactive")

