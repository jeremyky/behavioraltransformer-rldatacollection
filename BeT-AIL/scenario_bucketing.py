#!/usr/bin/env python3
"""
Scenario Bucketing for MetaDrive/Waymo - Create "Virtual Tracks"

This script analyzes MetaDrive/Waymo scenarios and clusters them into buckets
based on road geometry, traffic dynamics, and maneuver requirements.
This enables train-on-bucket-A / test-on-bucket-B evaluation similar to
BeT-AIL's track-based evaluation.

Usage:
    python scenario_bucketing.py --scenarios data/waymo_scenarios/ --output data/scenario_buckets.json
    python scenario_bucketing.py --load data/scenario_buckets.json --visualize
"""

import numpy as np
import pickle
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ScenarioSignature:
    """Complete signature of a scenario for bucketing."""
    scenario_id: str
    
    # Geometry features
    route_length_m: float
    curvature_mean: float
    curvature_p90: float
    curvature_p99: float
    curvature_max: float
    total_turn_deg: float
    max_single_turn_deg: float
    num_intersections: int
    num_lane_changes: int
    mean_lane_count: float
    median_lane_width: float
    
    # Traffic features
    density_mean: float
    density_p90: float
    num_pedestrians: int
    median_traffic_speed: float
    traffic_speed_iqr: float
    num_cutins: int
    num_forced_yields: int
    ttc_min: float
    ttc_p05: float
    ttc_violations: int  # TTC < 1.5s
    num_stops: int
    
    # Maneuver tokens
    token_string: str
    token_hash: int
    
    # Metadata
    success: bool
    episode_length: int
    total_reward: float
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to normalized feature vector for clustering."""
        geom_features = np.array([
            self.route_length_m / 1000.0,  # normalize by 1km
            self.curvature_mean * 100,
            self.curvature_p90 * 100,
            self.curvature_max * 100,
            self.total_turn_deg / 360.0,
            self.max_single_turn_deg / 90.0,
            self.num_intersections / 5.0,
            self.num_lane_changes / 5.0,
            self.mean_lane_count / 4.0,
        ])
        
        traffic_features = np.array([
            self.density_mean / 10.0,
            self.density_p90 / 15.0,
            self.num_pedestrians / 5.0,
            self.median_traffic_speed / 30.0,  # m/s
            self.traffic_speed_iqr / 10.0,
            self.num_cutins / 5.0,
            self.ttc_min / 10.0,
            self.ttc_p05 / 5.0,
            self.ttc_violations / 3.0,
            self.num_stops / 3.0,
        ])
        
        # Token counts as features (normalized by episode length)
        token_counts = self._token_bow()
        
        return np.concatenate([geom_features, traffic_features, token_counts])
    
    def _token_bow(self) -> np.ndarray:
        """Bag-of-tokens representation."""
        tokens = self.token_string.split()
        token_types = ['S', 'CL', 'CR', 'X', 'L', 'R', 'LC+', 'LC-', 'M', 'G', 'Y', 'R']
        counts = Counter(tok[:2] if len(tok) > 1 else tok for tok in tokens)
        bow = np.array([counts.get(t, 0) for t in token_types], dtype=float)
        return bow / max(len(tokens), 1)  # normalize


def compute_curvature_stats(centerline_xy: np.ndarray, ds: float = 1.0) -> Dict:
    """
    Compute curvature statistics along a centerline.
    
    Args:
        centerline_xy: (N, 2) array of xy coordinates
        ds: sampling distance in meters
        
    Returns:
        Dictionary of curvature features
    """
    if len(centerline_xy) < 3:
        return {
            'length_m': 0.0,
            'k_mean': 0.0,
            'k_p90': 0.0,
            'k_p99': 0.0,
            'k_max': 0.0,
            'turn_sum_deg': 0.0,
            'max_turn_deg': 0.0,
        }
    
    # Resample for consistent spacing
    xy = resample_polyline(centerline_xy, step=ds)
    
    # Compute headings
    dx = np.diff(xy[:, 0])
    dy = np.diff(xy[:, 1])
    headings = np.arctan2(dy, dx)
    headings = np.unwrap(headings)  # handle wrap-around
    
    # Curvature = change in heading / distance
    dheading = np.diff(headings)
    kappa = np.abs(dheading / ds)
    
    # Detect continuous turns
    turn_segments = []
    current_turn = 0
    for dh in dheading:
        if abs(dh) > 0.01:  # turning threshold
            current_turn += abs(dh)
        else:
            if current_turn > 0:
                turn_segments.append(current_turn)
            current_turn = 0
    if current_turn > 0:
        turn_segments.append(current_turn)
    
    return {
        'length_m': polyline_length(xy),
        'k_mean': float(np.mean(kappa)) if len(kappa) > 0 else 0.0,
        'k_p90': float(np.percentile(kappa, 90)) if len(kappa) > 0 else 0.0,
        'k_p99': float(np.percentile(kappa, 99)) if len(kappa) > 0 else 0.0,
        'k_max': float(np.max(kappa)) if len(kappa) > 0 else 0.0,
        'turn_sum_deg': float(np.degrees(np.sum(np.abs(dheading)))),
        'max_turn_deg': float(np.degrees(max(turn_segments))) if turn_segments else 0.0,
    }


def compute_traffic_features(ego_traj: np.ndarray, other_trajs: List[np.ndarray], 
                             times: np.ndarray, radius: float = 40.0) -> Dict:
    """
    Compute traffic interaction features.
    
    Args:
        ego_traj: (T, 2) ego vehicle trajectory (x, y)
        other_trajs: List of (T, 2) other vehicle trajectories
        times: (T,) timestamps
        radius: detection radius in meters
        
    Returns:
        Dictionary of traffic features
    """
    densities = []
    ttcs = []
    cutins = 0
    
    for t in range(len(times)):
        ego_pos = ego_traj[t]
        
        # Count vehicles within radius
        count = 0
        min_ttc = float('inf')
        
        for other in other_trajs:
            if t >= len(other):
                continue
            other_pos = other[t]
            dist = np.linalg.norm(ego_pos - other_pos)
            
            if dist < radius:
                count += 1
                
                # Compute TTC if approaching
                if t < len(times) - 1 and t < len(other) - 1:
                    ego_vel = ego_traj[t+1] - ego_traj[t]
                    other_vel = other[t+1] - other[t]
                    rel_vel = np.linalg.norm(ego_vel - other_vel)
                    if rel_vel > 0.1:  # moving
                        ttc = dist / rel_vel
                        min_ttc = min(min_ttc, ttc)
                
                # Detect cut-ins (lateral movement into ego's path)
                if dist < 20 and t > 0 and t < len(other) - 1:
                    lateral_movement = abs((other[t+1] - other[t-1])[1])
                    if lateral_movement > 2.0:  # lane-width threshold
                        cutins += 1
        
        densities.append(count)
        if min_ttc < float('inf'):
            ttcs.append(min_ttc)
    
    ttc_array = np.array(ttcs) if ttcs else np.array([10.0])
    
    return {
        'dens_mean': float(np.mean(densities)),
        'dens_p90': float(np.percentile(densities, 90)),
        'ttc_min': float(np.min(ttc_array)),
        'ttc_p05': float(np.percentile(ttc_array, 5)),
        'ttc_violations': int(np.sum(ttc_array < 1.5)),
        'cutins': cutins,
    }


def generate_token_string(centerline: np.ndarray, ego_traj: np.ndarray, 
                          lane_changes: List[str], intersections: List[int]) -> str:
    """
    Generate maneuver token string for a scenario.
    
    Tokens:
    - S: Straight
    - CL30, CL60, CL90: Curve left (30°, 60°, 90°)
    - CR30, CR60, CR90: Curve right
    - X: Intersection straight
    - L: Left turn at intersection
    - R: Right turn at intersection
    - LC+: Lane change right
    - LC-: Lane change left
    - M: Merge
    - G, Y, R: Signal colors
    """
    tokens = []
    
    # Compute turns along route
    if len(centerline) >= 3:
        headings = np.arctan2(np.diff(centerline[:, 1]), np.diff(centerline[:, 0]))
        headings = np.unwrap(headings)
        dheadings = np.diff(headings)
        
        for i, dh in enumerate(dheadings):
            angle_deg = abs(np.degrees(dh))
            
            # Check if at intersection
            at_intersection = any(abs(i - idx) < 5 for idx in intersections)
            
            if angle_deg < 10:
                tokens.append('X' if at_intersection else 'S')
            elif angle_deg < 45:
                tokens.append('CL30' if dh > 0 else 'CR30')
            elif angle_deg < 75:
                tokens.append('CL60' if dh > 0 else 'CR60')
            else:
                tokens.append(('L' if at_intersection else 'CL90') if dh > 0 else 
                             ('R' if at_intersection else 'CR90'))
    
    # Add lane changes
    tokens.extend(lane_changes)
    
    return ' '.join(tokens)


# Helper functions

def resample_polyline(xy: np.ndarray, step: float = 1.0) -> np.ndarray:
    """Resample polyline at uniform spacing."""
    if len(xy) < 2:
        return xy
    
    # Compute cumulative distances
    dists = np.sqrt(np.sum(np.diff(xy, axis=0)**2, axis=1))
    cum_dist = np.concatenate([[0], np.cumsum(dists)])
    
    # Resample at regular intervals
    total_length = cum_dist[-1]
    if total_length < step:
        return xy
    
    sample_dists = np.arange(0, total_length, step)
    resampled = np.array([np.interp(sample_dists, cum_dist, xy[:, i]) for i in range(2)]).T
    
    return resampled


def polyline_length(xy: np.ndarray) -> float:
    """Compute total length of polyline."""
    if len(xy) < 2:
        return 0.0
    return float(np.sum(np.sqrt(np.sum(np.diff(xy, axis=0)**2, axis=1))))


def extract_scenario_signature(trajectory: Dict, scenario_id: str) -> ScenarioSignature:
    """
    Extract complete scenario signature from a trajectory.
    
    Args:
        trajectory: Dictionary with trajectory data
        scenario_id: Unique identifier for this scenario
        
    Returns:
        ScenarioSignature object
    """
    # Extract basic info
    obs = trajectory['observations']
    acts = trajectory['actions']
    rews = trajectory['rewards']
    infos = trajectory.get('infos', [{}] * len(obs))
    
    episode_length = len(obs)
    total_reward = float(np.sum(rews))
    success = infos[-1].get('arrive_dest', False) if infos else False
    
    # Extract ego trajectory (assume first 2 dims are x, y position)
    ego_traj = obs[:, :2] if obs.shape[1] >= 2 else np.zeros((len(obs), 2))
    
    # Placeholder for road geometry (would need MetaDrive API access)
    # In practice, you'd extract this from MetaDrive's map
    centerline = ego_traj  # simplified: use ego path as centerline
    
    # Compute geometry features
    geom = compute_curvature_stats(centerline)
    
    # Compute traffic features (simplified without other vehicles data)
    # In practice, extract from MetaDrive's perception
    traffic = {
        'dens_mean': 2.0,  # placeholder
        'dens_p90': 4.0,
        'ttc_min': 3.0,
        'ttc_p05': 2.0,
        'ttc_violations': 0,
        'cutins': 0,
    }
    
    # Generate tokens
    lane_changes = ['LC+' if act[0] > 0.3 else 'LC-' if act[0] < -0.3 else '' 
                    for act in acts]
    lane_changes = [lc for lc in lane_changes if lc]
    
    token_str = generate_token_string(centerline, ego_traj, lane_changes, [])
    
    return ScenarioSignature(
        scenario_id=scenario_id,
        route_length_m=geom['length_m'],
        curvature_mean=geom['k_mean'],
        curvature_p90=geom['k_p90'],
        curvature_p99=geom['k_p99'],
        curvature_max=geom['k_max'],
        total_turn_deg=geom['turn_sum_deg'],
        max_single_turn_deg=geom['max_turn_deg'],
        num_intersections=0,  # would extract from map
        num_lane_changes=len(lane_changes),
        mean_lane_count=2.0,  # placeholder
        median_lane_width=3.5,  # placeholder
        density_mean=traffic['dens_mean'],
        density_p90=traffic['dens_p90'],
        num_pedestrians=0,  # placeholder
        median_traffic_speed=10.0,  # placeholder
        traffic_speed_iqr=5.0,  # placeholder
        num_cutins=traffic['cutins'],
        num_forced_yields=0,  # placeholder
        ttc_min=traffic['ttc_min'],
        ttc_p05=traffic['ttc_p05'],
        ttc_violations=traffic['ttc_violations'],
        num_stops=np.sum(np.linalg.norm(acts, axis=1) < 0.1) if len(acts) > 0 else 0,
        token_string=token_str,
        token_hash=hash(token_str),
        success=success,
        episode_length=episode_length,
        total_reward=total_reward,
    )


def cluster_scenarios(signatures: List[ScenarioSignature], n_clusters: int = 8) -> Dict:
    """
    Cluster scenarios into buckets based on their signatures.
    
    Args:
        signatures: List of ScenarioSignature objects
        n_clusters: Number of clusters (virtual tracks)
        
    Returns:
        Dictionary with cluster assignments and statistics
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    
    # Extract feature vectors
    feature_matrix = np.array([sig.to_feature_vector() for sig in signatures])
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_matrix)
    
    # Cluster
    print(f"\nClustering {len(signatures)} scenarios into {n_clusters} buckets...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Compute silhouette score
    sil_score = silhouette_score(features_scaled, cluster_labels)
    print(f"Silhouette score: {sil_score:.3f}")
    
    # Organize results
    buckets = defaultdict(list)
    for sig, label in zip(signatures, cluster_labels):
        buckets[int(label)].append(sig.scenario_id)
    
    # Compute bucket statistics
    bucket_stats = {}
    for bucket_id, scenario_ids in buckets.items():
        bucket_sigs = [sig for sig in signatures if sig.scenario_id in scenario_ids]
        
        bucket_stats[bucket_id] = {
            'size': len(scenario_ids),
            'scenario_ids': scenario_ids,
            'avg_length_m': np.mean([s.route_length_m for s in bucket_sigs]),
            'avg_curvature': np.mean([s.curvature_mean for s in bucket_sigs]),
            'avg_density': np.mean([s.density_mean for s in bucket_sigs]),
            'success_rate': np.mean([s.success for s in bucket_sigs]),
            'common_tokens': Counter(' '.join([s.token_string for s in bucket_sigs]).split()).most_common(5),
        }
    
    return {
        'n_clusters': n_clusters,
        'silhouette_score': float(sil_score),
        'buckets': bucket_stats,
        'cluster_centers': kmeans.cluster_centers_.tolist(),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
    }


def save_bucketing(signatures: List[ScenarioSignature], clustering: Dict, output_path: str):
    """Save scenario signatures and bucketing results."""
    output = {
        'signatures': [asdict(sig) for sig in signatures],
        'clustering': clustering,
        'metadata': {
            'n_scenarios': len(signatures),
            'n_buckets': clustering['n_clusters'],
        }
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Saved bucketing to {output_path}")


def print_bucket_summary(clustering: Dict):
    """Print human-readable summary of buckets."""
    print("\n" + "="*80)
    print("BUCKET SUMMARY - Virtual Tracks")
    print("="*80)
    
    for bucket_id in sorted(clustering['buckets'].keys()):
        stats = clustering['buckets'][bucket_id]
        print(f"\n📁 Bucket {bucket_id}: {stats['size']} scenarios")
        print(f"   Avg route length: {stats['avg_length_m']:.1f}m")
        print(f"   Avg curvature:    {stats['avg_curvature']:.4f}")
        print(f"   Avg density:      {stats['avg_density']:.1f} vehicles")
        print(f"   Success rate:     {100*stats['success_rate']:.1f}%")
        print(f"   Common maneuvers: {', '.join([f'{tok}({cnt})' for tok, cnt in stats['common_tokens'][:3]])}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Bucket MetaDrive/Waymo scenarios into virtual tracks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input pickle file with trajectories')
    parser.add_argument('--output', type=str, default='data/scenario_buckets.json',
                        help='Output JSON file for bucketing results')
    parser.add_argument('--n_clusters', type=int, default=8,
                        help='Number of clusters (virtual tracks)')
    parser.add_argument('--load', type=str, default=None,
                        help='Load existing bucketing from JSON')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize bucket distributions')
    
    args = parser.parse_args()
    
    if args.load:
        # Load existing bucketing
        with open(args.load, 'r') as f:
            data = json.load(f)
        print_bucket_summary(data['clustering'])
        
        if args.visualize:
            print("\n📊 Visualization would go here (matplotlib/plotly)")
            print("   - t-SNE plot of scenarios colored by bucket")
            print("   - Distribution plots for each bucket")
            print("   - Token frequency heatmaps")
    
    else:
        # Load trajectories
        print(f"Loading trajectories from {args.input}...")
        with open(args.input, 'rb') as f:
            trajectories = pickle.load(f)
        
        print(f"✓ Loaded {len(trajectories)} trajectories")
        
        # Extract signatures
        print("\nExtracting scenario signatures...")
        signatures = []
        for i, traj in enumerate(tqdm(trajectories, desc="Processing")):
            sig = extract_scenario_signature(traj, scenario_id=f"scenario_{i:04d}")
            signatures.append(sig)
        
        # Cluster
        clustering = cluster_scenarios(signatures, n_clusters=args.n_clusters)
        
        # Print summary
        print_bucket_summary(clustering)
        
        # Save
        save_bucketing(signatures, clustering, args.output)
        
        print("\n✓ Bucketing complete!")
        print(f"\nNext steps:")
        print(f"  1. Review buckets: python scenario_bucketing.py --load {args.output}")
        print(f"  2. Select train/test buckets (e.g., bucket 0 for train, bucket 3 for test)")
        print(f"  3. Filter trajectories by bucket ID for training")

