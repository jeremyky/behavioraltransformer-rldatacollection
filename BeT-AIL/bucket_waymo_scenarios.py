#!/usr/bin/env python3
"""
Bucket Waymo scenarios directly from converted scenario files.

This script loads Waymo scenario .pkl files, extracts their signatures,
and clusters them into buckets WITHOUT needing to run them first.

Usage:
    python bucket_waymo_scenarios.py --scenario_dir ../drive-rig/datasets/waymo_converted_test/waymo_converted_test_0/ \
                                      --output data/waymo_buckets.json \
                                      --n_clusters 8
"""

import numpy as np
import pickle
import json
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from tqdm import tqdm
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


@dataclass
class WaymoScenarioSignature:
    """Signature extracted from Waymo scenario file."""
    scenario_id: str
    file_path: str
    
    # Geometry features
    route_length_m: float
    curvature_mean: float
    curvature_p90: float
    curvature_max: float
    total_turn_deg: float
    num_lanes_mean: float
    num_lane_changes_estimated: int
    
    # Traffic features  
    num_vehicles: int
    num_pedestrians: int
    num_cyclists: int
    traffic_density_estimate: float
    
    # Maneuver tokens
    token_string: str
    token_hash: int
    
    # Metadata
    duration_seconds: float
    map_features: str
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to normalized feature vector for clustering."""
        geom = np.array([
            self.route_length_m / 500.0,
            self.curvature_mean * 100,
            self.curvature_p90 * 100,
            self.curvature_max * 100,
            self.total_turn_deg / 180.0,
            self.num_lanes_mean / 4.0,
            self.num_lane_changes_estimated / 5.0,
        ])
        
        traffic = np.array([
            self.num_vehicles / 20.0,
            self.num_pedestrians / 10.0,
            self.num_cyclists / 5.0,
            self.traffic_density_estimate / 0.5,
        ])
        
        # Token counts
        token_counts = self._token_bow()
        
        return np.concatenate([geom, traffic, token_counts])
    
    def _token_bow(self) -> np.ndarray:
        """Bag-of-tokens representation."""
        tokens = self.token_string.split()
        token_types = ['S', 'CL', 'CR', 'X', 'L', 'R', 'LC', 'M', 'U']
        counts = Counter(tok[:2] if len(tok) > 1 else tok[0] if tok else '' for tok in tokens)
        bow = np.array([counts.get(t, 0) for t in token_types], dtype=float)
        return bow / max(len(tokens), 1)


def load_waymo_scenario(pkl_path: str) -> Dict:
    """Load a single Waymo scenario .pkl file."""
    try:
        with open(pkl_path, 'rb') as f:
            scenario = pickle.load(f)
        return scenario
    except Exception as e:
        print(f"Warning: Failed to load {pkl_path}: {e}")
        return None


def extract_centerline_from_map(scenario: Dict) -> np.ndarray:
    """
    Extract centerline from Waymo scenario map data.
    
    Waymo scenarios have map features including lane centerlines.
    """
    try:
        # Try to get map features
        if 'map_features' in scenario:
            map_features = scenario['map_features']
            
            # Find lane centerlines
            centerlines = []
            for feature_id, feature in map_features.items():
                if 'polyline' in feature or 'centerline' in feature:
                    line = feature.get('polyline', feature.get('centerline', []))
                    if len(line) > 0:
                        centerlines.append(np.array(line))
            
            # Use the longest centerline as main route
            if centerlines:
                longest = max(centerlines, key=len)
                return longest[:, :2]  # xy only
        
        # Fallback: use SDC (self-driving car) trajectory as centerline
        if 'sdc_track' in scenario or 'tracks' in scenario:
            tracks = scenario.get('tracks', {})
            if 'sdc' in tracks or 0 in tracks:
                sdc_track = tracks.get('sdc', tracks.get(0, {}))
                if 'state' in sdc_track:
                    states = sdc_track['state']
                    # Extract xy positions
                    if 'position' in states:
                        return np.array(states['position'])[:, :2]
                    elif isinstance(states, dict) and 'x' in states:
                        x = np.array(states['x'])
                        y = np.array(states['y'])
                        return np.stack([x, y], axis=1)
                    elif isinstance(states, np.ndarray) and states.shape[1] >= 2:
                        return states[:, :2]
        
        # Last resort: create dummy centerline
        return np.array([[0, 0], [100, 0], [200, 0]])
        
    except Exception as e:
        print(f"Warning: Could not extract centerline: {e}")
        return np.array([[0, 0], [100, 0]])


def compute_curvature_stats(centerline: np.ndarray) -> Dict:
    """Compute curvature statistics from centerline."""
    if len(centerline) < 3:
        return {
            'length_m': 0.0,
            'k_mean': 0.0,
            'k_p90': 0.0,
            'k_max': 0.0,
            'turn_sum_deg': 0.0,
        }
    
    # Compute cumulative distance
    diffs = np.diff(centerline, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    total_length = np.sum(segment_lengths)
    
    # Compute headings and curvature
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    headings = np.unwrap(headings)
    dheadings = np.diff(headings)
    
    # Curvature = angular change / arc length
    kappa = np.abs(dheadings[:-1] / (segment_lengths[1:] + 1e-6))
    
    return {
        'length_m': float(total_length),
        'k_mean': float(np.mean(kappa)) if len(kappa) > 0 else 0.0,
        'k_p90': float(np.percentile(kappa, 90)) if len(kappa) > 0 else 0.0,
        'k_max': float(np.max(kappa)) if len(kappa) > 0 else 0.0,
        'turn_sum_deg': float(np.degrees(np.sum(np.abs(dheadings)))),
    }


def count_vehicles_and_agents(scenario: Dict) -> Dict:
    """Count different agent types in scenario."""
    counts = {
        'vehicles': 0,
        'pedestrians': 0,
        'cyclists': 0,
    }
    
    try:
        if 'tracks' in scenario:
            tracks = scenario['tracks']
            for track_id, track_data in tracks.items():
                if track_id == 'sdc' or track_id == 0:
                    continue
                
                obj_type = track_data.get('object_type', track_data.get('type', 'vehicle'))
                
                if 'vehicle' in str(obj_type).lower() or obj_type == 1:
                    counts['vehicles'] += 1
                elif 'pedestrian' in str(obj_type).lower() or obj_type == 2:
                    counts['pedestrians'] += 1
                elif 'cyclist' in str(obj_type).lower() or 'bicycle' in str(obj_type).lower():
                    counts['cyclists'] += 1
                else:
                    counts['vehicles'] += 1  # default
        
        # Alternative: dynamic_map_states
        if 'dynamic_map_states' in scenario:
            # Count from dynamic states
            pass
            
    except Exception as e:
        print(f"Warning: Could not count agents: {e}")
    
    return counts


def generate_tokens_from_geometry(centerline: np.ndarray) -> str:
    """Generate maneuver tokens from centerline geometry."""
    if len(centerline) < 3:
        return "S"
    
    tokens = []
    
    # Compute turn angles
    diffs = np.diff(centerline, axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    headings = np.unwrap(headings)
    dheadings = np.diff(headings)
    
    # Sliding window to detect maneuvers
    window_size = 10
    for i in range(0, len(dheadings), window_size):
        window = dheadings[i:i+window_size]
        total_turn = np.sum(window)
        angle_deg = abs(np.degrees(total_turn))
        
        if angle_deg < 10:
            tokens.append('S')  # Straight
        elif angle_deg < 30:
            tokens.append('CL' if total_turn > 0 else 'CR')  # Slight curve
        elif angle_deg < 60:
            tokens.append('L' if total_turn > 0 else 'R')  # Turn
        else:
            tokens.append('U')  # U-turn or sharp turn
    
    return ' '.join(tokens) if tokens else 'S'


def extract_scenario_signature(scenario: Dict, scenario_id: str, file_path: str) -> WaymoScenarioSignature:
    """Extract signature from Waymo scenario."""
    
    # Extract centerline
    centerline = extract_centerline_from_map(scenario)
    
    # Compute geometry
    geom = compute_curvature_stats(centerline)
    
    # Count agents
    agents = count_vehicles_and_agents(scenario)
    
    # Estimate traffic density (vehicles per 100m)
    density = agents['vehicles'] / max(geom['length_m'] / 100.0, 0.1)
    
    # Generate tokens
    tokens = generate_tokens_from_geometry(centerline)
    
    # Estimate number of lanes (from map or default)
    num_lanes = 2.0  # default
    try:
        if 'map_features' in scenario:
            map_features = scenario['map_features']
            lane_counts = []
            for feature in map_features.values():
                if 'num_lanes' in feature:
                    lane_counts.append(feature['num_lanes'])
            if lane_counts:
                num_lanes = np.mean(lane_counts)
    except:
        pass
    
    # Estimate duration
    duration = 10.0  # default 10 seconds
    try:
        if 'timestamps' in scenario:
            duration = float(len(scenario['timestamps']) * 0.1)  # assuming 10Hz
    except:
        pass
    
    # Estimate lane changes from lateral movement
    lane_changes = max(0, int((geom['length_m'] / 100.0) * (density / 5.0)))  # heuristic
    
    return WaymoScenarioSignature(
        scenario_id=scenario_id,
        file_path=file_path,
        route_length_m=geom['length_m'],
        curvature_mean=geom['k_mean'],
        curvature_p90=geom['k_p90'],
        curvature_max=geom['k_max'],
        total_turn_deg=geom['turn_sum_deg'],
        num_lanes_mean=num_lanes,
        num_lane_changes_estimated=lane_changes,
        num_vehicles=agents['vehicles'],
        num_pedestrians=agents['pedestrians'],
        num_cyclists=agents['cyclists'],
        traffic_density_estimate=density,
        token_string=tokens,
        token_hash=hash(tokens),
        duration_seconds=duration,
        map_features="unknown",
    )


def load_all_scenarios(scenario_dir: str) -> List[WaymoScenarioSignature]:
    """Load and extract signatures from all Waymo .pkl files in directory."""
    scenario_dir = Path(scenario_dir)
    pkl_files = list(scenario_dir.glob("*.pkl"))
    
    print(f"\nFound {len(pkl_files)} .pkl files in {scenario_dir}")
    print("="*80)
    
    signatures = []
    
    for pkl_file in tqdm(pkl_files, desc="Processing scenarios"):
        scenario = load_waymo_scenario(str(pkl_file))
        if scenario is None:
            continue
        
        scenario_id = pkl_file.stem
        signature = extract_scenario_signature(scenario, scenario_id, str(pkl_file))
        signatures.append(signature)
    
    print(f"\n✓ Successfully extracted {len(signatures)} scenario signatures")
    return signatures


def cluster_scenarios(signatures: List[WaymoScenarioSignature], n_clusters: int = 8) -> Dict:
    """Cluster scenarios into buckets."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    
    # Extract features
    feature_matrix = np.array([sig.to_feature_vector() for sig in signatures])
    
    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_matrix)
    
    # Cluster
    print(f"\nClustering {len(signatures)} scenarios into {n_clusters} buckets...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(features_scaled)
    
    # Silhouette score
    sil_score = silhouette_score(features_scaled, labels)
    print(f"Silhouette score: {sil_score:.3f}")
    
    # Organize by bucket
    buckets = defaultdict(list)
    for sig, label in zip(signatures, labels):
        buckets[int(label)].append({
            'scenario_id': sig.scenario_id,
            'file_path': sig.file_path,
        })
    
    # Compute bucket stats
    bucket_stats = {}
    for bucket_id in buckets.keys():
        bucket_sigs = [sig for sig in signatures if sig.scenario_id in [s['scenario_id'] for s in buckets[bucket_id]]]
        
        bucket_stats[bucket_id] = {
            'size': len(buckets[bucket_id]),
            'scenarios': buckets[bucket_id],
            'avg_length_m': float(np.mean([s.route_length_m for s in bucket_sigs])),
            'avg_curvature': float(np.mean([s.curvature_mean for s in bucket_sigs])),
            'avg_num_vehicles': float(np.mean([s.num_vehicles for s in bucket_sigs])),
            'avg_density': float(np.mean([s.traffic_density_estimate for s in bucket_sigs])),
            'common_tokens': Counter(' '.join([s.token_string for s in bucket_sigs]).split()).most_common(3),
        }
    
    return {
        'n_clusters': n_clusters,
        'silhouette_score': float(sil_score),
        'buckets': bucket_stats,
        'cluster_centers': kmeans.cluster_centers_.tolist(),
    }


def save_results(signatures: List[WaymoScenarioSignature], clustering: Dict, output_path: str):
    """Save bucketing results."""
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
    
    print(f"\n✓ Saved to {output_path}")


def print_bucket_summary(clustering: Dict):
    """Print bucket summary."""
    print("\n" + "="*80)
    print("WAYMO SCENARIO BUCKETS - Virtual Tracks")
    print("="*80)
    
    for bucket_id in sorted(clustering['buckets'].keys()):
        stats = clustering['buckets'][bucket_id]
        print(f"\n📁 Bucket {bucket_id}: {stats['size']} scenarios")
        print(f"   Route length:  {stats['avg_length_m']:.1f}m")
        print(f"   Curvature:     {stats['avg_curvature']:.4f}")
        print(f"   Vehicles:      {stats['avg_num_vehicles']:.1f}")
        print(f"   Density:       {stats['avg_density']:.2f} veh/100m")
        print(f"   Common tokens: {', '.join([f'{tok}({cnt})' for tok, cnt in stats['common_tokens']])}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Bucket Waymo scenarios into virtual tracks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--scenario_dir', type=str, required=True,
                        help='Directory containing Waymo .pkl scenario files')
    parser.add_argument('--output', type=str, default='data/waymo_buckets.json',
                        help='Output JSON file')
    parser.add_argument('--n_clusters', type=int, default=8,
                        help='Number of buckets')
    
    args = parser.parse_args()
    
    # Load scenarios
    signatures = load_all_scenarios(args.scenario_dir)
    
    if len(signatures) == 0:
        print("❌ No scenarios loaded! Check your directory path.")
        exit(1)
    
    # Cluster
    clustering = cluster_scenarios(signatures, args.n_clusters)
    
    # Print summary
    print_bucket_summary(clustering)
    
    # Save
    save_results(signatures, clustering, args.output)
    
    print("\n✓ Bucketing complete!")
    print(f"\nNext step:")
    print(f"  python validate_buckets.py --buckets {args.output} --scenario_dir {args.scenario_dir}")

