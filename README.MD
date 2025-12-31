## scenario bucketing work

analyze pkl files, store scenario data in the format metadrive can load
conversion process: waymo .tfrecord files (raw waymo open dataset format)
conversion: scenarionet.convert_waymo converts them to metadrives scenario format. the result is a pkl file containing the original waymo data (trajectories, map features, metadata) in metadrives unified format so it can be replayed in the simulator.

sample pkl structure
{
    "metadata": {
        "dataset": "waymo",
        "source_file": "training_20s.tfrecord-00014-of-01000",
        "scenario_id": "dd0c8c27fdd6ef59"  # Original Waymo ID
    },
    "tracks": {...},           # Vehicle/pedestrian trajectories (from Waymo)
    "map_features": {...},     # Road geometry (from Waymo)
    "dynamic_map_states": {...} # Traffic lights (from Waymo)
}


bucket_waymo_scenarios: 

   python bucket_waymo_scenarios.py \
     --scenario_dir ../data/waymo/converted/waymo_converted_test_0/ \
     --output data/waymo/buckets/waymo_buckets.json \
     --n_clusters 8

    


run bucketing: uses kmeans on normalized features

Geometry: route length, curvature (mean/p90/max), total turn, lanes, lane changes
Traffic: vehicle/pedestrian/cyclist counts, density
Maneuver tokens: S (straight), CL/CR (curves), X (intersection), L/R (turns), etc.


To run scenarios together and check similarity:


validate_buckets:

   python validate_buckets.py \
     --buckets data/waymo/buckets/waymo_buckets.json \
     --scenario_dir ../data/waymo/converted/waymo_converted_test_0/ \
     --bucket_id 0 \
     --num_scenarios 3

========





'''
root/
├── data/                          # Top-level data directory
│   ├── waymo/
│   │   ├── raw/                   # Original Waymo scenario files (.pkl)
│   │   ├── converted/             # Converted/processed scenarios
│   │   ├── buckets/               # Bucketing results (JSON files)
│   │   └── trajectories/          # Expert trajectory recordings
│   ├── rig/                       # Human-in-the-loop data from driving rig
│   │   ├── takeovers/             # Human takeover data
│   │   ├── corrections/           # Corrective trajectories
│   │   └── configs/               # Scenario configs used in rig
│   └── models/                    # Trained model checkpoints
│       ├── bet/                   # Behavior Transformer models
│       └── residuals/             # Residual policy models
├── BeT-AIL/                       # Main codebase
├── rig/                           # Driving rig code (MetaDrive, etc.)
└── README.MD
'''


========

# 🚗 BeT-AIL: Scenario-Conditioned Driving with Behavior Transformers and Human-in-the-Loop Residuals

## Overview

This project explores **scenario-conditioned autonomous driving** by combining large-scale expert demonstrations (Waymo), **Behavior Transformers**, and **human-in-the-loop corrective data** collected through a custom driving rig.

The core idea is to:
1. **Group similar driving scenarios** from real-world datasets (Waymo / MetaDrive),
2. **Train a Behavior Transformer** to learn expert driving behavior within each scenario class,
3. **Correct deviations** using adversarial residual learning and human takeover data,
4. **Continuously improve policies** using high-quality human interventions rather than raw imitation alone.

This enables robust driving behavior that adapts to rare or ambiguous scenarios where pure imitation learning breaks down.

---

## Motivation

Imitation learning from large datasets such as Waymo performs well **in-distribution**, but often fails when:
- The agent slightly deviates from expert trajectories,
- The scenario is underspecified or ambiguous,
- Recovery behavior is required.

Inspired by the **BeT-AIL (Behavior Transformer + Adversarial Imitation Learning)** framework, this project introduces:
- **Scenario-aware policy learning**, and
- **Human corrective data collection** to close the gap between expert demonstrations and real-world deviations.

---

## High-Level Pipeline

Waymo Scenarios
│
▼
Scenario Bucketing / Clustering
│
▼
Behavior Transformer (Expert Policy)
│
▼
Adversarial Residual Correction
│
▼
Human-in-the-Loop Driving Rig
│
▼
Filtered High-Quality Takeover Data
│
▼
Reinforcement / Residual Policy Training



---

## 1. Scenario Bucketing

Large-scale Waymo scenarios are processed and grouped into **structurally similar driving situations**.

Scenario features may include:
- Road topology (intersections, merges, straight segments),
- Agent density and interaction patterns,
- Traffic control elements (signals, stop signs),
- Dynamic complexity (cut-ins, yielding, overtakes).

Bucketing is performed using metadata from:
- Waymo scenario files, and/or
- MetaDrive’s internal scenario representations.

This ensures that each learned policy is conditioned on a **coherent scenario distribution**, rather than averaging across unrelated behaviors.

---

## 2. Behavior Transformer (Expert Policy)

For each scenario bucket, a **Behavior Transformer** is trained following the BeT-AIL framework.

The transformer:
- Consumes expert state–action trajectories,
- Learns to autoregressively predict expert actions,
- Implicitly conditions on scenario structure through the data distribution.

This produces a strong **expert prior** that:
- Accurately imitates human driving in-distribution,
- Serves as a base policy for downstream correction.

---

## 3. Adversarial Residual Learning

Pure behavior cloning is brittle under distribution shift. To address this, we learn a **residual policy** that corrects deviations from expert behavior.

Key ideas:
- The behavior transformer outputs a nominal action,
- A learned residual modifies this action when deviations occur,
- Adversarial training encourages realism and stability.

The residual policy is especially important for:
- Recovery from off-nominal states,
- Mitigating compounding errors,
- Bridging gaps between simulation and human driving styles.

---

## 4. Human-in-the-Loop Driving Rig

Learned policies are integrated into a **custom driving rig** that allows humans to:
- Take control of vehicles inside Waymo-derived scenarios,
- Override the agent when necessary,
- Execute alternative but valid driving behaviors.

During each session, the system logs:
- Vehicle states and control inputs,
- Timing and context of human takeovers,
- Full trajectory rollouts before and after deviations.

The driving rig functions as a **data collection engine**, not just an evaluation tool.

---

## 5. Data Filtering and Quality Control

Not all human interventions are equally useful.

Filtering is applied to retain **high-quality corrective data**, such as:
- Interventions that improve safety or efficiency,
- Takeovers aligned with scenario constraints,
- Smooth, consistent control trajectories.

Low-signal or noisy interventions are discarded to avoid corrupting training.

The resulting dataset emphasizes:
- Meaningful corrections,
- Rare-but-important behaviors,
- Human strategies absent from expert logs.

---

## 6. Downstream Learning

Filtered human data is used to:
- Improve residual policies,
- Fine-tune scenario-conditioned behavior transformers,
- Support reinforcement learning with strong expert priors.

This creates a **closed-loop learning system** where:
- Expert data provides structure,
- Humans provide corrections,
- Policies improve iteratively.

---

## Project Goals

- Learn **robust, scenario-aware driving policies**
- Reduce failure under distribution shift
- Efficiently leverage human input
- Enable continual policy improvement

---

## Current Status

- [x] Waymo scenario ingestion
- [x] Scenario bucketing infrastructure
- [x] Driving rig with full trajectory logging
- [ ] Behavior Transformer training
- [ ] Adversarial residual learning
- [ ] Human-in-the-loop data collection experiments

---

## Future Work

- Multimodal conditioning (VLM-based scene understanding)
- Automatic takeover triggering
- Cross-scenario generalization analysis
- Real-world deployment and validation

---

## Repository Structure

```
kuo-research/
├── data/                          # Top-level data directory (gitignored)
│   ├── waymo/
│   │   ├── raw/                   # Original Waymo scenario files (.pkl)
│   │   ├── converted/             # Converted/processed scenarios
│   │   ├── buckets/               # Bucketing results (JSON files)
│   │   └── trajectories/          # Expert trajectory recordings
│   ├── rig/                       # Human-in-the-loop data from driving rig
│   │   ├── takeovers/             # Human takeover data
│   │   ├── corrections/           # Corrective trajectories
│   │   └── configs/               # Scenario configs used in rig
│   └── models/                    # Trained model checkpoints
│       ├── bet/                   # Behavior Transformer models
│       └── residuals/             # Residual policy models
├── BeT-AIL/                       # Main codebase
│   ├── bucket_waymo_scenarios.py  # Scenario bucketing
│   ├── validate_buckets.py        # Bucket validation
│   ├── record_metadrive_expert.py # Expert trajectory recording
│   └── data/                      # Small metadata files (buckets JSON, etc.)
├── rig/                           # Driving rig code (MetaDrive, etc.)
│   ├── md_hybrid_and_replay/      # Main driving rig implementation
│   ├── md_hybridcontrol/          # Hybrid control system
│   └── datasets/                  # Legacy dataset location (to be migrated)
└── README.MD
```

### Data Organization

All large data files (`.pkl`, `.npz`, model checkpoints) are gitignored. Only small metadata files (`.json`, `.yaml`, `.txt`) are tracked.

**Waymo Data Locations:**
- **Converted scenarios**: `data/waymo/converted/`
- **Bucketing results**: `data/waymo/buckets/`
- **Expert trajectories**: `data/waymo/trajectories/`

**Rig Data Locations:**
- **Human takeovers**: `data/rig/takeovers/`
- **Corrective trajectories**: `data/rig/corrections/`
- **Scenario configs**: `data/rig/configs/`

---

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd kuo-research
   ```

2. **Set up conda environment for BeT-AIL:**
   ```bash
   cd BeT-AIL
   
   # Option A: Use setup script (recommended)
   chmod +x setup_env.sh
   ./setup_env.sh
   
   # Option B: Manual setup
   conda env create -f environment.yml
   conda activate betail
   # Install PyTorch based on your system (see BeT-AIL/SETUP.md)
   ```

3. **Set up data directories:**
   ```bash
   # Create data structure
   mkdir -p data/waymo/{raw,converted,buckets,trajectories}
   mkdir -p data/rig/{takeovers,corrections,configs}
   mkdir -p data/models/{bet,residuals}
   ```

4. **Place Waymo data:**
   - Copy your Waymo scenario files (`.pkl`) to `data/waymo/converted/`
   - Or use the conversion scripts in `rig/` to process raw Waymo data

5. **Verify installation:**
   ```bash
   conda activate betail
   python -c "import metadrive; import sklearn; print('✓ Dependencies installed')"
   ```

---

## Quick Start

See `BeT-AIL/QUICKSTART_WAYMO.md` for detailed instructions on:
- Bucketing Waymo scenarios
- Validating buckets
- Recording expert trajectories
- Training Behavior Transformers
