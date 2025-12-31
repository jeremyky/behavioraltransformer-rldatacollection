# Data Directory

This directory contains all datasets, model checkpoints, and generated data for the BeT-AIL project.

## Structure

```
data/
├── waymo/              # Waymo dataset files
│   ├── raw/           # Original Waymo scenario files (.pkl)
│   ├── converted/     # Converted/processed scenarios
│   ├── buckets/       # Bucketing results (JSON files)
│   └── trajectories/  # Expert trajectory recordings
├── rig/               # Human-in-the-loop data from driving rig
│   ├── takeovers/     # Human takeover data
│   ├── corrections/   # Corrective trajectories
│   └── configs/       # Scenario configs used in rig
└── models/            # Trained model checkpoints
    ├── bet/          # Behavior Transformer models
    └── residuals/    # Residual policy models
```

## File Organization

### Waymo Data
- **Raw scenarios**: Place original Waymo `.pkl` files in `waymo/raw/`
- **Converted scenarios**: Processed scenarios go in `waymo/converted/`
- **Buckets**: JSON files from scenario bucketing go in `waymo/buckets/`
- **Trajectories**: Expert trajectory recordings go in `waymo/trajectories/`

### Rig Data
- **Takeovers**: Human takeover events and logs
- **Corrections**: Corrective trajectories from human interventions
- **Configs**: Scenario configuration files used during rig sessions

### Models
- **bet/**: Behavior Transformer model checkpoints
- **residuals/**: Residual policy model checkpoints

## Git Ignore

Large data files (`.pkl`, `.npz`, model checkpoints) are gitignored. Only small metadata files (`.json`, `.yaml`, `.txt`) are tracked in git.

## Migration

To migrate existing Waymo files to this structure, run:
```bash
./setup_data_structure.sh
```

This will move files from:
- `rig/datasets/waymo_converted_test/` → `data/waymo/converted/`
- `rig/md_hybrid_and_replay/waymo_database/` → `data/waymo/converted/waymo_database/`
- `rig/md_autocenter_old/scenarionet/waymo_dataset/` → `data/waymo/converted/waymo_dataset_old/`

