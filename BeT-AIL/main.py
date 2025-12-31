import os
import json
import argparse
from pathlib import Path
from dataclasses import asdict

import numpy as np
import gym  # if you're on Gymnasium, you can keep `import gymnasium as gym`
from tqdm import tqdm
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.wrappers import RolloutInfoWrapper

from utils.gym_utils import sample_expert_transitions, sample_expert_trajectories
from run_adversarial import run_adversarial
from utils.config import ImitationExpConfig, DecisionTransformerConfig
from utils.discriminator_augment import (
    DiscAugmentListConfig,
    DiscAugmentGradientPenaltyConfig,
    DiscAugmentEntropyConfig,
)

from online_dt.train_online_DT import Experiment
from online_dt.bet_config import BeTConfig


ENV_ID = "MountainCarContinuous-v0"


def make_vec_env(env_id: str, seed: int) -> DummyVecEnv:
    """Factory + seeding; returns a single-process DummyVecEnv suitable for SB3/imitation."""
    def _thunk():
        env = gym.make(env_id)
        # Gymnasium-style reset seeding works in SB3>=2.0; harmless on classic gym too
        try:
            env.reset(seed=seed)
        except TypeError:
            # older gym fallback
            env.seed(seed)
        return RolloutInfoWrapper(env)

    venv = DummyVecEnv([_thunk])
    # SB3 seeding for the vec wrapper
    try:
        venv.seed(seed)
    except Exception:
        pass
    return venv


def _common_setup(seed: int):
    rng = np.random.default_rng(seed)
    disc_aug = DiscAugmentListConfig(
        config_list=[DiscAugmentGradientPenaltyConfig(), DiscAugmentEntropyConfig()]
    )
    exp_config = ImitationExpConfig(disc_augment_reward=disc_aug)

    # ensure RL logging dir exists (sample_expert_* often writes/reads here)
    rl_dir = Path(exp_config.logging_directory) / "rl"
    rl_dir.mkdir(parents=True, exist_ok=True)

    return rng, exp_config, rl_dir


def run_AIL(seed: int = 0):
    train_env = make_vec_env(ENV_ID, seed)
    rng, exp_config, rl_dir = _common_setup(seed)

    # Get (or create) expert transitions
    print("\n" + "="*60)
    print("PHASE 1/2: Training Expert and Collecting Demonstrations")
    print("="*60)
    transitions = sample_expert_transitions(train_env, rng, dir=str(rl_dir))
    print(f"[AIL] Loaded expert transitions: {len(transitions.acts)} steps")

    # Adversarial IL (e.g., GAIL/AIRL/your BeTAIL variant)
    print("\n" + "="*60)
    print("PHASE 2/2: Adversarial Imitation Learning")
    print("="*60)
    run_adversarial(train_env, exp_config, transitions)
    print("\n✓ [AIL] Training complete.")


def run_BeT(seed: int = 0):
    train_env = make_vec_env(ENV_ID, seed)
    eval_env = make_vec_env(ENV_ID, seed + 1)

    rng, exp_config, rl_dir = _common_setup(seed)

    bet_config = BeTConfig()
    
    # Gather full expert trajectories for DT pretraining
    print("\n" + "="*60)
    print("PHASE 1/2: Training Expert and Collecting Trajectories")
    print("="*60)
    transitions = sample_expert_trajectories(
        train_env, rng, dir=str(rl_dir), expert_timesteps=22_000
    )
    total_steps = sum(len(traj) for traj in transitions)
    print(f"✓ [BeT] Collected expert trajectories: {len(transitions)} episodes, {total_steps} steps")

    # Kick off Online DT/BeT experiment
    print("\n" + "="*60)
    print("PHASE 2/2: Behavior Transformer Pretraining")
    print(f"  Max iterations: {bet_config.max_pretrain_iters}")
    print(f"  Updates per iteration: {bet_config.num_updates_per_pretrain_iter}")
    print("="*60)
    
    experiment = Experiment(
        asdict(bet_config),
        env=train_env,
        eval_env=eval_env,
        transitions=transitions,
    )

    onlinedt_config = Path(experiment.logger.log_path) / "online_dt_config.json"
    with onlinedt_config.open("w") as f:
        json.dump(asdict(bet_config), f, indent=2)

    print(f"[BeT] Logging to: {experiment.logger.log_path}\n")
    
    # Many repos implement Experiment.__call__; if yours uses .run(), adjust accordingly.
    bet_path = experiment(bet_config)
    
    print(f"\n✓ [BeT] Saved DT/BeT model to: {bet_path}")

    return str(onlinedt_config), bet_path, transitions


def run_BeT_AIL(onlinedt_config: str, bet_path: str, transitions=None, seed: int = 0):
    train_env = make_vec_env(ENV_ID, seed)
    rng, _, rl_dir = _common_setup(seed)

    disc_aug = DiscAugmentListConfig(
        config_list=[DiscAugmentGradientPenaltyConfig(), DiscAugmentEntropyConfig()]
    )
    # aug_action_range = alpha in the paper
    bet_aug_config = DecisionTransformerConfig(
        enable=True,
        path_to_model=bet_path,
        path_to_exp_config=onlinedt_config,
        aug_action_range=0.1,
    )
    exp_config = ImitationExpConfig(
        disc_augment_reward=disc_aug,
        decision_transformer_config=bet_aug_config,
    )

    print("\n" + "="*60)
    print("PHASE 3/3: BeT-AIL Adversarial Training with BeT Augmentation")
    print("="*60)
    
    if transitions is None:
        transitions = sample_expert_transitions(train_env, rng, dir=str(rl_dir))
        print(f"[BeT-AIL] Loaded expert transitions: {len(transitions.acts)} steps")
    else:
        print(f"[BeT-AIL] Reusing provided expert transitions")

    run_adversarial(train_env, exp_config, transitions)
    print("\n✓ [BeT-AIL] Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BeT-AIL on MountainCarContinuous-v0")
    parser.add_argument("--algorithm", type=str, default="BeT-AIL",
                        choices=["AIL", "BeT", "BeT-AIL"],
                        help="Training algorithm to use")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    args = parser.parse_args()

    print("\n" + "="*60)
    print(f"  BeT-AIL Training - Algorithm: {args.algorithm}")
    print(f"  Environment: {ENV_ID}")
    print(f"  Seed: {args.seed}")
    print("="*60)

    if args.algorithm == "BeT-AIL":
        onlinedt_config, bet_path, transitions = run_BeT(seed=args.seed)
        run_BeT_AIL(onlinedt_config, bet_path, transitions=transitions, seed=args.seed)
    elif args.algorithm == "AIL":
        run_AIL(seed=args.seed)
    elif args.algorithm == "BeT":
        run_BeT(seed=args.seed)
    
    print("\n" + "="*60)
    print("  🎉 ALL TRAINING COMPLETE!")
    print("="*60 + "\n")
