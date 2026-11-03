"""CPU exercise: sequential SAC on two Pendulum dynamics; not a CRL benchmark."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from single_task import evaluate, write_json

ROOT = Path(__file__).resolve().parent


def save_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_env(gravity, output, label):
    return Monitor(gym.make("Pendulum-v1", g=gravity), str(output / f"{label}.monitor.csv"))


def make_model(env, seed):
    return SAC("MlpPolicy", env, device="cpu", seed=seed, verbose=0,
               learning_rate=3e-4, buffer_size=20_000, learning_starts=500,
               batch_size=64, train_freq=1, gradient_steps=1,
               policy_kwargs={"net_arch": [64, 64]})


class Trace(BaseCallback):
    """Development evaluations do not feed back into training or model selection."""
    def __init__(self, output, label, gravities, every):
        super().__init__()
        self.output, self.label, self.gravities, self.every = output, label, gravities, every
        self.rows = []

    def _on_training_start(self):
        self.start = time.perf_counter()

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            row = {"phase": self.label, "phase_steps": self.num_timesteps}
            for task, gravity in self.gravities.items():
                row[f"{task}_dev_return"] = evaluate(self.model, 10_000, 3, gravity)["mean_return"]
            row["elapsed_seconds_including_eval"] = time.perf_counter() - self.start
            self.rows.append(row)
            save_rows(self.output / f"{self.label}_curve.csv", self.rows)
            print(f"{self.label}: step={self.num_timesteps}, A={row['A_dev_return']:.1f}, "
                  f"B={row['B_dev_return']:.1f}", flush=True)
        return True


def evaluate_both(model, label, gravities, episodes):
    result = {task: evaluate(model, 20_000, episodes, gravity) for task, gravity in gravities.items()}
    print(f"{label}: A={result['A']['mean_return']:.1f}, B={result['B']['mean_return']:.1f}", flush=True)
    return result


def train_phase(model, output, label, args, gravities):
    updates_before = model._n_updates
    start = time.perf_counter()
    # Reset the phase counter and warm-up; this does NOT reinitialize networks/optimizers.
    model.learn(args.steps, reset_num_timesteps=True,
                callback=Trace(output, label, gravities, args.eval_every))
    if not all(bool(torch.isfinite(p).all()) for p in model.policy.parameters()):
        raise RuntimeError(f"Non-finite policy parameters in {label}")
    updates = model._n_updates - updates_before
    if updates <= 0:
        raise RuntimeError("No gradient updates. Use more than 500 steps per phase.")
    model.save(output / label)
    return {"phase_steps": model.num_timesteps, "gradient_updates": updates,
            "seconds_including_development_eval": time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", type=int, default=10_000, help="Steps per training phase; three phases total")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gravity-a", type=float, default=10.0)
    parser.add_argument("--gravity-b", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("name must be a single folder name")
    if args.steps <= 500 or min(args.eval_every, args.eval_episodes, args.gravity_a, args.gravity_b) <= 0:
        parser.error("steps must exceed 500; evaluation counts and gravity must be positive")
    if args.gravity_a == args.gravity_b:
        parser.error("Use different gravities for A and B")
    output = ROOT / "runs" / args.name
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    gravities = {"A": args.gravity_a, "B": args.gravity_b}
    metadata = {"kind": "two_task_CPU_learning_exercise_not_CKA_RL", "arguments": vars(args),
                "python": sys.version, "platform": platform.platform(),
                "versions": {p: importlib.metadata.version(p) for p in ["torch", "gymnasium", "stable-baselines3"]},
                "device": "cpu", "threads": 1, "task_gravities": gravities,
                "protocol": "Same reward/observation/action; change gravity. No task ID in policy input.",
                "B_switch": "Keep actor, critic, target networks, optimizers and entropy state; clear replay buffer; restart 500-step random warm-up.",
                "evaluation": "3 fixed development episodes; separate fixed held-out episodes; no tuning or early stopping"}
    write_json(output / "environment.json", metadata)
    random_result = evaluate_both(None, "random", gravities, args.eval_episodes)

    env_a = make_env(args.gravity_a, output, "train_A")
    sequential = make_model(env_a, args.seed)
    env_b = None
    try:
        timing_a = train_phase(sequential, output, "after_A", args, gravities)
        after_a = evaluate_both(sequential, "after_A", gravities, args.eval_episodes)

        # Core task switch: reuse learned networks, replace environment, remove old experience.
        env_b = make_env(args.gravity_b, output, "finetune_B")
        sequential.set_env(env_b, force_reset=True)
        sequential.replay_buffer.reset()
        assert sequential.replay_buffer.size() == 0
        timing_b = train_phase(sequential, output, "after_A_then_B", args, gravities)
        after_b = evaluate_both(sequential, "after_A_then_B", gravities, args.eval_episodes)
    finally:
        env_a.close()
        if env_b is not None:
            env_b.close()

    # A fresh model on B separates B difficulty from transfer through A initialization.
    independent_env = make_env(args.gravity_b, output, "scratch_B")
    try:
        independent = make_model(independent_env, args.seed)
        timing_scratch = train_phase(independent, output, "scratch_B", args, gravities)
        scratch_b = evaluate_both(independent, "scratch_B", gravities, args.eval_episodes)
    finally:
        independent_env.close()

    rows = [{"checkpoint": label, "A_mean_return": values["A"]["mean_return"],
             "B_mean_return": values["B"]["mean_return"]}
            for label, values in [("random", random_result), ("after_A", after_a),
                                  ("after_A_then_B", after_b), ("scratch_B", scratch_b)]]
    save_rows(output / "evaluation_matrix.csv", rows)
    summary = {"random": random_result, "after_A": after_a, "after_A_then_B": after_b, "scratch_B": scratch_b,
               "A_return_drop_after_B": after_a["A"]["mean_return"] - after_b["A"]["mean_return"],
               "B_final_return_difference_vs_scratch": after_b["B"]["mean_return"] - scratch_b["B"]["mean_return"],
               "timing": {"train_A": timing_a, "finetune_B": timing_b, "scratch_B": timing_scratch},
               "limitations": ["One seed and two related dynamics only; no guaranteed forgetting.",
                               "B final-return difference is not a standard forward-transfer metric.",
                               "Full SAC state is retained; does not isolate actor-only knowledge transfer.",
                               "No CKA-RL, knowledge merging, or historical replay method implemented.",
                               "No normalized cross-task average; interpret each task return separately."]}
    write_json(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ["A_return_drop_after_B", "B_final_return_difference_vs_scratch"]}, indent=2))
    print(f"DONE: {output}")


if __name__ == "__main__":
    main()
