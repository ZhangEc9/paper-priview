"""CPU SAC learning exercise on Pendulum-v1; not a CKA-RL reproduction."""
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
import psutil
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parent


def evaluate(model, seed: int, episodes: int, gravity: float) -> dict:
    """A fresh environment; identical episode seeds for each compared policy."""
    env = gym.make("Pendulum-v1", g=gravity)
    returns = []
    lengths = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            env.action_space.seed(seed + episode)
            reward_sum, length = 0.0, 0
            while True:
                action = env.action_space.sample() if model is None else model.predict(obs, deterministic=True)[0]
                obs, reward, terminated, truncated, _ = env.step(action)
                reward_sum += float(reward)
                length += 1
                if terminated or truncated:
                    break
            returns.append(reward_sum)
            lengths.append(length)
    finally:
        env.close()
    return {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns)),
            "episode_returns": returns, "episode_lengths": lengths, "first_seed": seed}


class Progress(BaseCallback):
    def __init__(self, output: Path, every: int, gravity: float):
        super().__init__()
        self.output, self.every, self.gravity = output, every, gravity
        self.rows = []
        self.peak_rss = 0

    def _on_training_start(self):
        self.start = time.perf_counter()

    def _on_step(self):
        rss = psutil.Process().memory_info().rss
        self.peak_rss = max(self.peak_rss, rss)
        if self.num_timesteps % self.every == 0:
            result = evaluate(self.model, seed=10_000, episodes=3, gravity=self.gravity)
            row = {"step": self.num_timesteps, "elapsed_seconds_including_eval": time.perf_counter() - self.start,
                   "development_mean_return": result["mean_return"],
                   "development_std_return": result["std_return"], "rss_mib": rss / 2**20}
            self.rows.append(row)
            with (self.output / "evaluations.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(self.rows)
            print(f"step={self.num_timesteps:>6}  dev_return={result['mean_return']:>8.1f}  "
                  f"elapsed={row['elapsed_seconds_including_eval']:.1f}s  rss={rss/2**20:.0f}MiB", flush=True)
        return True


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def train(args):
    output = ROOT / "runs" / args.name
    # Refuse accidental overwrite; different runs must have different names.
    output.mkdir(parents=True, exist_ok=False)
    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "gymnasium", "stable-baselines3", "numpy", "psutil"]}
    write_json(output / "environment.json", {"python": sys.version, "executable": sys.executable,
               "platform": platform.platform(), "versions": versions,
               "device": "cpu", "torch_threads": args.threads, "arguments": vars(args)})
    torch.set_num_threads(args.threads)
    env = Monitor(gym.make("Pendulum-v1", g=args.gravity), str(output / "train.monitor.csv"))
    model = SAC("MlpPolicy", env, device="cpu", seed=args.seed, verbose=0,
                learning_rate=3e-4, buffer_size=20_000, learning_starts=500,
                batch_size=64, train_freq=1, gradient_steps=1,
                policy_kwargs={"net_arch": [64, 64]})
    before = {name: p.detach().clone() for name, p in model.policy.named_parameters()}
    random_result = evaluate(None, 20_000, args.eval_episodes, args.gravity)
    initial_result = evaluate(model, 20_000, args.eval_episodes, args.gravity)
    print(f"Random policy: {random_result['mean_return']:.1f}; untrained SAC: {initial_result['mean_return']:.1f}", flush=True)
    callback = Progress(output, args.eval_every, args.gravity)
    start = time.perf_counter()
    try:
        model.learn(total_timesteps=args.steps, callback=callback)
        elapsed = time.perf_counter() - start
        final_result = evaluate(model, 20_000, args.eval_episodes, args.gravity)
        finite = all(bool(torch.isfinite(p).all()) for p in model.policy.parameters())
        change = sum(float((p.detach() - before[name]).square().sum())
                     for name, p in model.policy.named_parameters()) ** 0.5
        if not finite or not np.isfinite(change):
            raise RuntimeError("Non-finite model parameters detected; inspect this run before using it.")
        model.save(output / "model")
        loaded = SAC.load(output / "model.zip", device="cpu")
        probe, _ = env.reset(seed=30_000)
        action1 = model.predict(probe, deterministic=True)[0]
        action2 = loaded.predict(probe, deterministic=True)[0]
        np.testing.assert_allclose(action1, action2, rtol=1e-6, atol=1e-6)
        buffer = model.replay_buffer
        buffer_array_bytes = sum(value.nbytes for value in vars(buffer).values() if isinstance(value, np.ndarray))
        summary = {"kind": "single_task_SAC_learning_exercise_not_CKA_RL",
                   "env_id": "Pendulum-v1", "gravity": args.gravity,
                   "seed": args.seed, "timesteps": model.num_timesteps,
                   "gradient_updates": model._n_updates, "learning_starts": 500,
                   "random_policy": random_result, "initial_policy": initial_result, "trained_policy": final_result,
                   "return_gain_over_random": final_result["mean_return"] - random_result["mean_return"],
                   "return_gain_over_initial": final_result["mean_return"] - initial_result["mean_return"],
                   "learning_wall_seconds_including_periodic_eval": elapsed,
                   "steps_per_second_including_periodic_eval": model.num_timesteps / elapsed,
                   "sampled_peak_process_rss_mib": callback.peak_rss / 2**20,
                   "replay_buffer_allocated_array_bytes": buffer_array_bytes,
                   "checkpoint_bytes": (output / "model.zip").stat().st_size,
                   "policy_parameter_change_l2_including_critic_targets": change,
                   "all_policy_parameters_finite": finite, "checkpoint_reload_action_matches": True,
                   "limitations": ["one seed; not research evidence", "small network and Pendulum only",
                                   "speed does not predict Meta-World training cost",
                                   "RSS is sampled and is not GPU memory",
                                   "checkpoint does not include replay buffer"]}
        write_json(output / "summary.json", summary)
        plot(callback.rows, random_result["mean_return"], output)
        print(f"DONE: trained={final_result['mean_return']:.1f}; updates={model._n_updates}; "
              f"wall={elapsed:.1f}s; reload check passed", flush=True)
        print(f"Results: {output}", flush=True)
    finally:
        env.close()


def plot(rows, random_return, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    if rows:
        ax.plot([r["step"] for r in rows], [r["development_mean_return"] for r in rows], marker="o", label="SAC: 3 development episodes")
    ax.axhline(random_return, color="gray", linestyle="--", label="Random: held-out initial states")
    ax.set(xlabel="Environment steps", ylabel="Episode return (higher is better)",
           title="Pendulum-v1 | CPU learning exercise | one seed")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output / "learning_curve.png", dpi=150)
    plt.close(fig)


def watch(args):
    torch.set_num_threads(args.threads)
    model = SAC.load(args.model, device="cpu")
    # This is a local visual demonstration, not another evaluation score.
    env = gym.make("Pendulum-v1", g=args.gravity, render_mode="human" if args.window else "rgb_array")
    frames = []
    obs, _ = env.reset(seed=args.seed)
    try:
        for step in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if not args.window and step % 2 == 0:
                from PIL import Image
                frames.append(Image.fromarray(env.render()).resize((320, 320)))
            if terminated or truncated:
                break
    finally:
        env.close()
    if frames:
        path = Path(args.model).resolve().parent / "policy.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("train")
    run.add_argument("--name", required=True, help="A new run folder name, e.g. my_first_10k")
    run.add_argument("--steps", type=int, default=10_000)
    run.add_argument("--eval-every", type=int, default=1000)
    run.add_argument("--eval-episodes", type=int, default=10)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--gravity", type=float, default=10.0)
    view = commands.add_parser("watch")
    view.add_argument("--model", required=True)
    view.add_argument("--seed", type=int, default=20_000)
    view.add_argument("--threads", type=int, default=1)
    view.add_argument("--gravity", type=float, default=10.0, help="Must match the training gravity")
    view.add_argument("--window", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.gravity <= 0:
        parser.error("threads and gravity must be positive")
    if args.command == "train":
        if min(args.steps, args.eval_every, args.eval_episodes) < 1:
            parser.error("steps, eval-every and eval-episodes must be positive")
        if Path(args.name).name != args.name or args.name in (".", ".."):
            parser.error("name must be a single folder name")
        train(args)
    else:
        watch(args)


if __name__ == "__main__":
    main()
