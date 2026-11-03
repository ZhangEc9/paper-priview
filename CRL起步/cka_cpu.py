"""CPU harness around pinned, unmodified CKA-RL training code.

smoke: independent SAC + 4 CKA tasks, capacity 2, shortened warmup.
run: configurable sequential CKA or independent SAC, upstream warmup by default.
Each stage is a separate process; raw upstream evaluation remains in its log.
Additional evaluation uses bounded deterministic actions in fresh environments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import types

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "CKA-RL" / "experiments" / "meta-world"
CKA_COMMIT = "aa8817ace9f53120d5d205b9e4cafb3cdaa47803"
MW_COMMIT = "c822f28f582ba1ad49eb5dcf61016566f28003ba"


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def git_revision(folder):
    return subprocess.check_output(["git", "-C", str(folder), "rev-parse", "HEAD"], text=True).strip()


def configure():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    sys.path.insert(0, str(UPSTREAM))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # MuJoCo 2.3.7 on Windows cannot open XMLs under Chinese directory names.
    # Cache byte-identical assets under an ASCII path; source checkout stays clean.
    import metaworld.envs.asset_path_utils as assets
    source = Path(assets.ENV_ASSET_DIR_V2)
    if os.name == "nt" and not str(source).isascii():
        cache = Path(tempfile.gettempdir()) / "cka_rl_assets" / MW_COMMIT / "assets_v2"
        if not str(cache).isascii():
            raise RuntimeError("Set TEMP to a writable ASCII directory for MuJoCo 2.3.7")
        for file in source.rglob("*"):
            if file.is_file():
                dest = cache / file.relative_to(source)
                digest = hashlib.sha256(file.read_bytes()).digest()
                if not dest.exists() or hashlib.sha256(dest.read_bytes()).digest() != digest:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(file, dest)
                assert hashlib.sha256(dest.read_bytes()).digest() == digest
        assets.ENV_ASSET_DIR_V2 = str(cache)


def doctor():
    configure()
    import numpy as np
    import torch
    import gymnasium
    import mujoco
    import metaworld.envs.asset_path_utils as assets
    from tasks import get_task
    rows = []
    for task_id in range(4):
        np.random.seed(100 + task_id)
        env = get_task(task_id)
        env.action_space.seed(100 + task_id)
        obs, _ = env.reset()
        assert obs.shape == (39,) and np.isfinite(obs).all()
        for i in range(500):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert np.isfinite(obs).all() and np.isfinite(reward)
            assert not terminated and truncated == (i == 499)
        rows.append({"task_id": task_id, "obs_dim": 39, "action_dim": 4, "horizon": 500})
        env.close()
    result = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
              "gymnasium": gymnasium.__version__, "mujoco": mujoco.__version__, "tasks": rows,
              "asset_directory": assets.ENV_ASSET_DIR_V2,
              "cka_commit": git_revision(ROOT / "CKA-RL"),
              "metaworld_commit": git_revision(ROOT / "vendor" / "Metaworld")}
    dump(ROOT / "cka_environment_check.json", result)
    print(json.dumps(result, indent=2))


def stage(config_path):
    configure()
    import numpy as np
    import torch
    from tasks import get_task
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    initial = {}
    merge_events = []
    from models import CkaRlAgent
    original_merge = CkaRlAgent.merge_vectors

    def observe_merge(self, mean_vectors, logstd_vectors):
        before = self.num_vectors
        snapshots = [{k: v.detach().clone() for k, v in d.items()}
                     for d in (mean_vectors, logstd_vectors)] if before > self.pool_size else []
        original_merge(self, mean_vectors, logstd_vectors)
        for head, old, new in zip(("mean", "logstd"), snapshots, (mean_vectors, logstd_vectors)):
            for name, element in old.items():
                similarity = torch.full((before, before), -1.)
                for i in range(before):
                    for j in range(i + 1, before):
                        similarity[i, j] = torch.cosine_similarity(element[i].flatten(), element[j].flatten(), dim=0)
                i, j = divmod(int(similarity.argmax()), before)
                error = float((new[name][-1] - (element[i] + element[j]) / 2).abs().max())
                assert i < j and error == 0 and len(new[name]) == self.pool_size
                merge_events.append({"head": head, "tensor": name, "before": before,
                                     "after": len(new[name]), "pair": [i, j],
                                     "cosine": float(similarity[i, j]), "mean_max_abs_error": error})

    CkaRlAgent.merge_vectors = observe_merge

    def capture(actor):
        initial.update({n: p.detach().clone() for n, p in actor.named_parameters()})

    source_path = UPSTREAM / "run_sac.py"
    source = source_path.read_text(encoding="utf-8")
    marker = "    start_time = time.time()"
    assert source.count(marker) == 1
    # Only a read-only parameter snapshot is inserted. No training changes.
    instrumented = source.replace(marker, "    _capture_initial(actor)\n" + marker)
    old_argv = sys.argv
    sys.argv = [str(source_path), *cfg["upstream_args"]]
    training_module = types.ModuleType("__main__")
    namespace = training_module.__dict__
    namespace.update({"__file__": str(source_path), "_capture_initial": capture})
    previous_main = sys.modules["__main__"]
    sys.modules["__main__"] = training_module
    started = time.perf_counter()
    try:
        exec(compile(instrumented, str(source_path), "exec", dont_inherit=True), namespace)
    finally:
        sys.argv = old_argv
        sys.modules["__main__"] = previous_main
    train_seconds = time.perf_counter() - started
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_files = list((Path(config_path).parent / "runs").rglob("events.out.tfevents.*"))
    assert len(event_files) == 1, "Expected exactly one training event file per stage"
    events = EventAccumulator(str(event_files[0]), size_guidance={"scalars": 0})
    events.Reload()
    latest_metrics = {}
    with Path(config_path).with_name("training_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "step", "value"])
        for tag in events.Tags()["scalars"]:
            values = events.Scalars(tag)
            for event in values:
                assert math.isfinite(event.value), f"Non-finite logged metric: {tag}"
                writer.writerow([tag, event.step, event.value])
            latest_metrics[tag] = values[-1].value
    actor = namespace["actor"]
    model = actor.model
    args = namespace["args"]
    assert all(torch.isfinite(p).all() for p in actor.parameters())
    changed = {n: float((p.detach() - initial[n]).abs().max()) for n, p in actor.named_parameters()}
    assert any(v > 0 for v in changed.values()), "Training did not change parameters"
    assert len(namespace["q_optimizer"].state) > 0, "Critic optimizer never updated"
    assert len(namespace["actor_optimizer"].state) > 0, "Actor optimizer never updated"
    frozen_changes = {n: changed[n] for n, p in actor.named_parameters() if not p.requires_grad}
    assert all(v == 0 for v in frozen_changes.values()), "Frozen knowledge vectors changed during training"
    checkpoint = Path(cfg["checkpoint"])
    if cfg["model_type"] == "cka-rl":
        from models import CkaRlAgent
        reloaded = CkaRlAgent.load(str(checkpoint), 39, 4, map_location="cpu")
    else:
        from models import SimpleAgent
        reloaded = SimpleAgent.load(str(checkpoint), map_location="cpu")
    probe = torch.zeros(3, 39)
    with torch.no_grad():
        reload_error = max(float((a - b).abs().max()) for a, b in zip(model(probe), reloaded(probe)))
    assert reload_error == 0, "Saved actor differs from trained actor"
    evaluation = []
    for task_id in cfg["eval_tasks"]:
        for episode in range(cfg["eval_episodes"]):
            eval_seed = 10000 + episode  # same environment seed per task across stages/methods
            np.random.seed(eval_seed)
            env = get_task(task_id)
            obs, _ = env.reset()
            total, success_any, success_final = 0., 0., 0.
            for step in range(1, 501):
                with torch.no_grad():
                    mean, _ = reloaded(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                    action = (torch.tanh(mean) * actor.action_scale + actor.action_bias)[0].numpy()
                assert np.isfinite(action).all() and env.action_space.contains(action)
                obs, reward, term, trunc, info = env.step(action)
                total += float(reward)
                success_final = float(info["success"])
                success_any = max(success_any, success_final)
                if term or trunc:
                    break
            assert step == 500, "Unexpected episode horizon"
            evaluation.append({"task_id": task_id, "episode": episode, "seed": eval_seed,
                               "return": total, "success_final": success_final,
                               "success_any": success_any, "steps": step})
            env.close()
    pool_count = getattr(model.fc_mean, "num_weights", 0)
    incoming = cfg["index"] if cfg["model_type"] == "cka-rl" else 0
    assert pool_count == min(incoming, cfg["pool_size"])
    report = {"config": cfg, "training_and_raw_eval_seconds": train_seconds,
              "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "latest_logged_metrics": latest_metrics,
              "critic_optimizer_steps": max(int(s["step"].item()) for s in namespace["q_optimizer"].state.values()),
              "actor_optimizer_steps": max(int(s["step"].item()) for s in namespace["actor_optimizer"].state.values()),
              "finite_parameters": True, "max_parameter_change": max(changed.values()),
              "alpha_parameter_changes": {n: v for n, v in changed.items() if "alpha" in n},
              "incoming_vectors": incoming, "stored_old_vectors": pool_count,
              "merge_triggered": incoming > cfg["pool_size"],
              "merge_events": merge_events, "frozen_parameter_changes": frozen_changes,
              "reload_max_abs_error": reload_error,
              "evaluation_action": "tanh(mean)*action_scale+action_bias",
              "evaluation": evaluation}
    dump(Path(config_path).with_name("result.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run(args):
    if git_revision(ROOT / "CKA-RL") != CKA_COMMIT or git_revision(ROOT / "vendor" / "Metaworld") != MW_COMMIT:
        raise RuntimeError("Pinned upstream commits do not match")
    for checkout in (ROOT / "CKA-RL", ROOT / "vendor" / "Metaworld"):
        if subprocess.run(["git", "-C", str(checkout), "diff", "--quiet", "HEAD", "--"]).returncode:
            raise RuntimeError(f"Tracked upstream files were changed: {checkout}")
    smoke = args.command == "smoke"
    tasks = [0, 1, 2, 3] if smoke else args.tasks
    steps = 1500 if smoke else args.steps
    learning_starts = 200 if smoke else args.learning_starts
    random_end = 300 if smoke else args.random_actions_end
    pool_size = 2 if smoke else args.pool_size
    if pool_size < 1 or args.eval_episodes < 1 or args.buffer_size < 128 or any(t not in range(24) for t in tasks):
        raise ValueError("Require pool>=1, eval_episodes>=1, buffer>=128, task IDs 0..23")
    if steps <= max(learning_starts + 2, random_end):
        raise ValueError("Use enough steps for both optimizer updates and policy-driven environment actions")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)  # never overwrite earlier experiments
    groups = [("simple", tasks[:1]), ("cka-rl", tasks)] if smoke else [(args.method, tasks)]
    results = []
    for method, sequence in groups:
        previous = []
        for index, task in enumerate(sequence):
            folder = output / method / f"stage_{index:02d}_task_{task}"
            folder.mkdir(parents=True)
            run_name = f"task_{task}__{method}__run_sac__{args.seed}"
            checkpoint = folder / "checkpoints" / run_name
            cli = ["--model-type", method, "--task-id", str(task), "--seed", str(args.seed),
                   "--no-cuda", "--total-timesteps", str(steps), "--buffer-size", str(args.buffer_size),
                   "--learning-starts", str(learning_starts), "--random-actions-end", str(random_end),
                   "--pool-size", str(pool_size), "--num-evals", "1",
                   "--save-dir", str(folder / "checkpoints"), "--tag", "tensorboard"]
            if previous and method == "cka-rl":
                cli += ["--prev-units", *previous]
            cfg = {"index": index, "task_id": task, "model_type": method, "pool_size": pool_size,
                   "seed": args.seed, "profile": "mechanics_smoke" if smoke else "short_baseline",
                   "checkpoint": str(checkpoint), "upstream_args": cli,
                   "eval_tasks": list(dict.fromkeys(sequence[:index + 1])),
                   "eval_episodes": args.eval_episodes, "cka_commit": CKA_COMMIT, "metaworld_commit": MW_COMMIT}
            config = folder / "config.json"
            dump(config, cfg)
            print(f"START {method} task={task} steps={steps}; log={folder / 'train.log'}", flush=True)
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
            with (folder / "train.log").open("w", encoding="utf-8") as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "_stage", str(config)],
                               cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            results.append(result)
            previous.append(str(checkpoint))
            print(f"DONE {method} task={task}: {result['training_and_raw_eval_seconds']:.1f}s, old_vectors={result['stored_old_vectors']}, merge={result['merge_triggered']}", flush=True)
            dump(output / "results.json", results)
            with (output / "evaluation.csv").open("w", newline="", encoding="utf-8-sig") as f:
                fields = ["method", "stage", "trained_task", "task_id", "episode", "seed", "return", "success_final", "success_any", "steps"]
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for row in results:
                    for evaluation in row["evaluation"]:
                        writer.writerow({"method": row["config"]["model_type"], "stage": row["config"]["index"],
                                         "trained_task": row["config"]["task_id"], **evaluation})
    print(f"Results: {output}", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "_stage":
        stage(sys.argv[2])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor", "smoke", "run"])
    parser.add_argument("--output", default=str(ROOT / "runs" / "cka_smoke"))
    parser.add_argument("--method", choices=["simple", "cka-rl"], default="cka-rl")
    parser.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--random-actions-end", type=int, default=10000)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=3)
    args = parser.parse_args()
    doctor() if args.command == "doctor" else run(args)


if __name__ == "__main__":
    main()
