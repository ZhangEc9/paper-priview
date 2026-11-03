"""Evaluate fixed actor parameter interpolation; no training, not CKA-RL."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from single_task import evaluate, write_json

ROOT = Path(__file__).resolve().parent


def interpolate_actor(destination, actor_a, actor_b, alpha):
    """Mix actor parameters only. Critic state is unused during predict/evaluate."""
    params_a = dict(actor_a.named_parameters())
    params_b = dict(actor_b.named_parameters())
    params_out = dict(destination.named_parameters())
    if params_a.keys() != params_b.keys() or params_a.keys() != params_out.keys():
        raise ValueError("Actor architectures differ")
    with torch.no_grad():
        for name, out in params_out.items():
            if params_a[name].shape != params_b[name].shape:
                raise ValueError(f"Incompatible parameter: {name}")
            # theta(alpha) = theta_A + alpha * (theta_B - theta_A)
            out.copy_((1 - alpha) * params_a[name] + alpha * params_b[name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="my_two_tasks_10k", help="Existing two_task.py run name")
    parser.add_argument("--name", required=True, help="New output run name")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    for name in [args.source, args.name]:
        if Path(name).name != name or name in (".", ".."):
            parser.error("Use single folder names")
    if args.episodes < 1:
        parser.error("episodes must be positive")
    source = ROOT / "runs" / args.source
    metadata = json.loads((source / "environment.json").read_text(encoding="utf-8"))
    gravities = metadata["task_gravities"]
    torch.set_num_threads(1)
    model_a = SAC.load(source / "after_A.zip", device="cpu")
    model_b = SAC.load(source / "after_A_then_B.zip", device="cpu")
    if model_a.observation_space != model_b.observation_space or model_a.action_space != model_b.action_space:
        raise ValueError("Observation or action spaces differ")
    probe = copy.deepcopy(model_a)
    output = ROOT / "runs" / args.name
    output.mkdir(parents=True, exist_ok=False)
    weights_a = copy.deepcopy(model_a.actor.state_dict())
    weights_b = copy.deepcopy(model_b.actor.state_dict())
    rows = []
    detailed = {}
    # Predeclared grid. Do not select and report the best held-out alpha.
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        interpolate_actor(probe.actor, model_a.actor, model_b.actor, alpha)
        row = {"alpha": alpha}
        details = {}
        for task, gravity in gravities.items():
            # New initial states, distinct from development/held-out seeds in two_task.py.
            result = evaluate(probe, 40_000, args.episodes, gravity)
            row[f"{task}_mean_return"] = result["mean_return"]
            row[f"{task}_std_return"] = result["std_return"]
            details[task] = result
        rows.append(row)
        detailed[str(alpha)] = details
        print(f"alpha={alpha:.2f}: A={row['A_mean_return']:.1f}, B={row['B_mean_return']:.1f}", flush=True)
    # Endpoint identity is a code correctness property, not a performance claim.
    observations = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [-1.0, 0.0, -1.0]], dtype=np.float32)
    for alpha, reference in [(0.0, model_a), (1.0, model_b)]:
        interpolate_actor(probe.actor, model_a.actor, model_b.actor, alpha)
        np.testing.assert_allclose(probe.predict(observations, deterministic=True)[0],
                                   reference.predict(observations, deterministic=True)[0], rtol=1e-6, atol=1e-6)
    assert all(torch.equal(v, model_a.actor.state_dict()[k]) for k, v in weights_a.items())
    assert all(torch.equal(v, model_b.actor.state_dict()[k]) for k, v in weights_b.items())
    with (output / "parameter_interpolation.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    param_bytes = sum(p.numel() * p.element_size() for p in model_a.actor.parameters())
    write_json(output / "summary.json", {
        "kind": "fixed_actor_interpolation_learning_exercise_not_CKA_RL",
        "source_run": args.source, "task_gravities": gravities, "first_eval_seed": 40_000,
        "episodes": args.episodes, "alphas": [r["alpha"] for r in rows], "results": detailed,
        "endpoint_action_checks_passed": True, "source_actors_unchanged": True,
        "actor_parameter_bytes_only": {"one_actor": param_bytes, "two_separate_actors": 2 * param_bytes,
                                      "base_plus_full_size_delta": 2 * param_bytes},
        "limitations": ["Only actor parameters are mixed; critics are not trained or evaluated.",
                        "No alpha selected. Choosing a method requires separate development and held-out evaluation.",
                        "Independent frozen policies require task identity for routing; no router is learned here.",
                        "Byte count excludes checkpoint overhead, critics, optimizers, replay and metadata.",
                        "One training seed and two related tasks; interpolation need not preserve performance."]})
    print(f"DONE: {output}")


if __name__ == "__main__":
    main()
