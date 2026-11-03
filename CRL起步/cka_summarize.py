"""Export original TensorBoard scalars and summarize completed CPU stages."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def summarize(output):
    rows = []
    summaries = ["# CPU 短实验实际结果", "", "仅用于验证程序流程，不代表论文性能复现。", "",
                 "| 方法 | 阶段/任务 | 训练及原评估秒数 | 旧向量数 | 触发合并 | 当前任务回报均值 | 当前任务末步成功率 | 回合任一步成功率 |",
                 "|---|---|---:|---:|---|---:|---:|---:|"]
    for result_file in sorted(output.glob("*/stage_*/result.json")):
        r = json.loads(result_file.read_text(encoding="utf-8"))
        event_files = list((result_file.parent / "runs").rglob("events.out.tfevents.*"))
        assert len(event_files) == 1
        events = EventAccumulator(str(event_files[0]), size_guidance={"scalars": 0})
        events.Reload()
        latest = {}
        with result_file.with_name("training_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["tag", "step", "value"])
            for tag in events.Tags()["scalars"]:
                values = events.Scalars(tag)
                for event in values:
                    assert math.isfinite(event.value), tag
                    writer.writerow([tag, event.step, event.value])
                latest[tag] = values[-1].value
        r["latest_logged_metrics"] = latest
        result_file.write_text(json.dumps(r, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        rows.append(r)
        cfg = r["config"]
        current = [v for v in r["evaluation"] if v["task_id"] == cfg["task_id"]]
        summaries.append(f"| {cfg['model_type']} | {cfg['index']}/{cfg['task_id']} | {r['training_and_raw_eval_seconds']:.1f} | {r['stored_old_vectors']} | {r['merge_triggered']} | {mean(v['return'] for v in current):.2f} | {mean(v['success_final'] for v in current):.3f} | {mean(v['success_any'] for v in current):.3f} |")
    assert rows, "No completed stages"
    (output / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    summaries += ["", "训练步数、优化器更新次数、参数变化、重载误差、合并具体配对详见各阶段 result.json。",
                  "当前任务回报采用额外评估的 tanh 确定性动作；原脚本未经 tanh 的评估保存在 train.log 和 training_metrics.csv 中。",
                  "evaluation.csv 另外包含每个阶段对先前任务的复测；它是诊断数据，不能直接作为论文 PERF/FWT。"]
    (output / "结果摘要.md").write_text("\n".join(summaries) + "\n", encoding="utf-8")
    print("\n".join(summaries))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    summarize(parser.parse_args().output.resolve())
