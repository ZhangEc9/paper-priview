# 2026-11-03 实测记录

此目录保存轻量结果快照，数值来自公司 Windows CPU 的真实运行。

- [CKA 短实验摘要](cka_smoke/结果摘要.md)：simple 单任务和 CKA 四任务，任务各 1,500 步，知识池容量 2。
- [逐回合评估](cka_smoke/evaluation.csv)：原模型保存重载后，用 tanh 确定性动作额外评估。
- [完整汇总](cka_smoke/results.json)：参数更新次数、冻结和重载检查、合并配对及原始指标。
- `cka_smoke/{方法}/stage_*`：逐阶段配置、结果、原始文本日志和 TensorBoard 标量 CSV。
- `cka_environment_check.json`：版本、源码提交和前四任务的完整 500 步环境检查。
- `my_two_tasks_10k`：Pendulum 两任务顺序训练和独立训练对照的结果。
- `my_knowledge_probe`：Pendulum 参数插值结果。
- `failed_attempts`：首次启动器类型注解问题，以及旧 tyro 与 argparse 不兼容的失败日志；不计入训练结果。

JSON 和日志中的绝对路径是当时本机路径，作为来源记录保留，换电脑时应重新由启动器生成配置，不能直接照抄路径。模型权重、回放数据、TensorBoard 二进制事件和第三方论文没有上传；标量已完整导出到 CSV。

注意：本次成功率全为 0，程序流程验证通过；这不是论文性能复现。训练种子只有 1，每个当前任务额外评估只有 3 回合。失败记录和成功记录均未用于筛选报告中的学习成绩。
