# 代码说明与 CKA-RL 对接

日期：2026-11-02。

## 1. 目录

```text
CRL起步/
  00_从这里开始.md                 今天的命令与学习自测
  01_三件事步骤.md                 单任务、短序列、成本测量
  02_代码说明与CKA对接.md          本文
  single_task.py                   CPU SAC 单任务脚本
  requirements-cpu.txt             直接依赖
  requirements-cpu-lock.txt        实际安装完整版本
  .venv/                          最终 CPU 环境
  runs/                           每次运行一个独立文件夹
  CKA-RL/                         官方代码，保持下载版本
```

上游仓库：https://github.com/Fhujinwu/CKA-RL

本次下载 commit：`aa8817ace9f53120d5d205b9e4cafb3cdaa47803`。论文为 NeurIPS 2025，学姐给的 PDF 是 2026 年修订版。上游代码尚未在本机执行 Meta-World 实验。

## 2. 今天的训练流程

```text
创建训练环境和独立评估环境
  → 在固定评估初始状态上测随机策略和未训练 SAC
  → 收集 (state, action, reward, next_state, done) 到经验池
  → 500 步后开始从经验池抽样更新 actor/critic
  → 每 1000 步用独立环境做开发评估并记录
  → 训练结束，用另一组固定初始状态评估
  → 保存模型、重载并核对同一输入的动作一致
  → 输出摘要、CSV、曲线
```

SAC 的经验回放首先是单任务 off-policy RL 的训练机制。以后 CRL 中的历史任务记忆解决的是跨任务问题，两者有关联，但今天有 replay buffer 不等于已经实现持续学习。

## 3. 函数阅读顺序

| 函数／对象 | 你先看懂什么 |
|---|---|
| `main()` | 命令行如何决定运行名、步数、种子 |
| `train()` | 环境、算法、评估和保存如何串起来 |
| `SAC(...)` | 网络 `[64,64]`、CPU、经验池 20000、batch 64、开始更新步数 500 |
| `evaluate()` | reset、predict、step、terminated/truncated、累计回报 |
| `Progress` | 周期性评估、CSV 和内存采样 |
| `watch()` | 重新加载训练模型并生成演示 |

梯度公式在 Stable-Baselines3 的 SAC 实现中；这份脚本没有重新发明 SAC。可在编辑器中跳转 `SAC.train()` 看 critic target、actor loss、熵系数和 target update。暂时不必每行都读懂。

训练步数是环境交互次数；梯度更新次数另报；回合数又是另一个概念。不要把三者混为一谈。Pendulum 每局通常最多 200 步，时间上限结束的 `truncated` 与任务自然终止的 `terminated` 都需要正确处理。

最终模型保存不含 replay buffer，因此用于评估与演示；若要严格续训，需另外设计经验池和随机数状态保存。

## 4. CKA-RL 原代码对应位置

根目录是 `./CKA-RL`。

| 文件（相对原仓库） | 作用 |
|---|---|
| `experiments/meta-world/tasks.py` | 任务名称、顺序、环境构造；第 0 个是 hammer-v2 |
| `experiments/meta-world/run_sac.py` | 单任务 SAC 训练、评估、模型选择与保存；接入不同 agent |
| `experiments/meta-world/run_experiments.py` | 顺序启动多个任务，传入历史模型路径 |
| `experiments/meta-world/models/cka_rl.py` | `CkaRlAgent`、历史知识向量加载、组合系数和合并 |
| `experiments/meta-world/models/fuse_module.py` | 融合基础参数、历史向量和当前增量 |
| `experiments/meta-world/models/shared_arch.py` | 共享 MLP 架构 |

建议阅读 `CkaRlAgent.forward()` → `setup_vectors()` → `setup_alpha()` → `merge_vectors()`，同时回看论文 §4。

## 5. 已发现的复现注意点

1. 原 `requirements.txt` 是较完整的 Linux 环境导出，包含 `triton` 和 `nvidia-*`；不要在 Windows 上原样安装整份。Meta-World 还锁定了具体 git commit，不要随手换成最新版再声称复现原设定。
2. 启动器默认执行 `python3`，且用字符串拼接后按空格分割参数；Windows 解释器命名和含空格路径可能出问题。当前阶段直接调用正确解释器下的 `run_sac.py` 更易排查。
3. 不要把 CPU 入门包版本替换到原代码后默认视为兼容。较新 PyTorch 的 `torch.load` 默认行为也可能与原仓库保存整个模块的方式不同。学校环境另行核对。
4. 原 Meta-World 代码对均值头、logstd 头以及不同参数部分分别计算相似度、执行合并，完整知识单元的对应关系需要检查。这个问题先纳入复现审查，不预设为方法贡献。
5. 任务策略保存、知识池与历史任务评估的关系要查清；不能只看 `pool_size` 就宣称固定总内存、单模型免遗忘。
6. 当前下载的 `run_sac.py` 虽然声明了 `eval_every`，但未使用这个字段实现周期性评估，能看到的 `eval_agent()` 调用在训练末尾。学校复现时需要另外核对评估脚本，不能仅传入参数就认定有中间评估曲线。今天的 `single_task.py` 则明确实现了周期性开发评估。

## 6. 第一轮研究记录表

每次实验至少写：代码 commit、环境版本、环境名与任务顺序、种子、模型／算法、交互步数、更新次数、评估回合数、指标、墙钟时间、内存、保存状态及实际字节。

当前候选是知识合并准则；最先要做的是可信基线和查新，不是在 CPU 入门阶段开始堆叠新模块。
