# CKA-RL 原方法：CPU 基线起步

本阶段运行官方 CKA-RL 的 Meta-World SAC 实现。没有加入使用感知合并，也没有根据 Pendulum 插值结果改方法。这里的「跑通」是训练、知识复用、容量溢出合并和保存重载成功；不等于复现论文表格。

仓库中的轻量结果快照见 [2026-11-03 实测记录](records/2026-11-03/README.md)。关于零成功率、完整排错经验和学校 GPU 计划，见 [08](08_今日流程经验与学校GPU计划.md)。以下 `D:\论文学习` 路径为本次实测电脑路径，换电脑后改为自己的仓库位置。

## 1. 使用哪份代码

| 项目 | 来源 / 锁定版本 |
|---|---|
| CKA-RL 官方仓库 | https://github.com/Fhujinwu/CKA-RL |
| CKA-RL 提交 | `aa8817ace9f53120d5d205b9e4cafb3cdaa47803` |
| Meta-World | https://github.com/Farama-Foundation/Metaworld |
| Meta-World 提交 | `c822f28f582ba1ad49eb5dcf61016566f28003ba` |
| 解释器 / 设备 | Python 3.11.15，PyTorch 2.1.2+cpu，单线程、单环境 |
| 关键环境包 | Gymnasium 0.29.1、MuJoCo 2.3.7、SB3 2.2.1、NumPy 1.26.3 |

官方训练入口为 `D:\论文学习\CRL起步\CKA-RL\experiments\meta-world\run_sac.py`。官方目录保留原样。新增 `cka_cpu.py` 启动官方训练代码，在内存中插入一次只读参数快照，并在原训练和原评估结束后检查参数、重载模型、额外评估。合并函数仅包裹记录器，仍执行官方函数，再核对平均结果。

本地新建 `.venv-cka`，与原 Pendulum 练习的 `.venv` 分开。完整环境版本见 `requirements-cka-cpu-lock.txt`，Meta-World 另按上述提交从本地源码安装。

## 2. 为这台电脑处理了什么

1. 官方 requirements 是 Linux 环境导出，包含 CUDA 等组件；这里改用 Windows CPU 依赖，保留关键算法库版本。
2. 原 tyro 0.6.3 在 Python 3.11.15 的 argparse 上报 `too many values to unpack`，将命令行解析库升级为 1.0.16。没有修改 SAC 更新式。
3. MuJoCo 2.3.7 在 Windows 下打不开中文路径中的 XML。启动器将 Meta-World 的 assets_v2 缓存到 `%TEMP%\cka_rl_assets\<提交号>\assets_v2`，逐文件校验 SHA256，再在运行时指定资源位置。资源约 40 MB；官方文件内容不变。
4. 用当前环境的 `sys.executable` 和参数列表启动训练，避免官方启动器的 `python3`、空格拆命令问题。
5. 单线程，回放池默认 20,000，适合这台机器目前的空闲内存。论文为 1,000,000，后续正式对照要明确统一回放池容量。

## 3. 现在直接怎么跑

在 PowerShell 中逐行运行；不需要激活环境，也不需要显卡。

```powershell
Set-Location 'D:\论文学习\CRL起步'
$CkaPython = '.\.venv-cka\Scripts\python.exe'
& $CkaPython .\cka_cpu.py doctor
```

`doctor` 对前四个任务各跑完整的 500 步随机动作，核对 39 维观测、4 维动作、数值有限和回合截断。结果保存到 `cka_environment_check.json`。

短实验：

```powershell
& $CkaPython .\cka_cpu.py smoke --output .\runs\cka_smoke_my_first --eval-episodes 3
& $CkaPython .\cka_summarize.py .\runs\cka_smoke_my_first
```

输出目录必须使用新名字，避免覆盖已有实验。这个命令顺序执行：

| 方法 | 任务序列 | 每任务训练步数 | 开始更新 / 结束随机动作 | 知识池 |
|---|---|---:|---|---:|
| 独立 SAC（simple） | hammer | 1,500 | 200 / 300 | 不使用 |
| CKA-RL | hammer → push-wall → faucet-close → push-back | 1,500 | 200 / 300 | 2 |

短实验保留原网络宽度 256、batch 128、学习率 0.001 和 SAC/CKA 更新逻辑；提前开始训练是为了快速验证梯度更新，属于明确缩短的调试配置。

CKA 的旧向量数应依次为 **0、1、2、2**。第四阶段加载时先得到 3 个候选向量，超过容量 2，触发原版余弦选对与算术平均。当前可训练参数与旧向量池分别存在，不能把模型全部参数都算进这个计数。

每个阶段先执行官方原评估 1 回合，再加载刚保存的模型，对已出现的每种任务额外评估 3 回合。额外评估使用固定种子组 10000–10002 和独立环境。简单基线在 smoke 中只跑 hammer，作用是验证 SAC；这还不是完整的逐任务独立训练对照。

## 4. 从哪些文件看结果

以本次成功运行目录 `D:\论文学习\CRL起步\runs\cka_smoke_20261103_v3` 为例：

- `结果摘要.md`：各阶段耗时、向量数、是否合并和当前任务评估结果。
- `evaluation.csv`：额外评估的逐回合回报、末步成功、回合任一步成功，包含先前任务的诊断复测。
- `results.json`：合并后的完整机器可读记录。
- 各方法的 `stage_*/config.json`：任务、种子和传给官方训练代码的完整参数。
- 各阶段 `train.log`：官方日志和原评估，发生错误也从这里排查。
- 各阶段 `training_metrics.csv`：官方 TensorBoard 标量的无损导出，包括训练成功和损失。
- 各阶段 `result.json`：优化器实际更新次数、参数是否有限、参数变化、旧向量冻结检查、模型重载误差、合并的头/张量/配对/相似度/平均误差。
- 各阶段 `checkpoints`：官方格式的策略文件。CKA 为 fc、fc_mean、fc_logstd；simple 为 model.pt。

这些检查不是「看见程序结束就算成功」：必须有 actor 和 critic 更新，参数发生变化，旧知识冻结参数完全不变，保存重载后的输出误差为 0。容量溢出时还会检查实际选对与算术平均结果。

### 2026-11-03 实际完成记录

| 阶段 | 训练及原评估耗时 | 旧向量数 | 合并 | 当前任务额外评估平均回报 | 成功率 |
|---|---:|---:|---|---:|---:|
| simple / hammer | 52.3 秒 | 0 | 否 | 459.80 | 0/3 |
| CKA / hammer | 53.9 秒 | 0 | 否 | 459.80 | 0/3 |
| CKA / push-wall | 55.9 秒 | 1 | 否 | 16.80 | 0/3 |
| CKA / faucet-close | 54.9 秒 | 2 | 否 | 1168.11 | 0/3 |
| CKA / push-back | 55.7 秒 | 2 | 是，3→2 | 4.68 | 0/3 |

每阶段 critic 更新 **1,299 次**，actor 更新 **1,298 次**；所有保存重载输出误差为 **0**，所有旧向量训练变化为 **0**，记录的损失等标量均有限。这里只能说明机制正常执行，1,500 步尚未学会任务。

最后一次合并的四个参数块分别选择：mean.weight 为 (0,1)，mean.bias 为 (0,2)，logstd.weight 和 logstd.bias 为 (0,1)，算术平均最大绝对误差均为 0。**官方实现对参数块独立选对，并非整条知识向量统一选一对。** 目前只记录这个事实，没有改变方法，也不能据此断言它影响性能。

只有一个旧向量时，softmax 恒等于 1，所以该阶段 alpha 和 alpha_scale 没有变化是预期现象；出现两个旧向量后，这两个参数确实发生了更新。这里的 alpha 是 CKA 的可训练混合参数，与先前 Pendulum 插值实验的手动 α 不同。

## 5. 两种评估必须区分

官方 `eval_agent` 直接使用 `actor(obs)` 返回的高斯均值，而训练采样和 `get_action` 的确定性动作都经过 tanh 和动作缩放。这里没有改官方评估，保留它用于核对源码行为；额外评估统一使用：

```python
action = tanh(mean) * action_scale + action_bias
```

两个结果分别记录，不混用。在额外评估里同时保存「最后一步成功」和「回合任一步成功」；二者可能不同，比较时必须使用同一口径。

官方的 `eval_every` 参数没有实际控制周期评估，因此当前保存了训练回合统计和末尾评估，没有凭这个参数声称已经得到周期性独立评估曲线。

随机性沿用官方实现：训练入口设定 Python / NumPy / Torch 的种子，`get_task` 再从 NumPy 抽取环境种子；锁定版本的 Meta-World 构造器会调用 `env.seed`，同时设置 action_space 等空间的种子。虽然训练入口本身没有显式写 action_space.seed，它实际已由环境内部设置，不能据此断定随机探索未设种子。额外评估在构建环境前重设固定的 NumPy 种子。跨硬件和跨版本不承诺逐位一致，性能结论仍需多个独立训练种子。

## 6. 下一轮学习实验

先用小预算比较完整的相同任务序列，不引入新机制：

```powershell
& $CkaPython .\cka_cpu.py run --method simple --tasks 0 1 2 3 --steps 20000 --pool-size 2 --seed 1 --output .\runs\sac_4tasks_20k_seed1
& $CkaPython .\cka_cpu.py run --method cka-rl --tasks 0 1 2 3 --steps 20000 --pool-size 2 --seed 1 --output .\runs\cka_4tasks_20k_seed1
```

这里 simple 对每个任务从头训练；CKA 才传入先前策略。默认 warmup 恢复为论文的 5,000 / 10,000，回放池仍为 20,000。20k 是学习预算预试验，Meta-World 可能仍全部零成功，不能据此判定方法无效。先根据实际速度和奖励曲线确定学校机器上的预算，再扩大种子。

论文规模为每任务 **1M 步**，Meta-World 使用前 20 个任务索引（10 个任务重复两遍），报告 **10 个种子**，文中知识池容量 **8**。源码训练入口默认容量 9、序列启动器默认 20，所以必须显式写容量。这里 run 默认 8，smoke 固定 2 以提前触发合并。

现在没有复现论文 PERF/FWT。FWT 需要相同任务、相同预算的从头训练参考曲线并按论文定义积分，不能用末尾回报或这份旧任务诊断表代替。

## 7. 你现在只做三件事

1. 看成功运行的 `结果摘要.md`，确认四个 CKA 阶段的旧向量数量为 0/1/2/2。
2. 打开最后阶段的 `result.json`，找到 `merge_events`，对照官方 `models/cka_rl.py` 的 `merge_vectors`，解释它分别对哪几个参数块选了哪一对、如何取平均。
3. 区分「合并过程成功」与「学会任务」：核对实际成功率，再把你最想验证的一个问题写下来。后续才用长预算比较容量 2 和容量 8、以及多个种子。

如果只是检查本次结果，不用再跑一遍相同 smoke。失败的早期尝试保存在不带 v3 的同日前缀目录，供排错，不参与结果比较。

## 8. 以后重新配置

```powershell
Set-Location 'D:\论文学习\CRL起步'
.\setup_cka_cpu.ps1 -PythonExe 'D:\Anaconda\envs\pythonproject\python.exe'
```

PythonExe 应指向 Python 3.11。脚本获取锁定源码、创建隔离环境、安装 CPU PyTorch 和完整依赖锁文件、安装 Meta-World，最后执行 doctor。已经完成当前配置时不用重装。上游 CKA-RL 为 GPLv3；后续发布修改代码要保留许可证和来源。
