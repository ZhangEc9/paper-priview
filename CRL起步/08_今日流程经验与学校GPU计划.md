# 2026-11-03：完整流程、经验与下一步

## 今天做到哪里

主线是持续强化学习中的知识组合、压缩和复用。今天从 Pendulum 两任务练习，推进到了官方 CKA-RL 的 Meta-World CPU 短序列。已验证原方法的程序流程，尚未复现论文性能，也没有接入新的合并算法。

对应说明和脚本均已进入仓库；实际结果在 [records/2026-11-03](records/2026-11-03/README.md)，不是示例数字。论文 PDF、第三方源码、Python 环境和模型权重不随本次提交上传，第三方源码通过固定提交获取。

## 完整流程与结果

| 阶段 | 实际做了什么 | 得到了什么 | 还不能说明什么 |
|---|---|---|---|
| 两任务练习 | SAC 先学 Pendulum A(g=10)，再学 B(g=5)，并比较 B 从零训练 | A 从约 -231 降到 -1027；B 约 -124，从零 B 约 -150 | 一个训练种子不能证明稳定增益或泛化 |
| 参数插值 | 混合 A 与 A→B 的完整 actor，扫描 α | α=0 对保留 A 最好，B 差异较小 | 这不是 CKA 知识向量合并实验 |
| 方法与选题讨论 | 核对 CKA 实现，讨论历史系数加权最小二乘目标 | 得到待验证的建模候选，数学和实现可分别支撑两份毕设 | 未完成查新、未接入方法、未得到回报保证 |
| 环境与源码 | 固定官方 CKA-RL、Meta-World 提交，配置隔离 CPU 环境 | MuJoCo 无画面运行，前四任务各完成 500 步环境检查 | 没有验证学校 GPU 环境 |
| 训练短实验 | simple 1 任务、CKA 4 任务，每任务 1,500 步、池容量 2 | 每阶段 critic 更新 1,299 次、actor 更新 1,298 次；保存重载误差 0 | 不代表任务已经学会 |
| 知识合并 | 第四阶段从 3 个旧候选压到 2 个 | 各参数块选对、取平均均核对通过；冻结旧向量变化为 0 | 尚不能判断压缩对长训练性能的影响 |

当前任务额外评估平均回报：simple/hammer 459.80，CKA 四任务依次为 459.80、16.80、1168.11、4.68；当前任务各评估 3 回合，末步和任一步成功均为 0。不同任务奖励不能直接互相比较。

## 为什么成功率都是 0

每任务 1,500 步只相当于 3 个 500 步回合，其中前 300 步随机探索，前 200 步尚未启动梯度更新。这是检查程序能否训练与合并的调试预算。

论文每任务 1,000,000 步，约为本次的 **667 倍**。在相同短预算下换 GPU，策略不会因为计算设备改变而自动学会任务；GPU首先改变的是耗时。

Meta-World 的奖励含有接近、抓取、推动等中间进展，成功则由任务自己的完成阈值定义。例如有一千多回报，也可能没有完成任务。另有每任务仅 3 次评估的采样限制，0/3 不能精确估计真实成功概率。

当前证据是「参数更新、有限数值、冻结和合并正常」，不能推导「长训练一定成功」。如果提高预算后独立 SAC 仍持续零成功，应检查奖励曲线、探索、动作分布、任务重置和评估实现；不要直接把原因归于知识合并，也不要靠反复挑随机种子解决。

## 排错经验与需要保留的区分

| 问题 / 发现 | 处理与结论 |
|---|---|
| 官方 requirements 含 Linux/CUDA 专用项 | 选用 Windows CPU 关键依赖并保存完整锁文件，不整包照装 |
| Windows 下旧 MuJoCo 无法读取中文 XML 路径 | 将资源缓存到英文路径，逐文件校验 SHA256，官方源码不变 |
| 初版启动器执行命名空间与类型注解冲突 | 为训练代码建立独立的 `__main__` 模块，编译时关闭继承启动器的 future 标志 |
| tyro 0.6.3 与 Python 3.11.15 argparse 不兼容 | 命令行解析库升级到 1.0.16；不改训练更新式 |
| 官方启动器使用 python3 并按空格拆参数 | 使用当前解释器 `sys.executable` 和参数列表 |
| 官方评估直接输出高斯均值 | 保留原日志，另用 tanh+缩放的确定性动作在独立环境评估；两套结果不混用 |
| eval_every 参数没有被周期调用使用 | 不声称已有周期性独立评估；已导出训练回合统计和末尾评估 |
| 知识池容量的默认值不一致 | 文中 8、训练入口 9、序列启动器 20；实验显式写清容量，本次用 2 提前触发合并 |
| 只剩一个旧向量时 alpha 不变 | softmax 对单元素恒为 1，是预期情况；两个向量后已观察到参数变化 |
| 不同参数块合并的配对不同 | 官方分别处理 mean/logstd 的 weight/bias，本次实测配对不同；保留原算法，尚未证明这会影响效果 |
| 入口没有直接写 action_space.seed | 已沿调用链核实 Meta-World 的 env.seed 会设置它，不能仅看入口就断定未设种子 |

两次早期启动失败日志也随记录保存。它们属于工具兼容性排错，不属于学习效果失败。

## 是否需要回学校用 GPU

**建议把正式学习与多种子实验放到学校设备；现在公司电脑仍可阅读、分析和做小预算验证。** 没有必要在学校环境还没测过之前先租卡。

Meta-World 物理模拟主要在 CPU，GPU负责网络计算。模型是低维输入的 256 宽 MLP、batch 128，GPU不一定带来很大的端到端加速；学校 CPU、内存和散热也会影响速度。用实测的总耗时决定是否扩预算。

本次每任务训练加原评估约 53–56 秒，但 warmup、训练步数和回放池都与正式配置不同，不能简单乘 667 就当作可靠 ETA。论文 Meta-World 有 20 个任务阶段、10 个种子；单个方法就是 2 亿环境步，独立参考及其他对照还要额外计算。

### 学校 GPU 环境准备（计划，尚未在学校实测）

建议将仓库克隆到英文路径，例如 `C:\research\paper-priview`，使用 Python 3.11 和独立环境 `.venv-cka-gpu`。RTX 4060 可先尝试 PyTorch 2.1.2 的 CUDA 12.1 wheel，前提是驱动支持；以安装后的可用性检查为准。

在仓库根目录执行以下环境准备命令：

```powershell
py -3.11 -m venv CRL起步/.venv-cka-gpu
$GpuPython = '.\CRL起步\.venv-cka-gpu\Scripts\python.exe'
& $GpuPython -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
$GpuRequirements = Join-Path $env:TEMP 'cka-requirements-without-torch.txt'
Get-Content .\CRL起步\requirements-cka-cpu-lock.txt | Where-Object { $_ -notmatch '^torch==' } | Set-Content $GpuRequirements -Encoding utf8
& $GpuPython -m pip install -r $GpuRequirements
& $GpuPython -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
& .\CRL起步\setup_upstream.ps1
```

不能将 CPU 锁文件中的 `torch==2.1.2+cpu` 直接装回 GPU 环境，否则会把设备配置弄混。其余锁文件来自 Windows，Linux 租卡时也不能照装所有 Windows 专用依赖。

Meta-World 仍需安装指定源码提交。首次在仓库根目录配置：

```powershell
New-Item -ItemType Directory -Path .\CRL起步\vendor\Metaworld -Force | Out-Null
git -C .\CRL起步\vendor\Metaworld init
git -C .\CRL起步\vendor\Metaworld remote add origin https://github.com/Farama-Foundation/Metaworld.git
git -C .\CRL起步\vendor\Metaworld fetch --depth 1 origin c822f28f582ba1ad49eb5dcf61016566f28003ba
git -C .\CRL起步\vendor\Metaworld checkout --detach FETCH_HEAD
& $GpuPython -m pip install --no-deps -e .\CRL起步\vendor\Metaworld
& $GpuPython .\CRL起步\cka_cpu.py doctor
```

这些 Git 初始化命令仅用于首次配置，不要在已有其他提交或改动的目录里反复执行。`doctor` 检查环境并报告 CUDA 是否可用，其环境模拟本身使用 CPU。

**当前 `cka_cpu.py` 是 CPU 专用启动器，训练参数固定 `--no-cuda`。即使 CUDA 可用，直接运行它也不会使用 GPU。** 学校的下一项代码工作是增加设备选项，并核对旧模型加载、知识池张量和合并相似度矩阵的设备；官方 CKA 类里有默认新建 CPU 张量的地方，不能只删除 `--no-cuda` 就声称 GPU 序列可用。此部分未实现、未验证，不计入今天完成项。

## 下一步实验顺序与继续条件

1. **先让独立 SAC 学会一个任务。** 先跑 20k 测速和看曲线，再按预先记录的预算推进到 100k，必要时继续到论文 1M。20k 全零仍不足以判无效。任务顺序和预算提前记录，不按成绩反复筛任务。
2. **CPU/GPU 做相同配置的吞吐比较。** 总步数、回放池、warmup、评估次数一致；分别记录训练和评估耗时。大预算才恢复论文回放池 1M，并检查内存。
3. **跑原版 CKA 四任务与逐任务独立 SAC 对照。** 同预算、同任务、同种子组；容量 2 用于观察实际合并。先一个种子排错，再用至少 3 个训练种子检查方向，注明仍未达到论文的 10 个种子。
4. **再扩大到容量 8 和原 20 阶段序列。** 四任务容量 8 不会触发合并，只能作为容量影响的局部参考，不能据此证明压缩效果。
5. **统一评估协议并计算论文指标。** 额外的旧任务复测只是诊断；PERF/FWT 按论文定义重建，FWT 需要独立训练参考曲线。原评估与修正动作映射的评估分列。
6. **最后评估新想法是否值得做。** 原余弦平均、按合并成员数加权、历史使用系数加权逐步比较。先完成查新、坐标对齐和存储核算；没有重复收益就停止具体候选。

近期可在公司 CPU 上直接执行的 20k 命令见 [07 的下一轮学习实验](07_CKA原方法CPU基线.md#6-下一轮学习实验)。不要求今天再跑一次相同的 1,500 步短实验。

阅读建议：先看本文件和实际结果，再读 [07](07_CKA原方法CPU基线.md) 的操作细节；需要回顾原理时再读 [04](04_CPU两任务练习.md)、[05](05_从遗忘到知识合并.md)、[06](06_CKA复现与使用感知合并候选.md)。
