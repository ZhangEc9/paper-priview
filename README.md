# 持续学习研究记录

当前主线：持续强化学习中的知识组合、压缩与复用，优先考察 CKA-RL。

截至 **2026-11-03**，已完成 Pendulum 单任务、两任务顺序学习与参数插值练习，并在公司 Windows CPU 上跑通官方 **CKA-RL + Meta-World** 的训练、知识复用、容量溢出合并和保存重载。

CKA 短实验每任务仅 1,500 步，当前任务评估成功率均为 0。它验证了程序流程，**尚未复现论文性能**。下一步先验证独立 SAC 的单任务学习效果，再做相同预算的 CKA 对照。

## 今天的流程与结果

- [完整流程、为什么成功率为 0、经验与学校 GPU 计划](CRL起步/08_今日流程经验与学校GPU计划.md)
- [CPU 配置、原方法短实验说明与下一轮命令](CRL起步/07_CKA原方法CPU基线.md)
- [真实结果与失败日志归档](CRL起步/records/2026-11-03/README.md)
- [CKA 短实验实际结果表](CRL起步/records/2026-11-03/cka_smoke/结果摘要.md)

## CKA-RL CPU 配置与运行（Windows / PowerShell）

安装 Python 3.11、Git 后，在仓库根目录运行：

```powershell
$Python311 = (& py -3.11 -c "import sys; print(sys.executable)")
& ./CRL起步/setup_cka_cpu.ps1 -PythonExe $Python311
& ./CRL起步/.venv-cka/Scripts/python.exe ./CRL起步/cka_cpu.py smoke --output ./CRL起步/runs/cka_smoke_first
& ./CRL起步/.venv-cka/Scripts/python.exe ./CRL起步/cka_summarize.py ./CRL起步/runs/cka_smoke_first
```

配置脚本获取固定提交的官方 CKA-RL 和 Meta-World，建立独立 CPU 环境，处理 Windows 中文资源路径，再做环境检查。smoke 执行 simple 1 任务和 CKA 4 任务；池容量为 2，使第四阶段实际触发合并。每次运行使用新的输出目录。

**`cka_cpu.py` 当前固定使用 CPU。** 学校 GPU 配置和待完成的设备适配见上方计划，不把 CUDA 安装成功当作 GPU 训练已经验证。

## 回顾 Pendulum 练习（Windows / PowerShell）

先安装 Python 3.11 和 Git。在你选定的本地目录运行：

```powershell
git clone https://github.com/ZhangEc9/paper-priview.git
Set-Location paper-priview
py -3.11 -m venv CRL起步/.venv
& ./CRL起步/.venv/Scripts/python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
& ./CRL起步/.venv/Scripts/python.exe -m pip install -r ./CRL起步/requirements-cpu.txt
& ./CRL起步/.venv/Scripts/python.exe ./CRL起步/single_task.py train --name my_first_10k --steps 10000
```

如果默认 PyPI 下载停滞，可在安装 requirements 的命令末尾添加 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple`。

结果写入 `CRL起步/runs/my_first_10k/`，包括摘要、学习曲线、评估 CSV 和模型。每次运行换一个名字，避免覆盖。

生成动作演示 GIF：

```powershell
& ./CRL起步/.venv/Scripts/python.exe ./CRL起步/single_task.py watch --model ./CRL起步/runs/my_first_10k/model.zip
```

## 阅读顺序

1. [从这里开始](CRL起步/00_从这里开始.md)：今天的命令、输出与自测。
2. [三件事步骤](CRL起步/01_三件事步骤.md)：单任务 → 短任务序列 → 成本测量。
3. [代码说明与 CKA 对接](CRL起步/02_代码说明与CKA对接.md)：函数阅读和上游实现。
4. [配置验证记录](CRL起步/03_配置验证记录.md)：2026-11-02 公司 CPU 实测。
5. [我的第一次运行](CRL起步/我的第一次运行.md)：填写自己的结果和问题。
6. [CPU 两任务练习](CRL起步/04_CPU两任务练习.md)：任务切换、遗忘、从零训练对照。
7. [从遗忘到知识合并](CRL起步/05_从遗忘到知识合并.md)：参数插值与观察边界。
8. [CKA 与使用感知合并候选](CRL起步/06_CKA复现与使用感知合并候选.md)：建模讨论，尚未接入算法或完成查新。
9. [原方法 CPU 基线](CRL起步/07_CKA原方法CPU基线.md)和[今日流程与下一步](CRL起步/08_今日流程经验与学校GPU计划.md)：最新进展。

各文档内的本机运行步骤从仓库根目录开始；若已进入 CRL起步，不需要重复进入。学校 Meta-World 示例则从仓库根目录进入 CRL起步/CKA-RL/experiments/meta-world。文档中的设备描述是当时实测条件。

## CKA-RL 上游代码

官方仓库：[Fhujinwu/CKA-RL](https://github.com/Fhujinwu/CKA-RL)。锁定 commit：`aa8817ace9f53120d5d205b9e4cafb3cdaa47803`。

上游代码单独获取。在仓库根目录运行：

```powershell
& ./CRL起步/setup_upstream.ps1
```

`setup_upstream.ps1` 只获取 CKA-RL 源码；前文的 `setup_cka_cpu.ps1` 另含 Meta-World、CPU 依赖安装与环境检查。Meta-World 已在公司 CPU 验证，学校 GPU 尚待实测。不要在 Windows 上原样安装上游包含 Linux 专用依赖的完整 requirements。CKA-RL 使用上游 GPLv3 许可，Meta-World 使用其上游许可。

## 当前验证

本机 CPU 的 10000 步示例完成 9500 次梯度更新，训练约 108 秒；独立10局平均回报由随机策略的约 -1223 提高到约 -231。模型保存重载检查和 GIF 渲染通过。该结果只有一个随机种子，仅作为配置与学习验证，不是 CKA-RL 论文复现。

2026-11-03 的 CKA 短实验中，旧向量数依次为 0/1/2/2；每阶段 critic/actor 更新 1299/1298 次，冻结旧向量没有变化，保存重载误差为 0。最终阶段实际完成四个参数块的余弦选对与算术平均，误差为 0。每任务约 53–56 秒（训练及原评估）；短预算全部零成功。

仓库保存代码、说明、依赖清单，以及明确挑选的文本/CSV/JSON 实验记录。环境、模型权重、回放数据、第三方源码、论文 PDF 和凭据不纳入版本管理。早期文档保留当时的学习步骤，最新状态以上方今日记录为准。
