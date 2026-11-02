# 持续学习研究记录

当前主线：持续强化学习中的知识组合、压缩与复用，优先考察 CKA-RL。

当前起点是 **CPU 上的 SAC 单任务练习**：运行 Pendulum-v1，熟悉训练、评估、保存和加载。机器人任务和短任务序列仍待推进。

## 快速开始（Windows / PowerShell）

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

各文档内的本机运行步骤从仓库根目录开始；若已进入 CRL起步，不需要重复进入。学校 Meta-World 示例则从仓库根目录进入 CRL起步/CKA-RL/experiments/meta-world。文档中的设备描述是当时实测条件。

## CKA-RL 上游代码

官方仓库：[Fhujinwu/CKA-RL](https://github.com/Fhujinwu/CKA-RL)。锁定 commit：`aa8817ace9f53120d5d205b9e4cafb3cdaa47803`。

上游代码单独获取。在仓库根目录运行：

```powershell
& ./CRL起步/setup_upstream.ps1
```

此脚本只下载官方代码并切换到指定版本，不安装 Meta-World、CUDA 或执行训练。原作者代码继续按上游许可使用。Meta-World 环境尚待在学校设备验证；不要在 Windows 上原样安装上游包含 Linux 专用依赖的完整 requirements。

## 当前验证

本机 CPU 的 10000 步示例完成 9500 次梯度更新，训练约 108 秒；独立10局平均回报由随机策略的约 -1223 提高到约 -231。模型保存重载检查和 GIF 渲染通过。该结果只有一个随机种子，仅作为配置与学习验证，不是 CKA-RL 论文复现。

仓库保存代码、说明和依赖清单；环境、训练产物、第三方论文 PDF 和本机凭据不纳入版本管理。
