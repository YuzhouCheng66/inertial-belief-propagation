# Inertial Belief Propagation

这里归档的是原始**每 8 次 GBP mean sweep＋一次局部块修正**版本：**每组独立自适应 β，β≤0.995，无滤波**。代码整理成独立 Python 包，包含 smallGrid3D、FR079、Sphere、Cubicle 的数据、冻结 k0 模型和完整原始曲线，以及 FR079/Cubicle 的多次线性化结果。

组大小为 32；SE2 保留 4 个、SE3 保留 12 个组内物理低频方向。局部修正为平衡块预条件作用于残差，再叠加组内低频惯性，通过入边的 canonical η 注入。各组的 β 和重启时钟独立更新，没有统一 β、组间滤波或全局粗层。

![四数据集 GBP sweep 对比](results/figures/frozen_residuals.png)

冻结线性化的首次达标 sweep 比：smallGrid3D 为 65.72 倍、FR079 为 14.51 倍；Sphere 的纯 GBP 在 400,000 sweep 上限仍未达标，局部版本在 14,424 sweep 达标。Cubicle 两者均未达标。FR079 的长期曲线仍有震荡，Cubicle 在同 sweep 预算下残差可能更差，所以这些数值不能解读为稳定的通用 10 倍提速。

多次线性化每次做 **20 个完整 cycle**，包括第 20 次修正，再进入新切空间。precision、canonical η 及对应 covariance 的坐标解释都按仿射右 Lie 图做 warping；新线性化重新认证 precision、重建局部基，并清空惯性历史。

| 20 次线性化后的 cost | 纯 GBP | 原始局部惯性 GBP |
|---|---:|---:|
| FR079 | 18.8432240390 | 18.8155601636 |
| Cubicle | 69,273.3951762 | 59,848.1871313 |

两者都做 3,200 次 mean sweep，惯性版本另做 400 次局部修正。原始记录中的完整运行时间，惯性版本在这两个算例上均更长；precision 准备是主要成本。本仓库的“完全局部”指**内部加速方向和各组 β**；外层公共 Armijo 使用全局 cost 和内积，诊断范数也有归约。

另外三个实验的代码、数据和结果位于 [ibp_more_experiments](ibp_more_experiments/README.md)：

- **Gaussian/PDE**：128×128 扩散线性系统，冻结 precision，以局部物理残差和时间惯性修正 information messages。完整 BP 映射调用中位减少 **6.84 倍**，同为 4 线程的实测提速中位数 **6.41 倍**。
- **非高斯 Ising**：128×128 模型，使用精确非线性 cavity 更新，直接加速标量消息。映射调用中位减少 **10.03 倍**，4 线程提速 **9.68 倍**。
- **复数张量网络**：32×32 网络，消息为 2×2 Hermitian 正定矩阵，在无迹 matrix-log 坐标中修正并映回正定消息。映射调用中位减少 **5.57 倍**，4 线程提速 **4.84 倍**。

这三个实验采用组内独立 β≤0.95、局部增益 0.45、每 8 次 BP 一次修正。统计来自 seeds 201/202/203，取逐 seed 比值的中位数；图中横轴包含额外残差 proposal 的完整映射调用，区别于上面 SE2/SE3 图的 mean sweep 数量。

![三个额外实验的 BP 与 IBP 对比](ibp_more_experiments/figures/bp_vs_ibp_2x2.png)

安装、运行和验证命令见 [English README](README.md)。数学定义见 [algorithm.md](docs/algorithm.md)，结果和局限见 [results.md](docs/results.md)。Cubicle 的 information eigenvalue floor=1、Huber=5、固定根节点和 Lie-log 残差都明确记录；cost 不是 g2o 的最优值，也不是带真值的相对位姿误差。
