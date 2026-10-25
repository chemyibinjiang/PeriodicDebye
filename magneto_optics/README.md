# 磁光超结构：有限纳米颗粒阵列

这个模块把仓库扩展到**电磁磁光响应**：给定颗粒位置、半径、介电张量和磁化方向，
计算有限阵列的消光、吸收、散射、磁圆二色性和散射光的偏振。
原有 X 射线 Debye 程序保留在 `../mof_scattering_handoff_20261013/`。

当前版本为 **0.1.0**。默认示例使用明确标注的合成材料；它们用于检查方法和运行流程，
不是金、铁、钴或磁性石榴石的实验数据。

## 先说明我们在解什么

每个均匀球形颗粒用一个三分量的**电偶极子**表示。光照在一个颗粒上，颗粒产生的场又驱动
其他颗粒，因此必须一起求解所有颗粒的响应。磁光性质由介电张量的非对角项给出。

这与原有的 X 射线 Debye 强度相加不同：这里有偏振、复振幅以及颗粒之间反复发生的电磁耦合。
加速计算仍求解同一个有限偶极系统。

支持有限二维阵列、三维阵列、稀疏缺陷和不同颗粒半径/材料。任意坐标可用稠密参考算法；
FFT 要求颗粒中心确实位于正交网格上，允许不同轴的间距不同，不会把离网格坐标偷偷吸附到网格。

**适用范围：**均匀、无损宿主中的小颗粒电偶极近似。当前没有基底、无限周期衍射端口、
磁化动力学或单颗粒内部的高阶多极模式。高折射率大颗粒的磁偶极共振需要体离散 DDA、
多极方法或其他经验证的 Maxwell 求解器。有限物体的前向散射偏振也不能直接当作整块超表面的
透射率、Faraday 角或 Kerr 角。

## 数学模型

采用时间约定 $e^{-i\omega t}$，所有长度使用 nm。输入波长 $\lambda_0$ 为真空波长，
宿主折射率 $n_h$ 为正实数：

$$
\epsilon_h=n_h^2,\qquad k=2\pi n_h/\lambda_0.
$$

### 1. 磁化怎样改变材料

令 $C(\mathbf m)\mathbf v=\mathbf m\times\mathbf v$。材料模型为

$$
\boldsymbol\epsilon=\epsilon_d\mathbf I-i g C(\mathbf m).
$$

$\mathbf m$ 是模长不超过 1 的无量纲磁化矢量；磁化反转就是 $\mathbf m\to-\mathbf m$。
当 $\mathbf m=\hat{\mathbf z}$ 时：

$$
\boldsymbol\epsilon=
\begin{pmatrix}\epsilon_d&i g&0\\-i g&\epsilon_d&0\\0&0&\epsilon_d\end{pmatrix}.
$$

这里“虚数非对角项”不等于“损耗”。被动性要求整个 Hermitian 损耗矩阵半正定：

$$
\frac{\boldsymbol\epsilon-\boldsymbol\epsilon^\dagger}{2i}\succeq0.
$$

程序对材料表的每个节点检查这个条件。表内线性插值，不允许外推。
Python API 也接受一般被动的 $3\times3$ 介电张量。

### 2. 单颗粒的响应

球半径为 $a$、体积为 $V=4\pi a^3/3$。使用体积单位的极化率：

$$
\boldsymbol\alpha_0=3V(\boldsymbol\epsilon-\epsilon_h\mathbf I)
(\boldsymbol\epsilon+2\epsilon_h\mathbf I)^{-1},
$$

$$
\boldsymbol\alpha=
\left(\mathbf I-\frac{i k^3}{6\pi}\boldsymbol\alpha_0\right)^{-1}
\boldsymbol\alpha_0.
$$

第二式包含辐射反作用，使无吸收颗粒散射出去的能量得到正确记账。
代码用等价的合并矩阵求解，处理零对比度和静电共振，避免先形成发散的 $\alpha_0$。
物理偶极矩为 $\mathbf p=\epsilon_0\epsilon_h E_0\mathbf u$，其中归一化偶极量 $\mathbf u$
和 $\alpha$ 的长度单位都是 $\mathrm{nm}^3$，入射电场以 $E_0=1$ 归一化。

### 3. 颗粒之间怎样耦合

令 $\hat{\mathbf r}=\mathbf r/r$，离开源的电场传播张量为

$$
\mathbf G(\mathbf r)=\frac{e^{ikr}}{4\pi r^3}
\left[(kr)^2(\mathbf I-\hat{\mathbf r}\hat{\mathbf r}^{T})
+(1-ikr)(3\hat{\mathbf r}\hat{\mathbf r}^{T}-\mathbf I)\right].
$$

每个颗粒感受到入射场和其他颗粒的场：

$$
\mathbf u_i=\boldsymbol\alpha_i\left[\mathbf E_{{\rm inc},i}
+\sum_{j\ne i}\mathbf G(\mathbf r_i-\mathbf r_j)\mathbf u_j\right].
$$

合起来就是线性系统 $(\mathbf I-\boldsymbol\alpha\mathbf G)\mathbf u=
\boldsymbol\alpha\mathbf E_{\rm inc}$。自身 Green 项设为零；辐射修正已经放入 $\alpha$，
不能再加一次。

### 4. 输出怎样定义

设 $\mathbf E_{{\rm loc},i}=\mathbf E_{{\rm inc},i}+\sum_{j\ne i}\mathbf G_{ij}\mathbf u_j$。
在单位入射幅度下，截面单位为 $\mathrm{nm}^2$：

$$
C_{\rm ext}=k\operatorname{Im}\sum_i\mathbf E_{{\rm inc},i}^{\dagger}\mathbf u_i,
$$

$$
C_{\rm abs}=k\sum_i\left[\operatorname{Im}
(\mathbf E_{{\rm loc},i}^{\dagger}\mathbf u_i)
-\frac{k^3}{6\pi}|\mathbf u_i|^2\right],\qquad
C_{\rm sca}=C_{\rm ext}-C_{\rm abs}.
$$

独立的远场角积分用于验收能量守恒，而不是只检查最后这个相减恒等式：

$$
\frac{dC_{\rm sca}}{d\Omega}=\frac{k^4}{16\pi^2}
\left|(\mathbf I-\hat{\mathbf n}\hat{\mathbf n}^T)
\sum_j\mathbf u_j e^{-ik\hat{\mathbf n}\cdot\mathbf r_j}\right|^2.
$$

圆偏振用明确的矢量定义，避免不同文献对“左/右旋”的相反命名：

$$
\mathbf e_\pm=(\mathbf e_1\pm i\mathbf e_2)/\sqrt2,
\qquad \mathbf e_1\times\mathbf e_2=\hat{\mathbf k}.
$$

每个波长求解四个通道：$(+\mathbf M,+)$、$(+\mathbf M,-)$、$(-\mathbf M,+)$、$(-\mathbf M,-)$。
磁化奇对称圆二色信号为

$$
\Delta C_{\rm MO}=\frac12\{[C_+(+\mathbf M)-C_-(+\mathbf M)]
-[C_+(-\mathbf M)-C_-(-\mathbf M)]\}.
$$

这样可以把磁化偶对称的天然结构圆二色性分开报告。归一化分母是四个消光截面的平均值；
分母为零时返回空值。线偏振入射的前向散射椭圆由两个圆偏振解线性叠加得到；
零散射时偏振未定义，圆偏振时长轴方位角未定义，均不填造一个旋转角。

## 运行

在这个目录创建独立 Python 环境（Python 3.10+）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test,report]"
.\.venv\Scripts\python.exe scripts/run_magneto.py --config examples/synthetic/finite_array.json --out runs/my_array
```

本仓库已有的手工验证环境也可以运行，无需重新安装模块：

```powershell
..\mof_scattering_handoff_20261013\.venv\Scripts\python.exe scripts/run_magneto.py --config examples/synthetic/finite_array.json --out runs/my_array
```

请使用新的输出目录。结果包括 `spectrum.csv`、`geometry.npz`、完整配置以及 `metadata.json`。
元数据记录实际算法、残差、迭代次数、单位、材料来源、环境版本和输入/源码哈希。

| 示例配置 | 结构 | 波长点数 |
|---|---|---:|
| [single_sphere.json](examples/synthetic/single_sphere.json) | 半径 6 nm 的单球 | 61 |
| [finite_array.json](examples/synthetic/finite_array.json) | 12×12 个球，36 nm 间距，外包络 408×408×12 nm | 61 |
| [volume_array.json](examples/synthetic/volume_array.json) | 8×8×8 个球的有限三维结构 | 5 |
| [large_array.json](examples/synthetic/large_array.json) | 128×128 个球，外包络 4.584×4.584 μm，厚 12 nm | 61 |

每个波长点包含四个磁化/偏振通道。波长范围是 400–700 nm；61 点为示范网格，
需要更精确的共振峰位或峰高时应加密波长并验证材料表的插值。

材料输入提供常量或按波长的 `epsilon_diagonal`、`g` 表，复数写成 `[实部, 虚部]`。
`site_materials` 可以给每个位置分配不同材料及磁化方向。球半径可用一个数或逐颗粒列表。
JSON 中的未知字段、重复键、非有限值、不被动介电张量及重叠/相切颗粒会被拒绝。

## 加速做了什么

`dense` 构造全部相互作用并做 LU 分解，是小结构的参考。`dense_gmres` 保留同样的稠密矩阵，
但用迭代求解，用来区分迭代方法和 FFT 各自带来的收益。

`fft` 用**线性零填充卷积**计算 $G\mathbf u$，不会把有限阵列的两边接起来。利用空间 Green
张量的对称性只保存六个傅里叶分量。单次矩阵乘法从 $O(N^2)$ 降为 $O(P\log P)$，
$P$ 是填充网格的点数；稀疏但包围盒很大的结构不一定划算。

同一波长的四个通道复用 Green 核，每个磁化方向的稠密解复用 LU。迭代法使用单位对角块的
等价方程，显式检查最终残差；不收敛就报错。内存检查包含 FFT 工作区和 Krylov 基，
报告中的数组估算与实测进程 RSS 分开列出。当前实现是 CPU complex128。

## 怎样验收

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/validate_magneto.py --out runs/my_physics
.\.venv\Scripts\python.exe scripts/benchmark_magneto.py --out runs/my_performance --threads 1
```

结构验收包括数量、外包络尺寸、最近中心间距、不同半径下的最小表面间距、网格一致性和有限边界。
物理验收包括单球圆偏振本征解、双颗粒纵/横向闭式解、零磁光项、磁化反转、Onsager
源交换关系、独立角积分能量守恒及完整 Mie 球对照。

**数值精度与物理近似分别验收。** 0.1% 是本次偶极模型求解的初始数值目标，不是任意结构的
Maxwell 物理误差保证。Mie 检查中，$ka=0.02$ 的消光近似误差约 0.0224%，
而 $ka=0.08$ 时约 0.357%，即使线性求解残差接近机器精度也不会消失。
尺寸筛查指标只是提示，不能代替多极/网格收敛验证。

完整验收、实际速度、误差和瓶颈见 [VALIDATION_REPORT.md](VALIDATION_REPORT.md)。

## 模型依据与下一阶段

磁光张量和各向异性偶极框架可参阅
[de Sousa 等，Scientific Reports 6, 30803](https://www.nature.com/articles/srep30803)。
电磁 DDA 的背景与边界条件见
[DDSCAT 指南](https://arxiv.org/abs/1305.6497)。
本模块自行实现有限颗粒偶极求解，并未复现该文中的大颗粒内部多极共振。

如果目标是无限周期超表面，下一阶段需要周期 Green 函数、衍射级次和能流归一化，
才能定义透射/反射与 Faraday/Kerr 角；相应理论可参阅
[Draine–Flatau 周期目标方法](https://arxiv.org/abs/0809.0338)。
真实材料预测还需要用户提供有来源的色散/磁光数据、宿主、尺寸及磁化条件。
