# 周期磁光超表面：从一个小单元到透射、反射和旋转

版本 **0.2.0**，日期 **2026-10-26**。本教程对应 `scripts/run_periodic.py`。
示例材料是合成的被动磁光张量，方便核对公式；不是任何真实材料的实验预测。

## 1. 我们重复的是什么？

想象一张无限大的方格纸。我们只画一个格子里的几个小球，其余格子重复这幅图。
光照到它们以后，每个小球既受到入射光，也受到所有其他格子里小球发出的光。
周期边界让我们只求一个格子里的未知量，同时保留无限多副本的电磁相互作用。

这个格子叫**单元**。矩形边长是 $L_x,L_y$，面积 $A=L_xL_y$。
单元中有 $N$ 个球，中心 $\mathbf r_j=(x_j,y_j,0)$、半径 $a_j$。
第 $(m,n)$ 个副本的中心为

$$
\mathbf r_{jmn}=\mathbf r_j+\mathbf R_{mn},\qquad
\mathbf R_{mn}=(mL_x,nL_y,0),\quad m,n\in\mathbb Z.
$$

这里 $N$ 是**每个单元的球数**，不是整片超表面的总球数。整个模型是无限周期的。
例子里的 36 nm 周期也不是 36 nm 宽的有限样品。
边界两侧的球也可能碰撞，所以几何检查包含相邻单元和球自己的周期副本。

首版范围：正入射沿 $+z$，球心严格在 $z=0$，矩形周期，上下两侧折射率相同且无损。
支持不同球半径、不同材料及磁化方向。每个球用一个电偶极子表示。
基底、斜入射、多层结构、非球形颗粒内部模式、磁偶极与高阶多极尚未包含。

## 2. 小球怎样感受到磁性？

采用时间约定 $e^{-i\omega t}$。长度是 nm，输入的 $\lambda_0$ 是真空波长。
宿主折射率 $n_h$ 为正实数：

$$
\epsilon_h=n_h^2,\qquad k=2\pi n_h/\lambda_0.
$$

磁化矢量 $\mathbf m$ 通过介电张量改变小球的响应：

$$
\boldsymbol\epsilon=\epsilon_d\mathbf I-i g C(\mathbf m),
\qquad C(\mathbf m)\mathbf v=\mathbf m\times\mathbf v.
$$

当 $\mathbf m=(0,0,1)$ 时：

$$
\boldsymbol\epsilon=
\begin{pmatrix}\epsilon_d&i g&0\\-i g&\epsilon_d&0\\0&0&\epsilon_d\end{pmatrix}.
$$

把光分成 $\mathbf e_\pm=(\hat{\mathbf x}\pm i\hat{\mathbf y})/\sqrt2$ 两种圆偏振，
它们看到的介电常数分别是 $\epsilon_\pm=\epsilon_d\mp g$。
两种光的振幅和相位变化不同，合起来就可能让线偏振转动、变成椭圆。
对这个沿 $z$ 磁化的材料张量，磁化反转会交换这两个材料本征通道。
一般多球单元还可能有几何各向异性，不能据此认定整片结构的两个通道总能直接交换。

被动材料不能凭空供能。代码检查整个矩阵

$$
\frac{\boldsymbol\epsilon-\boldsymbol\epsilon^\dagger}{2i}\succeq0,
$$

而不是只检查矩阵各元素的虚部。材料表的每个节点都检查；表内线性插值，禁止外推。
合成 Drude 示例中的两圆通道为

$$
\epsilon_\pm(E)=5-\frac{7^2}{E(E+0.15i\mp0.025)},
\quad E=1239.841984/\lambda_0\;\mathrm{eV},
\quad\epsilon_d=\frac{\epsilon_++\epsilon_-}{2},\quad
g=\frac{\epsilon_--\epsilon_+}{2}.
$$

极化率把局部电场变成偶极响应，单位是 $\mathrm{nm}^3$：

$$
\boldsymbol\alpha_0=4\pi a^3(\boldsymbol\epsilon-\epsilon_h\mathbf I)
(\boldsymbol\epsilon+2\epsilon_h\mathbf I)^{-1},
\qquad
\boldsymbol\alpha=
\left(\mathbf I-\frac{ik^3}{6\pi}\boldsymbol\alpha_0\right)^{-1}\boldsymbol\alpha_0.
$$

第二式是辐射反作用修正。用归一化偶极量 $\mathbf u=\mathbf p/(\epsilon_0\epsilon_h E_0)$，
入射幅度取 $E_0=1$。代码使用稳定的合并矩阵求解，包括零对比度和静电共振情况。

## 3. 怎样把无限多格子的光加起来？

自由空间中的传播张量为

$$
\mathbf G(\mathbf r)=\frac{e^{ikr}}{4\pi r^3}
\left[(kr)^2(\mathbf I-\hat{\mathbf r}\hat{\mathbf r}^{T})
+(1-ikr)(3\hat{\mathbf r}\hat{\mathbf r}^{T}-\mathbf I)\right].
$$

周期传播张量把所有副本都加进去：

$$
\mathbf S_{ij}=\sum_{m,n\in\mathbb Z}'
\mathbf G(\mathbf r_i-\mathbf r_j-\mathbf R_{mn}).
$$

撇号只去掉 $i=j,m=n=0$ 这一项。**$\mathbf S_{ii}$ 通常不为零**：同一个球在其他格子的
副本仍然会影响它。孤立球自己的辐射修正已经在 $\alpha$ 里，不能再算一次。

直接截断一个有限圆盘会留下缓慢振荡的尾巴。程序用 **Ewald 分拆**，把晶格和分成快速收敛的
实空间和倒空间两部分，再解析去掉自身奇点。这计算的是无限周期问题。
分拆参数和两部分截断都在验收中改变，以检查答案是否稳定。

最后解的仍是有限维线性方程：

$$
\mathbf u_i=\boldsymbol\alpha_i\left(\mathbf E_{{\rm inc},i}
+\sum_j\mathbf S_{ij}\mathbf u_j\right),\qquad
(\mathbf I-\boldsymbol\alpha\mathbf S)\mathbf u=\boldsymbol\alpha\mathbf E_{\rm inc}.
$$

每个波长求两种磁化方向，每种方向求 $x,y$ 两个入射偏振。
这些解给出完整 Jones 矩阵；圆偏振解可以线性组合得到，不必重新分解矩阵。

<details>
<summary>可选数学：Ewald 分拆在代码中的明确公式</summary>

取正实分拆参数 $\eta$，定义实空间短程标量核

$$
f(r)=\frac{e^{ikr}\operatorname{erfc}(\eta r+ik/2\eta)
+e^{-ikr}\operatorname{erfc}(\eta r-ik/2\eta)}{8\pi r}.
$$

它的张量形式由解析导数得到：

$$
\mathbf D_s(\mathbf r)=\left(k^2f+\frac{f'}r\right)\mathbf I
+\left(f''-\frac{f'}r\right)\hat{\mathbf r}\hat{\mathbf r}^{T}.
$$

对每个倒格矢 $\mathbf K=(2\pi m/L_x,2\pi n/L_y)$，取
$w=\sqrt{k^2-|\mathbf K|^2}$ 的出射支（实部、虚部均非负），定义

$$
H_{\mathbf K}=\frac{i}{2Aw}\operatorname{erfc}\left(-\frac{iw}{2\eta}\right),
\quad D_{\ell,ab}=(k^2\delta_{ab}-K_aK_b)H_{\mathbf K},\quad a,b\in\{x,y\},
$$

$$
D_{\ell,zz}=|\mathbf K|^2H_{\mathbf K}
-\frac{\eta}{A\sqrt\pi}e^{w^2/(4\eta^2)},\qquad
D_{\ell,xz}=D_{\ell,yz}=0.
$$

共面正入射的最终核为

$$
\mathbf S_{ij}=\sum_{\mathbf R}'\mathbf D_s(\mathbf r_i-\mathbf r_j-\mathbf R)
+\sum_{\mathbf K}e^{i\mathbf K\cdot(\boldsymbol\rho_i-\boldsymbol\rho_j)}\mathbf D_\ell(\mathbf K)
+\delta_{ij}c_{\rm self}\mathbf I,
$$

$$
c_{\rm self}=\frac{k^3\operatorname{erfi}(k/2\eta)
+2\eta(\eta^2-k^2)e^{k^2/(4\eta^2)}/\sqrt\pi}{6\pi}
-\frac{ik^3}{6\pi}.
$$

这里的最后一项去掉被分拆带入的同一偶极自身场。实空间导数直接解析求值，没有数值微分。
每次计算比较实、倒空间方形截断 $C$ 与 $C+2$；改变 $\eta$ 的检查在独立验收中进行。
报告的截断变化用 $\max(\max|S|,L_{\min}^{-3}+k^2/L_{\min})$ 归一化，
是实测收敛诊断，不能视为严格误差区间或每个微小矩阵元的相对误差界。

</details>

## 4. 出射光往哪里走？

周期结构像光栅，只允许特定的衍射方向。每个整数对 $(m,n)$ 对应

$$
\mathbf G_{mn}=\left(\frac{2\pi m}{L_x},\frac{2\pi n}{L_y}\right),\qquad
k_{z,mn}=\sqrt{k^2-|\mathbf G_{mn}|^2}.
$$

当 $|\mathbf G_{mn}|<k$ 时，$k_z$ 为正实数，这一级可以把能量带到远处。
当 $|\mathbf G_{mn}|>k$ 时它是倏逝波，参与近场耦合，但不计入远场透射/反射功率。
传播方向为 $\hat{\mathbf n}_{mn}^{\pm}=(G_x,G_y,\pm k_z)/k$。

每一级的散射电场为

$$
\mathbf E_{{\rm sc},mn}^{\pm}=
\frac{ik^2}{2Ak_{z,mn}}
\left(\mathbf I-\hat{\mathbf n}_{mn}^{\pm}\hat{\mathbf n}_{mn}^{\pm T}\right)
\sum_j\mathbf u_j e^{-i\mathbf G_{mn}\cdot\boldsymbol\rho_j}.
$$

透射方向还要加上原本就向前走的入射光，但只加在零级：

$$
\mathbf E_{{\rm T},mn}=\delta_{m0}\delta_{n0}\mathbf E_{\rm inc}
+\mathbf E_{{\rm sc},mn}^{+},\qquad
\mathbf E_{{\rm R},mn}=\mathbf E_{{\rm sc},mn}^{-}.
$$

这一步区别于有限阵列的“前向散射偏振”。这里的 Faraday 角来自**总透射场**。
功率需要乘上法向能流因子：

$$
T_{mn}=\frac{k_{z,mn}}{k}\frac{|\mathbf E_{{\rm T},mn}|^2}{|\mathbf E_{\rm inc}|^2},
\qquad
R_{mn}=\frac{k_{z,mn}}{k}\frac{|\mathbf E_{{\rm R},mn}|^2}{|\mathbf E_{\rm inc}|^2},
\qquad T=\sum_{\rm open}T_{mn},\quad R=\sum_{\rm open}R_{mn}.
$$

只要存在非零衍射级，$T_{00}+R_{00}$ 一般就不等于总出射功率。
程序保存每一级的复电场和功率。
精确或很接近 $k_z=0$ 的 Rayleigh 阈值会被明确拒绝：当前采用
$|1-|\mathbf G|^2/k^2|\le10^{-8}$ 的排除范围。代码不把这个奇异级次静默丢掉。

## 5. 吸收必须独立算一次

局部场为 $\mathbf E_{{\rm loc},i}=\mathbf E_{{\rm inc},i}+\sum_j\mathbf S_{ij}\mathbf u_j$。
吸收率由局部材料耗散计算：

$$
\mathcal A=\frac{k}{A|\mathbf E_{\rm inc}|^2}
\sum_i\left[\operatorname{Im}(\mathbf E_{{\rm loc},i}^{\dagger}\mathbf u_i)
-\frac{k^3}{6\pi}|\mathbf u_i|^2\right].
$$

再检查

$$
T+R+\mathcal A=1.
$$

如果直接定义 $\mathcal A=1-T-R$，这条等式永远成立，便失去了检查错误的能力。
这里两边来自独立计算。小的负吸收或能量残差保留原值，不通过截断数值来伪造守恒。

## 6. 怎样得到 Faraday 和 Kerr 角？

Jones 矩阵的列分别对应入射 $x,y$，行分别对应出射的实验室 $x,y$：

$$
\begin{pmatrix}E_{{\rm T},x}\\E_{{\rm T},y}\end{pmatrix}
=\mathbf J_T\begin{pmatrix}E_{{\rm inc},x}\\E_{{\rm inc},y}\end{pmatrix},\qquad
\begin{pmatrix}E_{{\rm R},x}\\E_{{\rm R},y}\end{pmatrix}
=\mathbf J_R\begin{pmatrix}E_{{\rm inc},x}\\E_{{\rm inc},y}\end{pmatrix}.
$$

对 $x$ 线偏振入射，取对应矩阵第一列 $(E_x,E_y)$，定义

$$
S_0=|E_x|^2+|E_y|^2,\quad S_1=|E_x|^2-|E_y|^2,
\quad S_2=2\operatorname{Re}(E_x^*E_y),\quad S_3=2\operatorname{Im}(E_x^*E_y),
$$

$$
\theta=\tfrac12\operatorname{atan2}(S_2,S_1),\qquad
\chi=\tfrac12\arcsin(S_3/S_0).
$$

$\theta$ 是椭圆长轴方位角，$\chi$ 是椭圆率角。输出的 `faraday_*` 和 `kerr_*` 分别记录
透射和反射的原始偏振角。非磁性几何各向异性也能产生旋转；要研究磁光贡献，应同时看
磁化反转后的结果与 `magnetic_odd_rotation_deg`，而不能把全部原始旋转都归因于磁性。
它们是零级光的偏振参数，不是把不同出射方向混合后的单一偏振。
零电场没有可定义的偏振；纯圆偏振没有长轴方位角，程序返回空值。
弱反射时即使 Kerr 角很大，也必须同时看反射率，不能把一个几乎没有光的角度当成强响应。

**符号约定：**反射光也使用固定实验室 $(x,y)$。它与反射传播方向 $-z$ 不构成右手系。
如果对照文献采用随传播方向改变的基底，需先转换基底；不能直接比较 Kerr 角的正负。
方位角模 $180^\circ$，输出的磁化奇对称角先对角差作 $180^\circ$ 包裹再除以二，故模 $90^\circ$。
当两种磁化的轴方向正好相差 $90^\circ$ 时，两个半角分支等价，奇对称角返回空值。
该奇对称量不是自动展开的连续相位曲线。

在正方晶格、一个球、磁化沿 $z$ 的特殊情况下，还能手算：

$$
\beta_\pm=\frac{\alpha_\pm}{1-S_{xx}\alpha_\pm},\qquad
r_\pm=\frac{ik}{2A}\beta_\pm,\qquad t_\pm=1+r_\pm.
$$

例如 $x$ 入射的透射分量是 $E_x=(t_++t_-)/2$、$E_y=i(t_+-t_-)/2$。
这个简单通道解可直接核对一般矩阵求解。

## 7. 运行和输入

在 `magneto_optics` 目录中运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test,report]"
.\.venv\Scripts\python.exe scripts/run_periodic.py --config examples/periodic/square_sheet.json --out runs/my_periodic_sheet
```

已有仓库验证环境也能直接运行源代码：

```powershell
..\mof_scattering_handoff_20261013\.venv\Scripts\python.exe scripts/run_periodic.py --config examples/periodic/square_sheet.json --out runs/my_periodic_sheet
```

始终指定新的输出目录。安装后也可用 `magnetooptics-periodic --config ... --out ...`。

| 示例 | 单元与球 | 波长 |
|---|---|---|
| [square_sheet.json](examples/periodic/square_sheet.json) | 36×36 nm，每单元 1 个半径 6 nm 球 | 400–700 nm，61 点 |
| [two_site_sheet.json](examples/periodic/two_site_sheet.json) | 72×60 nm，每单元 2 个不同半径的球 | 400–700 nm，61 点 |
| [diffractive_sheet.json](examples/periodic/diffractive_sheet.json) | 650×570 nm，每单元 2 个球，包含多级衍射 | 5 个波长，避开阈值 |
| [lossless_sheet.json](examples/periodic/lossless_sheet.json) | 无损磁光张量，用于能量检查 | 4 个波长 |

JSON 格式为 `magnetooptics-periodic-1.0`。几何必填 `periods_nm`、`positions_nm`、`radius_nm`；
球心限定在 $[0,L_x)\times[0,L_y)$，$z=0$。半径可为标量或逐球列表。
用 `material` 指定统一材料，或用 `site_materials` 指定逐球材料名，两者不可并用。
材料格式与有限模型相同，复数使用 `[实部,虚部]`。
`incidence.direction` 只接受 `[0,0,1]`。不支持的字段会报错。

`solver.backend` 可选 `cached`、`vectorized` 或 `reference`，三者求解同一个周期偶极模型。
`reference` 逐对、逐晶格项累加，作为易检查的慢参考；`vectorized` 只向量化晶格求和；
`cached` 再复用严格相同的位移。性能报告把这两步加速分别比较。
`ewald_tolerance` 控制晶格求和，`rtol` 控制线性残差，`max_memory_mib` 限制估计数组工作区。
`threads` 指定 BLAS 线程数。

输出包括：

- `spectrum.csv`：两磁化方向下，$x,y,+,-$ 入射的总 T/R、零级 T/R、独立吸收、能量残差；
  Faraday/Kerr 方位角、椭圆率及磁化奇对称信号。
- `diffraction_orders.csv`：每个波长、磁化方向、偏振和传播级次的 T/R 与复电场。
- `metadata.json`：零级 Jones 矩阵、算法设置、求解残差、时间、材料来源、单位、约定和源码哈希。
- `geometry.npz` 与 `config.json`：实际单元几何和完整输入，配置哈希按保存后的实际字节计算。

## 8. 如何判断结果正确，以及加速了什么？

周期版本的验收不依赖有限模型的 FFT：其 Green 函数和远场端口是独立的新实现。
测试包括静电方形晶格的解析和、开放衍射级的辐射恒等式、单球圆通道闭式解、
磁化反转、无磁光项、原胞与重复超胞等价，以及损耗/无损多衍射级能量守恒。
还改变 Ewald 截断与分拆参数，比较缓存与逐对参考结果。

```powershell
.\.venv\Scripts\python.exe -m pytest -q -o junit_family=xunit1
.\.venv\Scripts\python.exe scripts/validate_periodic.py --out runs/periodic_physics
.\.venv\Scripts\python.exe scripts/benchmark_periodic.py --out runs/periodic_performance
```

加速利用周期几何中重复的位移复用 Green 张量，并复用同波长两磁化方向的相互作用矩阵、
同磁化方向两偏振的 LU 分解。不同波长重新计算传播与材料响应。
单元中所有两两块仍存为稠密矩阵，存储 $O(N^2)$，直接分解 $O(N^3)$；
这不是一个已经解决任意巨大复杂单元的 FFT 周期求解器。

**数值误差与物理模型误差分别验收。** 0.1% 的初始目标用于数值求解比较；
电偶极近似是否足够准确，还取决于球大小、间距和材料，需要多极/体离散收敛或独立 Maxwell 对照。
结构化测试里的高精度一致不能变成任意真实超表面达到 0.1% 物理误差的承诺。
61 点光谱也只是示范采样；尖锐共振和近阈值情况需单独加密验证。

共面纯电偶极模型还有一个物理限制：零级场满足 $\mathbf J_T=\mathbf I+\mathbf J_R$。
对单位幅度单侧入射，$T_{00}+R_{00}=\|\mathbf e+\mathbf r\|^2+\|\mathbf r\|^2\ge1/2$，
其他传播级功率非负，所以被动吸收率不超过 50%。这不是任意磁光超表面的限制，
引入磁响应、厚度或基底后需要重新推导。

实际性能、误差、测试数量和完整原始数据见 [PERIODIC_VALIDATION.md](PERIODIC_VALIDATION.md)。
历史有限阵列的测试与性能保留在 [VALIDATION_REPORT.md](VALIDATION_REPORT.md)，其源码哈希对应当时版本。

## 理论出处

周期偶极的 Bloch 条件、传播衍射级和能流定义可参考
[Draine 与 Flatau，JOSA A 25, 2693–2703 (2008)](https://arxiv.org/abs/0809.0338)。
二维 Helmholtz 周期 Green 函数的 Ewald 分拆参考
[Oroskar、Jackson 与 Wilton，J. Comput. Phys. 219, 899–911 (2006)](https://doi.org/10.1016/j.jcp.2006.06.050)。
本实现的张量分拆依据为
[Lunnemann、Šersić 与 Koenderink，Phys. Rev. B 88, 245109 (2013)，附录 A](https://arxiv.org/abs/1309.3641)，
并转换到本仓库的体积极化率单位和 $1/(4\pi)$ Green 函数归一化。
本实现保留每球单电偶极的模型边界；它不是这些方法中任意体离散结构的完整实现。
