# PeriodicDebye: X-ray scattering and magneto-optical arrays

**磁光超晶格（v0.4.0）：** 新增精确分层 Maxwell 求解器，以及 21 层 Bragg 微腔—镜面示例。
合成弱磁光层通过共振实现单程 45° 与指定镜面间隙下的 90° 返回，包含透射损失与吸收对照。
见 [材料说明、超晶格方程与运行方法](magneto_optics/SUPERLATTICE.md)。

**Faraday 镜面双程例子（v0.3.0）：** 已构造包含界面和多次反射的合成均匀膜，
实现线偏振单程 45°、经普通镜面折返后再转 45°，总计 90°。
见 [数学推导、偏振图与可调角度运行方法](magneto_optics/FARADAY_MIRROR.md)。

**周期磁光超表面（2026-10-26，v0.2.0）：** 新增正入射下二维周期颗粒阵列的透射、反射、
Faraday/Kerr 旋转和全部传播衍射级次。使用均匀宿主中的小球电偶极模型与合成材料，
从 [周期超表面数学教程](magneto_optics/PERIODIC_TUTORIAL.md) 和
[周期验收与性能报告](magneto_optics/PERIODIC_VALIDATION.md) 开始。

**磁光超结构扩展（2026-10-25）：** 新增有限纳米颗粒阵列的电磁耦合模块，支持磁光介电张量、
磁化反转、圆二色性及散射偏振，并提供独立物理验收和 FFT 加速。
从 [磁光模块中文教程](magneto_optics/README.md)、
[验收与性能报告](magneto_optics/VALIDATION_REPORT.md) 或
[本轮 goal](magneto_optics/GOAL.md) 开始。

The electromagnetic module is version **0.4.0**. It models finite arrays and
periodic sheets of small electric-dipole spheres, plus exact gyrotropic layer stacks and mirror cavities; it does not turn the X-ray Debye equation
into a Maxwell solver. The original X-ray tutorial follows below.

This project calculates how X-rays scatter from a crystal and from nanoparticles
inside its pores. It supports finite single crystals, ideal periodic single
crystals, and powder X-ray diffraction (PXRD). The current code is **v0.4.1**.

Start with the first four lessons: two atoms are enough to understand the main
idea. The later lessons build up to millions of atoms and randomly occupied
pores. Each equation is followed by what it means. The expandable sections are
optional extra mathematics.

**Reading route:** [waves](#1-atoms-send-out-waves) →
[Debye's equation](#2-a-powder-looks-in-every-direction) →
[two atoms](#3-try-it-with-two-atoms) →
[three modes](#4-three-ways-to-look-at-the-same-atoms) →
[particles in pores](#5-putting-a-nanoparticle-in-a-pore) →
[four contributions](#6-keep-all-four-kinds-of-atom-pairs) →
[fast calculation](#8-how-can-we-handle-a-100-nm-crystal) →
[run it](#10-run-the-examples-yourself).

## 1. Atoms send out waves

Imagine dropping two pebbles into water. Their ripples can reinforce or cancel
each other. Scattered X-rays also interfere. An atom's location changes the
phase of its wave: where it is in its repeating cycle.

We first **add the waves**, then calculate their intensity:

$$
A(\mathbf q)=\sum_{i=1}^{N} f_i(q)\,e^{\mathrm{i}\mathbf q\cdot\mathbf r_i},
\qquad I(\mathbf q)=|A(\mathbf q)|^2.
$$

Here is how to read those symbols:

| Symbol | Meaning |
|---|---|
| $N$ | Number of atoms |
| $i,j$ | Labels for atoms, like names on numbered beads |
| $\sum$ | Add all the listed terms |
| $\mathbf r_i$ | Position of atom $i$, in angstroms ($1\ \mathrm{nm}=10\ \text{Å}$) |
| $\mathbf q$ | Scattering vector: a direction and a size |
| $q=\lvert\mathbf q\rvert$ | Size of that vector, in inverse angstroms |
| $f_i(q)$ | How strongly atom $i$ scatters at this $q$ |
| $e^{\mathrm{i}\phi}=\cos\phi+\mathrm{i}\sin\phi$ | A compact way to describe a wave's phase; $\mathrm{i}^2=-1$ |
| $\lvert A\rvert^2$ | Square of the length of the total wave amplitude |

For this project's real atomic form factors, expanding the square gives

$$
I(\mathbf q)=\sum_i f_i(q)^2
+2\sum_{i<j}f_i(q)f_j(q)
\cos\!\left[\mathbf q\cdot(\mathbf r_i-\mathbf r_j)\right].
$$

The first sum is each atom's own contribution. The second sum describes pairs
of different atoms. The notation $i<j$ counts each pair once; the factor of two
accounts for the two cross terms in the squared amplitude. A pair contribution
can be negative because waves can cancel. The complete physical intensity is
nonnegative.

The connection to a diffraction angle is

$$
q=\frac{4\pi\sin\theta}{\lambda},
$$

where $\lambda$ is the X-ray wavelength and $2\theta$ is the angle between the
incoming and scattered beams. We use this convention throughout; $q$ already
includes the factor $2\pi$.

## 2. A powder looks in every direction

A single crystal has a particular orientation. An ideal random powder contains
many crystallites pointing in all directions. Its pattern is the average
intensity of a whole crystallite over those directions. Separate crystallites
are treated as adding intensities, without coherent interference between them.

Written as an equation, this whole-object average is

$$
I_{\rm powder}(q)=\frac{1}{4\pi}\int I(q\widehat{\mathbf n})\,d\Omega.
$$

$\widehat{\mathbf n}$ is a unit direction, and $d\Omega$ is a small patch of
directions on a sphere. Dividing by $4\pi$ makes the sum over the full sphere
an average.

Let $\mathbf r_{ij}=\mathbf r_i-\mathbf r_j$ join two atoms, and let
$r_{ij}=|\mathbf r_{ij}|$ be their distance. Averaging their directional
interference gives

$$
\left\langle\cos(\mathbf q\cdot\mathbf r_{ij})\right\rangle_{\rm directions}
=\frac12\int_{-1}^{1}\cos(qr_{ij}u)\,du
=\frac{\sin(qr_{ij})}{qr_{ij}}.
$$

Here $u$ is the cosine of the angle between the pair and the scattering vector.
The integral is a continuous average, like adding very many directions with
the correct weights. The angle brackets $\langle\cdot\rangle$ mean an average.

We give the last function a short name:

$$
\operatorname{sinc}(x)=
\begin{cases}
\sin(x)/x,&x\ne0,\\
1,&x=0.
\end{cases}
$$

Replacing the cosine by its average gives **Debye's scattering equation**:

$$
\boxed{
I_{\rm powder}(q)=\sum_i f_i(q)^2
+2\sum_{i<j}f_i(q)f_j(q)\operatorname{sinc}(qr_{ij})
}
$$

It says: **add the contribution from every atom and every atom pair, using
the distance of each pair**. It works for a finite crystal, a molecule, or an
irregular cluster. A repeating lattice is not required. This is the usual
random-powder Debye framework; see the
[IUCr explanation of the Debye equation](https://journals.iucr.org/a/issues/2016/06/00/me0628/index.html).

In NumPy, `np.sinc(x)` means $\sin(\pi x)/(\pi x)$. To evaluate our function,
write `np.sinc(q * r / np.pi)`.

## 3. Try it with two atoms

Put two identical atoms a distance $d$ apart. To make the arithmetic easy,
pretend both have constant scattering strength $f=1$. Real calculations use
element-dependent $f_i(q)$.

$$
\begin{aligned}
I_{\parallel}(q)&=2[1+\cos(qd)],\\
I_{\perp}(q)&=4,\\
I_{\rm powder}(q)&=2[1+\operatorname{sinc}(qd)].
\end{aligned}
$$

Parallel and perpendicular refer to the scattering vector relative to the line
joining the atoms. Perpendicular means the waves have the same phase. For a
parallel vector, their phase difference is $qd$.

| $qd$ | Parallel intensity | Perpendicular intensity | Powder intensity |
|---|---:|---:|---:|
| $0$ | 4 | 4 | 4 |
| $\pi$ | 0 | 4 | 2 |
| $2\pi$ | 4 | 4 | 2 |

At $qd=\pi$, the parallel waves cancel completely. The powder still scatters,
because its average includes many other directions. A powder average is not
the average of just the parallel and perpendicular cases.

Two useful checks follow immediately:

$$
I_{\text{one atom}}(q)=f(q)^2,
\qquad
I(0)=\left[\sum_i f_i(0)\right]^2.
$$

At $q=0$, every wave has the same phase. The last expression applies to one
fixed atom configuration. Random occupancy needs an additional average, which
we will calculate below. At very large pair separation times $q$, the sinc
terms tend to zero, leaving the self terms for a finite set of distinct atoms.

## 4. Three ways to look at the same atoms

| Mode | Question it answers | Mathematical operation |
|---|---|---|
| Finite single crystal | What does this finite object scatter in a chosen direction? | $I(\mathbf q)=\lvert A(\mathbf q)\rvert^2$ |
| Periodic single crystal | What happens when the crystal extends indefinitely? | Infinite-lattice limit, with Bragg weights and disorder diffuse scattering |
| PXRD | What does a random powder of these objects scatter? | Average intensity over directions; finite objects use Debye's equation |

These choices describe the measurement and boundary. Choices such as
`analytic`, `moments`, or `radial_moments` describe how the calculation is done.
They are not additional physical scattering modes.

### A finite crystal has edges

For identical complete cells, let $F(\mathbf q)$ be one cell's amplitude,
$\mathbf R_n$ the position of cell $n$, and $N_c$ the number of retained cells.
Then

$$
A(\mathbf q)=F(\mathbf q)S(\mathbf q),\qquad
S(\mathbf q)=\sum_{n=1}^{N_c}e^{\mathrm{i}\mathbf q\cdot\mathbf R_n},
\qquad I(\mathbf q)=|F(\mathbf q)|^2|S(\mathbf q)|^2.
$$

$S$ keeps track of the object's size and shape. For example, a straight chain
of $N$ identical atoms with spacing $d$ has

$$
I_{\parallel}(q)=f(q)^2
\left[\frac{\sin(Nqd/2)}{\sin(qd/2)}\right]^2,
$$

with the limit $N^2f(q)^2$ where $qd=2\pi m$ and $m$ is an integer.
Counting the $N-m$ pairs separated by $md$ gives its powder pattern:

$$
I_{\rm powder}(q)=f(q)^2
\left[N+2\sum_{m=1}^{N-1}(N-m)\operatorname{sinc}(qmd)\right].
$$

These finite patterns already contain size effects. As a rough scale,
features narrow like $2\pi/L$ when the crystal length $L$ increases; the exact
line shape depends on geometry. Do not add a second size broadening merely
because the object is small.

### An ideal infinite crystal has Bragg weights

In the infinite limit, coherent scattering concentrates at reciprocal lattice
vectors. These ideal peaks are delta functions: infinitely narrow mathematical
spikes described by their **weights**, not ordinary finite heights. Independent
random pore populations can also produce continuous diffuse intensity.

<details>
<summary>Show the periodic equations and the bulk-powder normalization</summary>

The code stores direct lattice vectors as the rows of a matrix $\mathbf A$.
Its reciprocal matrix and reflection vectors are

$$
\mathbf B=2\pi\mathbf A^{-T},\qquad
\mathbf G_{hkl}=(h,k,l)\mathbf B,\qquad
V_c=|\det\mathbf A|.
$$

$h,k,l$ are integers, and $V_c$ is the cell volume. For an independently
sampled cell amplitude $F$, define its mean and physical variance:

$$
\overline F(\mathbf q)=\mathbb E[F(\mathbf q)],\qquad
D(\mathbf q)=\mathbb E[|F(\mathbf q)|^2]-|\overline F(\mathbf q)|^2.
$$

$\mathbb E$ means an average over possible pore configurations. Identically
distributed, independent cells give the finite ensemble intensity

$$
\mathbb E[I_{N_c}(\mathbf q)]
=|S(\mathbf q)|^2|\overline F(\mathbf q)|^2+N_cD(\mathbf q).
$$

The first term describes the repeating mean structure; the second describes
its cell-to-cell fluctuations. Dividing by $N_c$ and taking the infinite limit
gives the project's per-cell convention:

$$
I_\infty(\mathbf q)=
\frac{(2\pi)^3}{V_c}\sum_{\mathbf G}
|\overline F(\mathbf G)|^2\delta^{(3)}(\mathbf q-\mathbf G)
+D(\mathbf q).
$$

For a bulk powder, reciprocal vectors with the same nonzero length $q_s$ form
a shell. Its radial delta weight is

$$
W_s=\frac{(2\pi)^3}{4\pi V_c q_s^2}
\sum_{|\mathbf G|=q_s}|\overline F(\mathbf G)|^2,
\qquad
I_{\rm bulk\ powder}(q)=\sum_{s:q_s>0}W_s\delta(q-q_s)
+\frac{1}{4\pi}\int D(q\widehat{\mathbf n})\,d\Omega.
$$

The factor $4\pi q_s^2$ comes from spherical reciprocal-space volume. Every
reflection in a shell is already counted, so there is no extra multiplicity
factor. The forward reflection at $q=0$ is stored separately and is omitted
from this displayed radial formula. These weights are not experimental peak
heights or instrument-corrected line areas.

A single randomly filled cell repeated forever has different correlations
from independent random cells. It does not represent this ensemble model.

</details>

## 5. Putting a nanoparticle in a pore

Think of each pore as a little room. It may be empty, or contain one rigid
particle. A rigid particle moves and turns as a whole; its atoms keep their
distances from each other.

For pore $k$, the model chooses:

1. **Occupancy:** $O_k=1$ with probability $p_k$, otherwise $O_k=0$.
2. **Size/template:** $t$ with probability $w_t$, conditional on occupancy;
   $\sum_t w_t=1$.
3. **Position and orientation:** displacement $\mathbf u_k$ and rotation
   $\mathbf R_k$, subject to the selected confinement model.

If template atom $a$ has local position $\mathbf x_{ta}$, its placed position is

$$
\mathbf r_{ka}=\mathbf c_k+\mathbf u_k+\mathbf R_k\mathbf x_{ta},
$$

where $\mathbf c_k$ is the pore centre. Uniform rotation means every orientation
is equally likely; choosing three Euler angles uniformly does not achieve that.
Size weights count particles, not their mass or their scattering intensity.

For a spherical wall, every atom centre must satisfy

$$
|\mathbf u_k+\mathbf R_k\mathbf x_{ta}|
\le R_{{\rm pore},k}-c_{\rm wall}.
$$

$R_{{\rm pore},k}$ is the pore radius and $c_{\rm wall}$ is the requested
clearance. A failed placement redraws position and orientation **at the same
selected size**. Otherwise difficult-to-fit sizes would quietly become rarer.
This conditional geometric model is a specified probability law; it does not
calculate a thermodynamic equilibrium.

### Why occupancy is not just a weaker atom

Take a guest with fixed amplitude $g$ and occupancy probability $p$. Because
$O$ is either zero or one, $O^2=O$. Therefore

$$
\mathbb E[|Og|^2]=p|g|^2,
\qquad
|\mathbb E[Og]|^2=p^2|g|^2.
$$

For $p=1/2$ and $g=1$, these are $1/2$ and $1/4$. Half the pores containing a
whole particle is different from every pore containing half its amplitude.
For any random complex amplitude $G$, the missing contribution is its variance:

$$
\mathbb E[|G|^2]=|\mathbb E[G]|^2
+\mathbb E[|G-\mathbb E[G]|^2].
$$

## 6. Keep all four kinds of atom pairs

Let $H$ mean the host atoms, and let $k,l$ label **individual physical pores**
throughout the finite crystal. The fast powder model assumes independent pores
and an isotropic local distribution around each centre.

We need two different descriptions of an occupied guest. Its mean amplitude
relative to its centre is

$$
b_k(q)=\mathbb E\!\left[
\sum_a f_{ta}(q)\operatorname{sinc}
\!\left(q|\mathbf u_k+\mathbf R_k\mathbf x_{ta}|\right)
\,\middle|\,O_k=1\right].
$$

The average includes size and accepted placement. Isotropy makes this scalar
quantity the direction-independent mean amplitude. Its full internal intensity is

$$
J_k(q)=\sum_t w_t\sum_{a,b\in t}
f_{ta}(q)f_{tb}(q)
\operatorname{sinc}\!\left(q|\mathbf x_{ta}-\mathbf x_{tb}|\right).
$$

The double sum includes $a=b$, so it includes each guest atom's self term.
Rigid motion does not change $J_k$: the internal pair distances stay fixed.
In general, $J_k\ne b_k^2$.

Now the complete expected finite powder intensity is

$$
\boxed{\mathbb E[I(q)]=I_{HH}(q)+I_{\rm internal}(q)+I_{HG}(q)+I_{GG}(q)}.
$$

**Host with host:** every pair of framework atoms, including self terms.

$$
I_{HH}(q)=\sum_{i,j\in H}f_i(q)f_j(q)
\operatorname{sinc}\!\left(q|\mathbf r_i-\mathbf r_j|\right).
$$

**Inside one guest:** every occupied particle's own Debye intensity.

$$
I_{\rm internal}(q)=\sum_k p_kJ_k(q).
$$

**Host with guest:** interference between the framework and each guest.

$$
I_{HG}(q)=2\sum_{i\in H}\sum_k
f_i(q)p_kb_k(q)
\operatorname{sinc}\!\left(q|\mathbf r_i-\mathbf c_k|\right).
$$

**Guest with another guest:** interference between different pores.

$$
I_{GG}(q)=2\sum_{k<l}p_kp_l b_k(q)b_l(q)
\operatorname{sinc}\!\left(q|\mathbf c_k-\mathbf c_l|\right).
$$

The internal term uses $p_k$ because a particle's atoms appear together. The
last term uses $p_kp_l$ because the two pores are independent. It excludes
$k=l$: those contributions are already in $I_{\rm internal}$.

Random particle orientation does **not** generally erase host–guest or
guest–guest interference. Both cross terms can be negative. The output columns
`host`, `guest_internal`, `host_guest`, `guest_guest`, and `total` correspond to
these equations. For anisotropic local distributions, directional moments and
angular averaging are needed instead of these scalar $b_k$ formulas.

<details>
<summary>A forward-scattering check that includes occupancy and size variation</summary>

Let $H_0=\sum_{i\in H}f_i(0)$ and $T_t=\sum_{a\in t}f_{ta}(0)$.
For the template mixture in pore $k$, write
$\mu_k=\sum_t w_tT_t$ and $\nu_k=\sum_t w_tT_t^2$. Then

$$
\mathbb E[I(0)]=
\left(H_0+\sum_kp_k\mu_k\right)^2
+\sum_k\left(p_k\nu_k-p_k^2\mu_k^2\right).
$$

The second sum is the variance of the total scattering strength from independent
pores. It includes both whole-particle occupancy and size variation. Squaring
only the expected total scattering strength would lose this term. Atomic form factors at zero
are taken from the actual table; their fitted values need not be exactly the
integer atomic numbers.

</details>

## 7. Some pore averages have an exact formula

For uniformly rotated templates, define

$$
B(q)=\sum_t w_t\sum_a f_{ta}(q)
\operatorname{sinc}(q|\mathbf x_{ta}|).
$$

If displacement and rotation are independent, isotropic, and unaltered by a
wall, the occupied mean amplitude separates into

$$
b(q)=B(q)\phi(q).
$$

$\phi$ describes how random centre motion changes the mean wave. For an
unbounded Gaussian displacement with standard deviation $\sigma$ in each
Cartesian direction,

$$
\phi_{\rm Gaussian}(q)=\exp(-q^2\sigma^2/2).
$$

For a centre chosen uniformly throughout the volume of a ball of radius $R$,

$$
\phi_{\rm ball}(q)=3\frac{\sin x-x\cos x}{x^3},
\qquad x=qR,\qquad \phi_{\rm ball}(0)=1.
$$

The ball's radius here is the **allowed centre displacement**, not the particle
radius or the pore radius. A deterministic mean shift can be absorbed into the
effective scattering centre when the analytic model allows it.

If a wall rejects placements, position and orientation generally become linked.
Then the separated formula cannot simply be reused. The code either certifies
that the wall never binds, or samples the accepted distribution. In the
supported isotropic spherical case it can use body coordinates:

$$
\mathbf v=\mathbf R^T\mathbf u,\qquad
|\mathbf u+\mathbf R\mathbf x_a|=|\mathbf v+\mathbf x_a|.
$$

This equality removes unnecessary rotations from the local integration while
keeping the same wall condition. See the
[pore guide](mof_scattering_handoff_20261013/mof_debye_lab/PORE_DISTRIBUTION_GUIDE.md)
for supported distributions and the zero-mean restriction on sampled positions.

## 8. How can we handle a 100 nm crystal?

The synthetic large example is a **100 × 100 × 100 nm cube**. It has 40 cells
along each edge, with a cell edge of $25\ \text{Å}=2.5\ \mathrm{nm}$:

$$
L=40\times2.5=100\ \mathrm{nm},\qquad N_c=40^3=64{,}000.
$$

There are 424 synthetic carbon host sites and one pore per cell:

$$
N_H=424\times64{,}000=27{,}136{,}000.
$$

At occupancy 0.6, the expected particle count is 38,400. With Au13/Au19/Au43
number weights 0.5/0.3/0.2, the expected number of guest atoms is

$$
64{,}000\times0.6\times(0.5\times13+0.3\times19+0.2\times43)
=798{,}720.
$$

These are synthetic numerical fixtures, not a chemically validated MOF. The
separate real IRMOF-1 host example has a **103.328 nm** edge for 40 cells.

Even the synthetic host alone has about $3.68\times10^{14}$ distinct atom
pairs. Direct Debye summation scales approximately as $N^2Q$, where $Q$ is the
number of sampled $q$ values. Storing every pair is unnecessary.

We count how often the same cell separation occurs, then reuse that count.
For a finite cell mask $M(\mathbf n)$, equal to 1 inside the retained shape and
0 outside, define

$$
C(\boldsymbol\ell)=\sum_{\mathbf n}
M(\mathbf n)M(\mathbf n+\boldsymbol\ell).
$$

$C$ counts ordered cell pairs separated by the integer cell offset
$\boldsymbol\ell$. If $\mathbf d_a$ are the host positions within one cell,
the same host Debye equation becomes

$$
I_{HH}(q)=\sum_{\boldsymbol\ell}C(\boldsymbol\ell)
\sum_{a,b}f_a(q)f_b(q)
\operatorname{sinc}\!\left(q|\boldsymbol\ell\mathbf A+\mathbf d_b-\mathbf d_a|\right).
$$

This regrouping is exact. A zero-padded FFT computes the finite shape counts
without wrapping one edge onto the opposite edge. The `radial_moments` backend
then approximates distance sums using radial bins and higher-order corrections.
It retains the finite Debye interference; it does not calculate just one
crystal orientation or replace the crystal by an infinite-lattice peak list.

<details>
<summary>What approximation is inside a radial bin?</summary>

For one species-pair channel, let $w_j$ be nonnegative pair multiplicities and
$r_j$ the positive pair distances. Set $r_j=\rho_b+\epsilon_j$ in bin $b$.
The implementation accumulates scaled versions of the moments

$$
M_{bm}=\sum_{j\in b}\frac{w_j}{r_j}\epsilon_j^m.
$$

For $q>0$, it evaluates the Taylor expansion through order $P$:

$$
\sum_j w_j\operatorname{sinc}(qr_j)
\approx\frac1q\operatorname{Im}\!
\sum_b e^{\mathrm{i}q\rho_b}
\sum_{m=0}^{P}\frac{(\mathrm{i}q)^m}{m!}M_{bm}.
$$

Uniform bin centres allow FFT evaluation. Keeping $m>0$ preserves information
about distances within a bin. A Taylor remainder bound for this positive-radius
sum is

$$
|E(q)|\le\frac{q^P}{(P+1)!}
\sum_j\frac{w_j}{r_j}|\epsilon_j|^{P+1}.
$$

Zero distances and $q=0$ are handled separately. Species factors and optional
displacement-merging bounds are propagated into the reported intensity bound.
This bound excludes floating-point roundoff, sampling error and missed features
between the chosen $q$ points. The implementation also requires a uniform grid
starting at zero with $r_{\max}<\pi/\Delta q$ and enforces transform-size limits.

</details>

## 9. What does a 0.1% accuracy target mean?

The target concerns the numerical solution of the stated model. Four questions
need separate checks:

| Question | Check |
|---|---|
| Did the radial approximation change the intensity? | Report its bound, refine bins/order, and compare with exact references |
| Were enough accepted pore placements sampled? | Increase samples, compare independent seeds, and inspect Monte Carlo SEM |
| Is the $q$ grid fine enough to resolve narrow peaks? | Refine $\Delta q$ and compare peak heights, positions and areas |
| Does the structure describe a real material? | Validate the CIF, guest templates, pore geometry and physical assumptions |

For $B$ independent batch intensities $I_1(q),\ldots,I_B(q)$, the sampled
intensity and its estimated standard error of the mean (SEM) are

$$
\overline I(q)=\frac1B\sum_{b=1}^{B}I_b(q),\qquad
\operatorname{SEM}(q)=
\sqrt{\frac{\sum_b[I_b(q)-\overline I(q)]^2}{B(B-1)}}.
$$

The fast pore estimator uses independent mean-amplitude replicas when forming
products, so it does not introduce the bias of squaring one noisy mean. Its
batch SEM measures numerical integration uncertainty; it is different from
the physical disorder variance $D$ in the periodic equations.

Our benchmark checks relative errors where the reference intensity is at least
1% of its non-forward peak, using $q\ge0.1\ \text{Å}^{-1}$. Near zeros it uses
absolute error scaled by that peak. Neither normalizing by the huge forward
beam nor quoting an SEM alone establishes 0.1% accuracy everywhere. The measured
[pore report](mof_scattering_handoff_20261013/PORE_DISTRIBUTION_REPORT.md) and
[large-PXRD report](mof_scattering_handoff_20261013/LARGE_PXRD_REPORT.md)
record the actual settings and validation results.

## 10. Run the examples yourself

From the repository root, create an isolated environment with Python 3.10 or
newer. These are PowerShell commands; use the environment's `bin/python` path
instead on Linux or macOS.

```powershell
cd mof_scattering_handoff_20261013
python -m venv .venv
cd mof_debye_lab
..\.venv\Scripts\python.exe -m pip install -e ".[test,fast]"
```

Start with the small physical examples. They check one atom, two atoms, a chain,
crystal extinctions and random guest distributions against known equations:

```powershell
..\.venv\Scripts\python.exe scripts/validate_physical_examples.py --out ../local_runs/my_tutorial_physics
```

Then try each of the three modes and a confined pore distribution:

```powershell
..\.venv\Scripts\python.exe scripts/run_scattering.py --config examples/modes/finite_single_crystal.json --out ../local_runs/my_tutorial_single
..\.venv\Scripts\python.exe scripts/run_scattering.py --config examples/modes/periodic_single_crystal.json --out ../local_runs/my_tutorial_periodic
..\.venv\Scripts\python.exe scripts/run_scattering.py --config examples/modes/pxrd_finite.json --out ../local_runs/my_tutorial_powder
..\.venv\Scripts\python.exe scripts/run_scattering.py --config examples/pore_distribution/active_wall.json --out ../local_runs/my_tutorial_pores
```

Use fresh output directories. The supplied pore example has $2\times2\times2$
cells, so it is a small starting point. The large synthetic run is explicit:

```powershell
..\.venv\Scripts\python.exe scripts/benchmark_pore_powder.py --cells 40 --threads 6 --samples 1048576 --seed 20261014 --out ../local_runs/my_tutorial_100nm
..\.venv\Scripts\python.exe -m pytest -q
```

The recorded v0.4.1 validation has **980 passing tests**. The implementation uses
kinematic, elastic scattering with real atomic factors; the default factors are
Gemmi IT92. Finite intensities are unnormalized, in electron squared. Thermal
motion, anomalous scattering, absorption, Lorentz/polarization corrections and
instrument response are outside this core. The separately optional installed
DebyeCalculator backend uses its own factors; see the mode guide before comparing
its output to IT92 calculations.

### Where to go next

| I want to… | Read |
|---|---|
| Configure the modes and understand output files | [Mode guide](mof_scattering_handoff_20261013/mof_debye_lab/MODE_GUIDE.md) |
| Import a CIF, expand symmetry and choose finite shapes | [CIF guide](mof_scattering_handoff_20261013/mof_debye_lab/R07_GUIDE.md) |
| Run a sourced real MOF | [IRMOF-1 example](mof_scattering_handoff_20261013/mof_debye_lab/examples/real_mof/README.md) |
| Inspect examples with known physical solutions | [Physical examples](mof_scattering_handoff_20261013/mof_debye_lab/examples/physical/README.md) |
| Configure particle sizes, occupancy and confinement | [Pore guide](mof_scattering_handoff_20261013/mof_debye_lab/PORE_DISTRIBUTION_GUIDE.md) |
| Understand large-crystal speed and accuracy | [Large PXRD guide](mof_scattering_handoff_20261013/mof_debye_lab/LARGE_PXRD_GUIDE.md) |
| Read the original prototype and provenance | [Historical project README](mof_scattering_handoff_20261013/mof_debye_lab/README.md) |

For another teaching treatment connecting diffraction angles, structure factors
and finite-particle Debye scattering, see
[IUCr: X-ray powder diffraction in education, Part II](https://journals.iucr.org/j/issues/2023/03/00/dv5004/index.html).
