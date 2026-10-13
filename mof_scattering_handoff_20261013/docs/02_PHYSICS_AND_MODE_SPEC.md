# 02 — Physics and mode specification

This is the implementation contract derived from the discussion. New mode names and interfaces below are **proposals**, not functions already present in v0.1. Initially retain kinematic elastic scattering, a fixed host, scalar neutral-atom real form factors, and no instrument corrections. Complex factors, thermal disorder, host relaxation and detector geometry require separately tested extensions.

## A. Units and coordinates

Use Cartesian positions in angstrom, scalar q and vector q in inverse angstrom, and

\[
q=4\pi\sin\theta/\lambda,\qquad \operatorname{sinc}(x)=\sin(x)/x.
\]

NumPy uses a normalized sinc, so call `np.sinc(q*r/np.pi)`. Let the direct lattice vectors be the **rows** of A; a fractional row coordinate x maps to `x @ A`. Reciprocal vectors are rows of `B = 2*pi*inv(A).T`, hence `q_hkl = hkl @ B` and `A @ B.T = 2*pi*I`. Do not assume orthogonal cells.

Document a particle template's rotation origin. Keep each rigid particle internally unwrapped even if it straddles a periodic cell face. Pore centres, host atoms and guest offsets must share the same coordinate frame.

## B. Three user-facing cases

| Case name (proposed) | Boundary/orientation | Raw output |
|---|---|---|
| `single_crystal_periodic` | Infinite periodic average, fixed host orientation | hkl/q-vector reflection table and separate continuous diffuse intensity |
| `pxrd` | Whole-object orientation average; explicit `finite` or `bulk` boundary | finite Debye I(q), or bulk shell weights plus powder-averaged diffuse intensity |
| `single_crystal_finite` | Finite atomic object/shape, fixed orientation | I(q-vector), optionally complex amplitudes and reciprocal-space slices |

Keep `guest_estimator = explicit | analytic | moments` separate from `scattering_case`, `boundary`, and `backend`. Analytical estimators must declare and enforce their supported distributions. The existing `--mode analytic|montecarlo` maps to `guest_estimator` only.

A powder orientation distribution belongs to the **whole host–guest object**. The guest rotation distribution is relative to the host. Nonuniform powder texture generally needs angular integration rather than the standard isotropic Debye shortcut.

## C. Finite single crystal and its powder average

For one finite configuration,

\[
A(\mathbf q)=\sum_i f_i(|\mathbf q|)e^{i\mathbf q\cdot\mathbf r_i},
\qquad I(\mathbf q)=|A(\mathbf q)|^2.
\]

Implement this simple reference first, with atom/q-vector tiling to bound memory. It includes finite-size interference without any extra Scherrer convolution. The random-powder average is

\[
I_{\mathrm{pow}}(q)=\frac{1}{4\pi}\int_{S^2}I(q\hat{\mathbf n})d\Omega
=\sum_i f_i^2+2\sum_{i<j}f_if_j\operatorname{sinc}(qr_{ij}).
\]

This identity is the primary cross-mode acceptance test. Increase angular resolution with q times object extent; a coarse spherical grid will miss narrow diffraction features. Do not convert a 1D Debye curve back into a directional pattern.

For complete identical cells under a binary mask w,

\[
A(\mathbf q)=F_{\mathrm{cell}}(\mathbf q)S_w(\mathbf q),\quad
S_w(\mathbf q)=\sum_n w_n e^{i\mathbf q\cdot\mathbf R_n}.
\]

For powder, average `|F_cell(q-vector) S_w(q-vector)|^2` as a product. Do not multiply independently averaged cell and shape intensities.

## D. Finite-shape lattice grouping for powder

Define integer-translation counts

\[
C_w(T)=\sum_n w_nw_{n+T}.
\]

A linearly zero-padded FFT autocorrelation gives these counts. With cell basis positions d_alpha,

\[
I(q)=\sum_T C_w(T)\sum_{\alpha\beta} f_\alpha f_\beta
\operatorname{sinc}\!\left(q|T A+d_\alpha-d_\beta|\right).
\]

Self terms are already included at T=0 and alpha=beta. Check `C(0)=M`, `sum C=M^2`, and `C(T)=C(-T)`. This is exact for retained full cells, not for unmodelled surface reconstruction. For atom-level cuts, use basis-specific masks/appropriate cross-correlations or an explicit atomic fallback. Cost depends on basis size, distinct translations and bounding-grid volume; do not promise linear scaling for arbitrary sparse geometry.

## E. Current restricted analytical finite-powder guest model

At pore k, occupation n_k is Bernoulli with probability p_k. Conditional on occupancy, select template t with number weight w_t, rotate the **whole** template uniformly on SO(3), and independently translate its centre by an isotropic Gaussian with per-coordinate standard deviation sigma. Allow a specified mean offset by absorbing it into c_k.

Define

\[
B(q)=\sum_t w_t\sum_{a\in t}f_a(q)\operatorname{sinc}(q|x_a|),\quad
P(q)=\sum_t w_t I_t^{\mathrm{Debye}}(q),\quad
D(q)=e^{-q^2\sigma^2/2}.
\]

Then the implemented finite average is

\[
\begin{aligned}
\langle I\rangle={}&I_H+\sum_k p_kP(q)\\
&+2B(q)D(q)\sum_kp_k\sum_{i\in H}f_i(q)\operatorname{sinc}(q|r_i-c_k|)\\
&+2B(q)^2D(q)^2\sum_{k<l}p_kp_l\operatorname{sinc}(q|c_k-c_l|).
\end{aligned}
\]

The internal term has p, not p squared: all atoms of a particle appear together. B is a mean **amplitude**, P a mean **intensity**. Do not substitute one for the other. Internal atom pairs rotate together and keep their distances.

The present formula assumes the same library and covariance across pores. Per-pore occupancy and mean offsets are supported. Hard-wall rejection, host clearance, anisotropic covariance, nonuniform rotations or correlated pores are outside this analytical implementation. Explicit sampling handles several of these local distributions, but not inter-pore correlations.

## F. General independent-pore amplitude moments — next shared layer

At fixed vector q define a pore's random amplitude, including occupancy and its spatial phase,

\[
g_k(\mathbf q)=n_k e^{i\mathbf q\cdot(c_k+u_k)}F_{t_k}(R_k^T\mathbf q),
\quad \mu_k=\mathbb E[g_k],\quad \nu_k=\mathbb E[|g_k|^2].
\]

For a fixed finite host and **independent pores**,

\[
\mathbb E[I(\mathbf q)]=\left|H(\mathbf q)+\sum_k\mu_k(\mathbf q)\right|^2
+\sum_k\left(\nu_k(\mathbf q)-|\mu_k(\mathbf q)|^2\right).
\]

This is valid for an arbitrary joint size/position/orientation distribution within each pore, provided the moments are evaluated for that actual distribution. Confinement can couple those variables. Obtain moments analytically only where justified, otherwise by deterministic quadrature or local Monte Carlo. This is a route to avoid sampling the entire large host in every configuration.

The expectation does not reproduce the speckle of one quenched finite configuration; provide distinct outputs for a realization and an ensemble mean. Track integration uncertainty separately from the physical variance nu minus |mu| squared.

**Moment sampling caution:** a noisy sample mean squared is biased for the population mean squared, especially after multiplying by a large lattice factor. Do not silently treat a small Monte Carlo mean as exact. For L independent cell-amplitude draws, sample mean gbar and unbiased complex sample variance s², an unbiased estimate of `|H+mu|²` is `|H+gbar|² - s²/L`. For the identical-independent-cell formula below, this gives `|S|²*(|H+gbar|²-s²/L)+M*s²`. Prefer analytical moments or converged quadrature when available; document uncertainty and possible finite-sample negative estimates rather than clipping them into a claim of exactness.

## G. Periodic single crystal with independently disordered cells

Let H(q-vector) be one host-cell amplitude, and G_n the random total guest amplitude **inside that cell's coordinate frame**. Cells are independent and identically distributed; within-cell correlations may be included in G_n. Define mu=E[G_n] and v=E[|G_n|²]-|mu|².

For M retained cells under any mask,

\[
\mathbb E[I_M(\mathbf q)]
=|H(\mathbf q)+\mu(\mathbf q)|^2|S_w(\mathbf q)|^2+M v(\mathbf q).
\]

For a three-dimensionally growing bulk crystal, using the Fourier convention above and cell volume V_c,

\[
\lim_{M\to\infty}\frac{|S_w(\mathbf q)|^2}{M}
=\frac{(2\pi)^3}{V_c}\sum_{G\in L^*}\delta^{(3)}(\mathbf q-G).
\]

Thus per-cell intensity is the sum of Bragg delta weights proportional to `|H(G)+mu(G)|²` and continuous diffuse intensity v(q-vector). Output a reflection table, not a finite ideal peak height. Record whether weights include the `(2*pi)^3/V_c` measure prefactor. The host–guest interference remains inside `|H+mu|²`.

A statistically random particle at every pore does not mean one sampled cell should be replicated periodically. That would impose correlations absent from the independent-cell model. Diffuse intensity can contain structured broad particle features; it need not be flat.

If cells are correlated, replace v by the lattice Fourier transform of the covariance. With `Gamma_T = E[(G_(n+T)-mu)(G_n-mu)*]`,

\[
D_{\rm corr}(\mathbf q)=\sum_T e^{i\mathbf q\cdot(T A)}\Gamma_T(\mathbf q).
\]

Long-range correlations can yield additional sharp features. This extension is not implemented and must not be assumed zero without declaring independence. A fluctuating host requires moments/covariance of the complete cell, not a fixed-host approximation labelled as relaxed structure.

## H. Bulk powder and instrument outputs

For G not equal to zero, the spherical average of a reciprocal-space delta is `delta(q-|G|)/(4*pi*|G|²)`. Under the per-cell reciprocal-space density convention in G, bulk powder therefore has

\[
I_{\rm pow,bulk}(q)=\frac{(2\pi)^3}{4\pi V_c}
\sum_{G\ne0}\frac{|H(G)+\mu(G)|^2}{|G|^2}\delta(q-|G|)
+\frac{1}{4\pi}\int v(q\hat n)d\Omega.
\]

This states a **radial intensity-density convention**, not a universal experimental powder stick convention. Keep reflection `|F|²`, powder shell weights, and experimental integrated line intensities separately named. Enumerate reciprocal vectors consistently so multiplicities are not counted twice. Treat the forward G=0 contribution separately; do not divide it by zero.

Instrument convolution and q-to-2theta conversion are later layers. State whether a quantity is a point value or a density before applying Jacobians. Use unit-area profile kernels and verify conservation under the chosen measure. Do not double-count finite-size broadening by convolving a fully finite atomistic pattern with another size-broadening term. Polarization, absorption, Lorentz, wavelength and resolution settings must be explicit. A reciprocal plane is not an Ewald-sphere detector projection.
