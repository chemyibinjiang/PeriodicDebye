# Synthetic independent pore populations

These are numerical fixtures, not chemically validated MOFs or measured loading.
The 25 A cell is an 8-by-8-by-8 carbon point grid with its central 8.5 A sphere
removed (424 host sites). Its central spherical pore has radius 8 A and an
additional atom-centre clearance of 0.5 A. All accepted Au coordinates therefore
lie within 7.5 A of the pore centre. The nearest synthetic host site is farther
than 8.5 A from the centre; this fixture does not infer chemical exclusion radii.

Every retained pore is independently occupied with probability 0.6. Conditional
on occupation, Au13/Au19/Au43 templates have number probabilities 0.5/0.3/0.2.
Their illustrative 4.08 A fcc lattice is fixed, and orientation is uniform on
SO(3). The supplied examples contain 2-by-2-by-2 cells and 601 q points.

- `analytic.json`: particle centres are uniform in a solid ball of radius 2 A.
  Even the largest template fits for every orientation, so confinement is
  certified and distribution moments can be integrated analytically.
- `active_wall.json`: the proposal ball has radius 4 A. Placements crossing the
  spherical wall are rejected at fixed template size. Position and orientation
  are jointly conditioned; exact size weights remain unchanged. The output
  includes Monte Carlo integration SEM from eight independent batches.

From the project directory:

```powershell
..\.venv\Scripts\python.exe scripts/run_scattering.py --config examples/pore_distribution/analytic.json --out ../local_runs/my_pores_analytic
..\.venv\Scripts\python.exe scripts/run_scattering.py --config examples/pore_distribution/active_wall.json --out ../local_runs/my_pores_active_wall
..\.venv\Scripts\python.exe scripts/benchmark_pore_powder.py --cells 40 --threads 6 --out ../local_runs/my_pores_100nm
```

The last command generates self-contained configurations and calculates a
100 nm cube with 27,136,000 host atoms represented by cell-pair counts, 64,000
independent pores, and 798,720 expected Au atoms. It reuses geometry for both
position models. It does not create a complete atom list or repeat one random
cell realization. `--samples` controls local integration work; the default
4096 samples per template/pore/replica is a demonstration, not a guarantee of
0.1% accuracy for actively conditioned distributions. See the project
[guide](../../PORE_DISTRIBUTION_GUIDE.md) and
[measured report](../../../PORE_DISTRIBUTION_REPORT.md).
