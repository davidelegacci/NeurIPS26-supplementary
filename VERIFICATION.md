# Verification record

Verified on 7 November 2026 in a new Python 3.12.3 virtual environment,
installing only `requirements.txt` (NumPy 2.4.3, SymPy 1.12.1, mpmath 1.3.0).

```sh
python reproduce.py --output reproduced
python -m unittest discover -s tests -v
python reproduce.py --saved-only
```

Results:

- All 40 sampled payoff tables and all 960 integer draws reproduce the original archive exactly.
- Both generator families agree with the original notebook's executed code:
  the same 8-by-24 harmonic-equation matrices and the same symbolic solutions.
- Each system has rank 7 and 17 free payoff entries.
- Both Figure 1 tables match their original archived calculations.
- All 42 pure-profile residuals and mixed harmonic identities vanish exactly.
- All 42 effective update spans have dimension 3, certified by rational row
  reductions and nonzero 3-by-3 minors.
- All 42 center Jacobians have rank 2, certified by rational row reductions,
  nonzero 2-by-2 minors, and zero determinants. Their kernel vectors and
  weighted skew-symmetry checks are included in the report.
- Jacobians independently reconstructed from vertex-field differences agree
  with the symbolic derivatives.
- The restricted entropic score Jacobians also have rank 2 in every case.
- The six regression tests pass, including the comparison against the original generator.

The reproducibility command also checks equality with the committed exact
report. Runtime/source metadata are saved separately, so environment metadata
does not affect equality of the mathematical results.
