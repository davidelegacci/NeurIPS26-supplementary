# NeurIPS 2026 supplementary: exact harmonic-game checks

Self-contained code and original data for the paragraph **“Exact checks and a
concrete illustration”** in `NeurIPS-2026/Appendix/App-Related.tex`.

This repository reproduces the following results:

| Games | Number | Effective update-space dimension | Center Jacobian rank |
|---|---:|---:|---:|
| Uniformly harmonic, three players with two actions each | 20 | 3 in every case | 2 in every case |
| Weighted harmonic, three players with two actions each | 20 | 3 in every case | 2 in every case |
| Connected, curved Nash set (Figure 1, left) | 1 | 3 | 2 |
| Disconnected Nash set (Figure 1, middle) | 1 | 3 | 2 |

Every rank is computed over the rationals. The repository contains the **actual
games used in the paragraph**, not a new sample. No learning simulations,
equilibrium solvers, notebooks, or external research repositories are needed.

## Run

Use Python 3.12 (the original run used 3.12.3):

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --output reproduced
python -m unittest discover -s tests -v
```

The first command regenerates the 40 sampled payoff tables from seed `20261107`,
compares every draw and payoff with the archive, and verifies all 42 games. It
also compares the newly calculated fields and matrices against the independently
saved original audit and the checked-in exact report. Expected output:

```text
PASS: 20 uniform + 20 weighted games, and both Figure 1 games.
All 42: dim U = 3; rank Dv(q) = 2; restricted entropic score rank = 2.
Exact agreement with original payoff tables and archived calculations.
Original seed, all random draws, and solved payoffs reproduced exactly.
```

To verify the existing payoff tables without regenerating the sample:

```sh
python reproduce.py --saved-only
```

Without `--output`, verification writes no reports. With `--output`, use a fresh
directory: the program refuses to overwrite previous results. GitHub Actions
runs the full reproduction and tests on Python 3.12.

## Contents

- `harmonic_games.py`: standalone generator and exact mixed payoff-difference field.
- `reproduce.py`: sample regeneration, exact checks, rank certificates, and report export.
- `data/games.json`: all 42 payoff tables, action measures, and the original random draws.
- `data/reference-checks.json`: original archived fields, bases, Jacobians, and ranks.
- `results/exact-checks.json`: independently recomputed results, including rank certificates.
- `results/generator-solutions.json`: both harmonic systems and their general solutions.
- `results/summary.csv`: one row per game.
- `results/environment.json`: dependency versions and input/source SHA-256 hashes.
- `provenance/original_generator.py`: the original notebook's executed mathematical cells,
  converted to plain Python. Tests compare its systems and symbolic solutions with
  the standalone implementation for both harmonic measures.
- `provenance/source-manifest.json`: original run identifier, versions, source hashes,
  and the list of extracted cases.
- `tests/test_exact.py`: original-generator agreement, full regeneration, direct
  payoff-equation checks, coordinate ordering, and certificate tests.
- `tools/import_legacy_archive.py`: records how the original archive was imported;
  it is **not** needed to use or reproduce this repository.

## Payoff and coordinate conventions

Players and pure actions are indexed from zero in the code. Each player has
actions `0` and `1`. Pure profiles occur in lexicographic order:

```text
000, 001, 010, 011, 100, 101, 110, 111
```

The 24-entry payoff list contains the eight payoffs of player 1, followed by
those of player 2, then player 3. In formulas, players are indexed $i=1,2,3$.
The effective strategy coordinate $x_i$ is the probability of the **second**
action, `1`. Thus

$$
v_i(x)=u_i(1,x_{-i})-u_i(0,x_{-i}).
$$

Rational numbers in JSON are exact strings, such as `"2/3"`; they are not
rounded floating-point values.

## Generating harmonic games

For prescribed positive action measures $\mu_i$, solve the homogeneous linear
equations, one per pure profile $a$:

$$
\sum_i\sum_{b_i}\mu_i(b_i)
\bigl[u_i(a)-u_i(b_i,a_{-i})\bigr]=0.
$$

The generator builds their coefficient matrix $A$ and solves $Au=0$ with
SymPy, retaining the original payoff-symbol names and ordering. Both 222
systems have rank 7 and 17 free payoff entries among the 24 entries.

Sampling uses NumPy `default_rng(20261107)`, first 20 uniform games with
measures $(1,1)$ for each player, then 20 weighted games with measures
$(1,2)$, $(1,3)$, $(1,4)$. One generator is used continuously; the seed is not
reset between families. Free entries receive integers from $[-10,10)$;
dependent entries are obtained from the exact solution and need not lie in
that interval or be integers.

**Exact sampling detail.** The original notebook draws 24 integers per game,
one for every payoff symbol, before substituting them into the solved table.
Only the 17 draws assigned to free symbols affect that game. The other seven
are discarded, but still advance the random-number generator. This behavior
is preserved, and all 24 draws are saved. The free-symbol indices are recorded
in `results/generator-solutions.json`.

The two Figure 1 games are fixed benchmark tables, not additional draws from
this sample. Their full original payoffs and measures are in `data/games.json`.

To generate a new harmonic game independently of the archived sample:

```python
import numpy as np
from harmonic_games import solve_family

family = solve_family([[1, 2], [1, 3], [1, 4]])
payoffs, draws = family.sample(np.random.default_rng(123))
print(payoffs)  # exact SymPy integers/rationals, in the documented payoff order
```

`solve_family` also accepts other finite action counts through the lengths of
the measure lists. The paragraph's field/rank audit is deliberately limited
to three-player, two-action games.

## Exact calculations

Set

$$
m_i=\mu_i(0)+\mu_i(1),\qquad
q_i=\frac{\mu_i(1)}{m_i},\qquad W=\mathrm{diag}(m_1,m_2,m_3).
$$

For each table the program verifies the pure-profile equations, constructs
$v$ from the payoffs, and checks exactly that

$$
(x-q)^\top Wv(x)=0,\qquad v(q)=0,\qquad
WDv(q)+Dv(q)^\top W=0.
$$

The weighted identity concerns $WDv(q)$: $Dv(q)$ itself need not be
skew-symmetric in the weighted examples.

### Dimension of the effective update space

The space in the paragraph is

$$
\mathcal U=\mathrm{span}\,\{v(x):x\in[0,1]^3\}.
$$

The field is multi-affine, so it is exactly the multilinear interpolation
of its eight vertex values. Consequently,

$$
\mathcal U=\mathrm{range}\, B,\qquad
B=[v(0,0,0)\ \cdots\ v(1,1,1)].
$$

The code checks the interpolation identity symbolically and computes the
rank of the rational $3\times8$ matrix $B$. For every game the report contains
$B$, a basis of its columns, its reduced row-echelon form, and a nonzero
$3\times3$ minor certifying rank 3. Minor row/column indices are zero-based.

### Center Jacobian

The program differentiates the polynomial field and evaluates $J=Dv(q)$.
It independently reconstructs this same matrix from differences between
vertex field values, weighted by the opponents' center probabilities.
For each game it saves $J$, its row reduction, a nonzero $2\times2$ minor,
its zero determinant, and a rational kernel vector. Together these certify
rank **exactly** 2, not merely numerical near-singularity.

For entropy, it also checks the score derivative $J\,\mathrm{diag}(q_i(1-q_i))$,
represented in the saved basis of $\mathcal U$. Its rank is 2 in all 42 games.

### Why these checks imply failure of Assumption 5

The relevant score map is $\pi_{z_0}(u)=v(Q(z_0+u))$ on $\mathcal U$.
Here $\mathcal U=\mathbb R^3$, so the harmonic center lies in the image of
the affine score space $z_0+\mathcal U$ for every interior initialization,
under the Legendre choice-map hypotheses used in the paper. At a score
representing $q$,

$$
D\pi_{z_0}=Dv(q)\,DQ,
$$

which is singular. Thus the required nonsingularity at every reachable zero
fails. “Reachable” here refers to the affine score space, not to a claim that
a particular learning trajectory visits the center. These exact instances
support the paragraph's reported checks; the preceding genericity argument
in the paper is an analytical argument, not a statistical inference from 40 samples.

## Provenance

The source is audit run `20261107-002715-434849`, originally produced with
Python 3.12.3, NumPy 2.4.3, and SymPy 1.12.1. That broader audit also included
seven examples/controls outside this paragraph; those are not part of this
42-game supplement. The numerical input tables and original reference
calculations have been retained independently of the new verification code.

The standalone generator is extracted from mathematical cells 0–35 of
`generate_harmonic_game_clean.ipynb`. The archived code copy omits notebook
outputs and later experiments and makes the two configuration cells
overridable. The import manifest records the original notebook's SHA-256
and other source hashes, without embedding machine-specific paths.
