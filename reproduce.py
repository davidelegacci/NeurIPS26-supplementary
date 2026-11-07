"""Regenerate the archived sample and verify every exact result in the paragraph."""
import argparse
import csv
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import platform

import numpy as np
import sympy as sp
import mpmath

from harmonic_games import binary_field, master_matrix, rational, solve_family

ROOT = Path(__file__).resolve().parent
SEED = 20261107
FAMILIES = (
    ("uniform-222", [[1, 1], [1, 1], [1, 1]], 20),
    ("weighted-222", [[1, 2], [1, 3], [1, 4]], 20),
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def exact_matrix(matrix):
    return [[str(v) for v in row] for row in matrix.tolist()]


def rank_certificate(matrix):
    """RREF gives an upper bound; a nonzero maximal minor gives the lower bound."""
    rref, pivots = matrix.rref()
    rank = len(pivots)
    rows = next((list(rows) for rows in combinations(range(matrix.rows), rank)
                 if matrix.extract(rows, pivots).det() != 0), [])
    determinant = matrix.extract(rows, pivots).det() if rank else sp.S.One
    require(rank == matrix.rank(), "Rank/RREF disagreement")
    return dict(rank=rank, rref=exact_matrix(rref), pivot_columns=list(pivots),
                minor_rows=rows, minor_columns=list(pivots),
                minor_determinant=str(determinant))


def audit_game(game):
    mu = [[rational(v) for v in row] for row in game["measures"]]
    n = len(mu)
    require(n == 3 and all(len(row) == 2 for row in mu), "Expected a 222 game")
    flat = [rational(v) for v in game["payoffs"]]
    A, profiles = master_matrix(mu)
    residual = A * sp.Matrix(flat)
    require(residual == sp.zeros(len(profiles), 1), "Pure-profile harmonic residual")
    x, v = binary_field(flat, n)
    masses = [sum(row) for row in mu]
    q = sp.Matrix([row[1] / sum(row) for row in mu])
    W = sp.diag(*masses)
    center = dict(zip(x, q))
    identity = sp.expand((sp.Matrix(x) - q).dot(W * v))
    require(identity == 0, "Mixed harmonic identity failed")
    require(v.subs(center) == sp.zeros(n, 1), "Center is not stationary")
    require(all(sp.diff(v[i], x[i]) == 0 for i in range(n)), "Own-action dependence")

    # The exact multiaffine interpolation identity proves that vertex values
    # span ALL field values, not just a finite subset of samples.
    B = sp.Matrix.hstack(*[v.subs(dict(zip(x, a))) for a in profiles])
    interpolation = sp.zeros(n, 1)
    for k, a in enumerate(profiles):
        weight = sp.prod(x[j] if a[j] else 1 - x[j] for j in range(n))
        interpolation += weight * B[:, k]
    require((interpolation - v).applyfunc(sp.expand) == sp.zeros(n, 1),
            "Multiaffine interpolation failed")
    span_certificate = rank_certificate(B)
    U = B[:, span_certificate["pivot_columns"]]
    J = v.jacobian(x).subs(center)
    require(W * J + J.T * W == sp.zeros(n), "Weighted skew symmetry failed")
    jacobian_certificate = rank_certificate(J)
    mobility = sp.diag(*[qi * (1 - qi) for qi in q])
    score = (U.T * U).inv() * U.T * J * mobility * U

    # Independently obtain Dv(q) from finite differences of the pure-profile
    # field values (exact because the field is multiaffine).
    J_direct = sp.zeros(n)
    for j in range(n):
        for k, a in enumerate(profiles):
            if a[j]:
                continue
            b = a[:j] + (1,) + a[j + 1:]
            weight = sp.prod(q[l] if a[l] else 1 - q[l] for l in range(n) if l != j)
            J_direct[:, j] += weight * (B[:, profiles.index(b)] - B[:, k])
    require(J_direct == J, "Independent Jacobian check failed")
    require(span_certificate["rank"] == 3, f"{game['name']}: update span is not 3")
    require(jacobian_certificate["rank"] == 2, f"{game['name']}: center rank is not 2")
    require(score.rank() == 2 and J.det() == 0, "Score-coordinate rank check failed")

    result = dict(name=game["name"], measures=game["measures"],
                  field=[str(f) for f in v], center=[str(qi) for qi in q],
                  pure_profile_residual=[str(r) for r in residual],
                  harmonic_identity=str(identity),
                  vertex_field_matrix=exact_matrix(B),
                  update_space_basis=exact_matrix(U),
                  update_space_dimension=U.cols, update_span_certificate=span_certificate,
                  center_jacobian=exact_matrix(J), center_jacobian_rank=J.rank(),
                  center_jacobian_determinant=str(J.det()),
                  center_jacobian_certificate=jacobian_certificate,
                  center_jacobian_kernel=[[str(v) for v in k] for k in J.nullspace()],
                  weighted_skew_residual=exact_matrix(W * J + J.T * W),
                  restricted_entropy_score_jacobian=exact_matrix(score),
                  restricted_entropy_score_rank=score.rank(),
                  fails_from_every_interior_initialization=True)
    return result


def regenerate_sample():
    rng = np.random.default_rng(SEED)
    games, families = [], {}
    for name, measures, count in FAMILIES:
        family = solve_family(measures)
        families[name] = dict(count=count, measures=measures,
                              master_matrix=exact_matrix(family.matrix),
                              master_equation_rank=family.matrix.rank(),
                              payoff_family_dimension=len(family.free_indices),
                              symbol_order=[str(s) for s in family.symbols],
                              free_symbol_indices=list(family.free_indices),
                              general_solution=[str(s) for s in family.solution])
        require(family.matrix.rank() == 7 and len(family.free_indices) == 17,
                "Unexpected harmonic family dimension")
        for k in range(count):
            payoffs, draws = family.sample(rng)
            games.append(dict(name=f"{name}-{k+1:02d}", measures=measures,
                              payoffs=[str(v) for v in payoffs],
                              drawn_payoff_parameters=draws))
    return games, families


def verify_archive(regenerate=True):
    archived = json.loads((ROOT / "data/games.json").read_text())
    require(archived["seed"] == SEED, "Archived seed differs")
    require(archived["action_profiles"] == [list(a) for a in product((0, 1), repeat=3)],
            "Archived pure-profile order differs")
    games = archived["games"]
    names = [g["name"] for g in games]
    require(len(games) == len(set(names)) == 42, "Expected 42 distinct archived games")
    families = None
    if regenerate:
        generated, families = regenerate_sample()
        expected = [g for g in games if g["name"].startswith(("uniform-222-", "weighted-222-"))]
        require(generated == expected, "Regenerated draws/payoffs differ from the original archive")
    records = [audit_game(g) for g in games]
    reference = json.loads((ROOT / "data/reference-checks.json").read_text())
    for row, expected in zip(records, reference["cases"]):
        for key, value in expected.items():
            require(row[key] == value, f"Original archive mismatch: {row['name']}: {key}")
    require(len(reference["cases"]) == len(records), "Missing reference records")
    report = dict(seed=SEED, cases=records)
    return report, families


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New directory for exact JSON/CSV reports")
    parser.add_argument("--saved-only", action="store_true", help="Check saved games without regenerating the sample")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output directory already exists; choose a fresh path")
    report, families = verify_archive(regenerate=not args.saved_only)
    expected_path = ROOT / "results/exact-checks.json"
    if expected_path.exists():
        require(report == json.loads(expected_path.read_text()), "Checked-in exact report differs")
    if args.output:
        args.output.mkdir(parents=True)
        (args.output / "exact-checks.json").write_text(json.dumps(report, indent=2) + "\n")
        if families:
            (args.output / "generator-solutions.json").write_text(json.dumps(families, indent=2) + "\n")
        with (args.output / "summary.csv").open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["game", "dim_U", "rank_Dv_q", "rank_restricted_score", "assumption_5_fails"])
            for row in report["cases"]:
                writer.writerow([row["name"], row["update_space_dimension"], row["center_jacobian_rank"],
                                 row["restricted_entropy_score_rank"], True])
        inputs = [ROOT / p for p in ("harmonic_games.py", "reproduce.py", "requirements.txt",
                                     "data/games.json", "data/reference-checks.json")]
        metadata = dict(python=platform.python_version(), numpy=np.__version__, sympy=sp.__version__,
                        mpmath=mpmath.__version__,
                        regenerated_sample=not args.saved_only,
                        sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
        (args.output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("PASS: 20 uniform + 20 weighted games, and both Figure 1 games.")
    print("All 42: dim U = 3; rank Dv(q) = 2; restricted entropic score rank = 2.")
    print("Exact agreement with original payoff tables and archived calculations.")
    if not args.saved_only:
        print("Original seed, all random draws, and solved payoffs reproduced exactly.")


if __name__ == "__main__":
    main()
