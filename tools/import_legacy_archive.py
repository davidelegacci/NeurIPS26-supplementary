"""One-time provenance import; NOT needed for reproduction from this repository.

Select the 42 paragraph cases from the original 49-case audit. Keep exact
inputs and independently archived reference results, without machine paths
or unrelated experiments. Extract the executed notebook cells as plain Python.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_KEYS = (
    "name", "field", "center", "update_space_basis", "update_space_dimension",
    "center_jacobian", "center_jacobian_rank", "restricted_entropy_score_jacobian",
    "restricted_entropy_score_rank", "fails_from_every_interior_initialization",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    source = args.archive / "results.json"
    notebook = args.archive / "generate_harmonic_game_clean.ipynb"
    raw = json.loads(source.read_text())
    selected = [row for row in raw["cases"] if row["name"].startswith(
        ("figure1-", "uniform-222-", "weighted-222-"))]
    if len(selected) != 42 or raw["seed"] != 20261107:
        raise ValueError("Unexpected archive")
    games = []
    for row in selected:
        game = {"name": row["name"], "measures": row["measures"],
                "payoffs": [str(v) for v in row["payoffs"]]}
        if "drawn_payoff_parameters" in row:
            game["drawn_payoff_parameters"] = row["drawn_payoff_parameters"]
        games.append(game)
    data = ROOT / "data"
    provenance = ROOT / "provenance"
    if data.exists() or provenance.exists():
        raise FileExistsError("Import refuses to overwrite data or provenance")
    data.mkdir()
    provenance.mkdir()
    payload = dict(seed=raw["seed"], action_profiles=[[a, b, c] for a in (0, 1)
                   for b in (0, 1) for c in (0, 1)],
                   payoff_order="player-major, then lexicographic action profiles",
                   coordinate_convention="x_i is the probability of action 1 (the second action)",
                   games=games)
    (data / "games.json").write_text(json.dumps(payload, indent=2) + "\n")
    reference = dict(cases=[{k: row[k] for k in REFERENCE_KEYS} for row in selected])
    (data / "reference-checks.json").write_text(json.dumps(reference, indent=2) + "\n")
    nb = json.loads(notebook.read_text())
    header = '''"""Executed mathematical cells of the archived harmonic-game generator.

Mechanically extracted from cells 0–35 of generate_harmonic_game_clean.ipynb.
Original code is unchanged except for trailing-whitespace cleanup and that configuration cells 4 and 6 are replaced
by defaults which may be overridden through runpy.run_path(init_globals=...).
Jupyter outputs, notebook metadata, and later experimental cells are omitted.
"""
if "skeleton" not in globals():
    skeleton = [2, 2, 2]
if "mu_values" not in globals():
    mu_values = [[1, 1], [1, 1], [1, 1]]

'''
    blocks = [f"# Original notebook cell {i}\n" + "\n".join(
                  line.rstrip() for line in "".join(cell["source"]).splitlines()).rstrip()
              for i, cell in enumerate(nb["cells"][:36])
              if cell["cell_type"] == "code" and i not in (4, 6)]
    (provenance / "original_generator.py").write_text(header + "\n\n".join(blocks) + "\n")
    metadata = dict(original_run=args.archive.name,
                    original_environment={k: raw[k] for k in ("python", "numpy", "sympy")},
                    source_sha256={"results.json": hashlib.sha256(source.read_bytes()).hexdigest(),
                                   "generate_harmonic_game_clean.ipynb": hashlib.sha256(notebook.read_bytes()).hexdigest()},
                    original_source_sha256={Path(k).name: v for k, v in raw["hashes"].items()},
                    selected_cases=[g["name"] for g in games],
                    extraction="42 paragraph cases; all payoff entries converted to exact rational strings; reference matrices copied unchanged")
    (provenance / "source-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Imported 42 original games and executed generator cells; no local paths published.")


if __name__ == "__main__":
    main()
