from pathlib import Path
import runpy
import unittest

import sympy as sp

from harmonic_games import binary_field, master_matrix, rational, solve_family
from reproduce import FAMILIES, rank_certificate, verify_archive

ROOT = Path(__file__).resolve().parents[1]


class ExactChecks(unittest.TestCase):
    def test_all_archived_games_and_seed(self):
        report, families = verify_archive()
        self.assertEqual(len(report["cases"]), 42)
        for family in families.values():
            self.assertEqual(family["master_equation_rank"], 7)
            self.assertEqual(family["payoff_family_dimension"], 17)

    def test_original_notebook_generator_agrees(self):
        for _, mu, _ in FAMILIES:
            ns = runpy.run_path(str(ROOT / "provenance/original_generator.py"),
                                init_globals={"skeleton": [2, 2, 2], "mu_values": mu})
            family = solve_family(mu)
            self.assertEqual(list(family.symbols), ns["dofs"])
            self.assertEqual(family.matrix, ns["A"])
            self.assertEqual(list(family.solution), ns["extracted_sol"])

    def test_master_equation_direct_payoffs(self):
        mu = [[1, 2], [1, 3], [1, 4]]
        A, profiles = master_matrix(mu)
        flat = sp.Matrix(range(24))
        direct = []
        for a in profiles:
            total = 0
            for i in range(3):
                for b, weight in enumerate(mu[i]):
                    alternative = a[:i] + (b,) + a[i + 1:]
                    total += weight * (flat[8 * i + profiles.index(a)]
                                       - flat[8 * i + profiles.index(alternative)])
            direct.append(total)
        self.assertEqual(A * flat, sp.Matrix(direct))

    def test_binary_order_and_sign(self):
        # Player 1 gains 1 from action 1; player 2 gains 2. No other dependence.
        x, field = binary_field([0, 0, 1, 1, 0, 2, 0, 2], 2)
        self.assertEqual(field, sp.Matrix([1, 2]))
        self.assertEqual(len(x), 2)

    def test_rational_rank_certificates(self):
        for M in (sp.Matrix([[1, 2, 3], [2, 4, 6]]), sp.zeros(3, 2),
                  sp.Matrix([[0, sp.Rational(1, 3)], [-sp.Rational(1, 3), 0]])):
            cert = rank_certificate(M)
            self.assertEqual(cert["rank"], M.rank())
            self.assertNotEqual(sp.Rational(cert["minor_determinant"]), 0)

    def test_reject_inexact_or_invalid_input(self):
        with self.assertRaises(TypeError):
            rational(0.5)
        with self.assertRaises(ValueError):
            master_matrix([[1, 0], [1, 1]])
        with self.assertRaises(ValueError):
            binary_field([0, 1], 2)


if __name__ == "__main__":
    unittest.main()
