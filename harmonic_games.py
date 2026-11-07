"""Exact finite-game generator, extracted from the original harmonic notebook.

Payoffs are ordered by player, then lexicographically by pure action profile.
All symbolic calculations use rational arithmetic; no floating-point ranks.
"""
from dataclasses import dataclass
from itertools import product

import sympy as sp


def rational(value):
    """Accept integers or exact rational strings, never floating-point inputs."""
    if isinstance(value, (float, sp.Float)):
        raise TypeError("Use integers or rational strings, not floats")
    return sp.Rational(value)


def validate_measures(measures):
    mu = tuple(tuple(rational(v) for v in row) for row in measures)
    if not mu or any(len(row) < 2 or any(v <= 0 for v in row) for row in mu):
        raise ValueError("Each player needs at least two strictly positive action measures")
    return mu


def master_matrix(measures):
    """A u = 0: sum_{i,b_i} mu_i(b_i)[u_i(a)-u_i(b_i,a_-i)] = 0."""
    mu = validate_measures(measures)
    profiles = list(product(*(range(len(row)) for row in mu)))
    positions = {a: k for k, a in enumerate(profiles)}
    m = len(profiles)
    matrix = sp.zeros(m, len(mu) * m)
    for row, a in enumerate(profiles):
        for i, weights in enumerate(mu):
            for b, weight in enumerate(weights):
                alternative = a[:i] + (b,) + a[i + 1:]
                matrix[row, i * m + row] += weight
                matrix[row, i * m + positions[alternative]] -= weight
    return matrix, profiles


@dataclass(frozen=True)
class HarmonicFamily:
    measures: tuple
    matrix: sp.Matrix
    symbols: tuple
    solution: tuple

    @property
    def free_indices(self):
        free = set().union(*(expr.free_symbols for expr in self.solution))
        return tuple(k for k, symbol in enumerate(self.symbols) if symbol in free)

    def sample(self, rng):
        # The archive draws for ALL payoff symbols, then substitutes into the
        # solved expressions. Only the 17 free-symbol draws affect each 222 game.
        # Keeping all 24 draws is essential to reproduce the original RNG stream.
        draws = [int(v) for v in rng.integers(-10, 10, len(self.symbols))]
        substitution = dict(zip(self.symbols, draws))
        payoffs = [expr.subs(substitution) for expr in self.solution]
        if self.matrix * sp.Matrix(payoffs) != sp.zeros(self.matrix.rows, 1):
            raise AssertionError("Generated payoffs violate a pure-profile equation")
        return payoffs, draws


def solve_family(measures):
    """Same symbol names/order and SymPy solve convention as notebook cells 22–33."""
    mu = validate_measures(measures)
    matrix, profiles = master_matrix(mu)
    symbols = tuple(sp.Symbol(f"h{i}{k}")
                    for i in range(len(mu)) for k in range(1, len(profiles) + 1))
    solution_map = sp.solve(list(matrix * sp.Matrix(symbols)), symbols)
    solution = tuple(symbol.subs(solution_map) for symbol in symbols)
    if matrix * sp.Matrix(solution) != sp.zeros(matrix.rows, 1):
        raise AssertionError("Symbolic solution does not solve the harmonic system")
    return HarmonicFamily(mu, matrix, symbols, solution)


def binary_field(payoffs, n):
    """v_i = u_i(1,x_-i)-u_i(0,x_-i), with x_i = P(action 1)."""
    flat = [rational(v) for v in payoffs]
    profiles = list(product((0, 1), repeat=n))
    if len(flat) != n * len(profiles):
        raise ValueError("Payoff table does not match the binary game skeleton")
    index = {a: k for k, a in enumerate(profiles)}
    x = sp.symbols(f"x1:{n+1}")
    field = []
    for i in range(n):
        value = sp.S.Zero
        for a in profiles:
            if a[i] == 1:
                continue
            b = a[:i] + (1,) + a[i + 1:]
            probability = sp.prod(x[j] if a[j] else 1 - x[j]
                                  for j in range(n) if j != i)
            value += (flat[i * len(profiles) + index[b]]
                      - flat[i * len(profiles) + index[a]]) * probability
        field.append(sp.expand(value))
    return x, sp.Matrix(field)
