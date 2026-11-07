"""Executed mathematical cells of the archived harmonic-game generator.

Mechanically extracted from cells 0–35 of generate_harmonic_game_clean.ipynb.
Original code is unchanged except for trailing-whitespace cleanup and that configuration cells 4 and 6 are replaced
by defaults which may be overridden through runpy.run_path(init_globals=...).
Jupyter outputs, notebook metadata, and later experimental cells are omitted.
"""
if "skeleton" not in globals():
    skeleton = [2, 2, 2]
if "mu_values" not in globals():
    mu_values = [[1, 1], [1, 1], [1, 1]]

# Original notebook cell 0
from sympy import *
import itertools as it
import numpy as np

# Original notebook cell 1
# Make symbols: methods to dynamically create as many as needed strings and Sympy symbols
# Shift is starting index; default 0

def make_strings(N,s, shift = 0):
	my_strings = []
	for i in range(shift, N + shift):
	    tmp_st = f'{s}{i}'
	    my_strings.append(tmp_st)
	return my_strings

def make_symbols(N,s, shift = 0):
	my_symbols = []
	for i in range(shift, N + shift):
	    tmp_st = f'{s}{i}'
	    globals()[tmp_st] = Symbol(tmp_st)
	    my_symbols.append(globals()[tmp_st])
	return my_symbols

# Original notebook cell 8
# N
numPlayers = len(skeleton)
players = range(numPlayers)

# -------------------------
# Number of action profiles
# -------------------------

# A
numPures = prod(skeleton)

# List of A_{-i} = for each player, number of action profiles of other players
numPuresMinus = [  int(numPures / skeleton[i]) for i in players ]

# -------------------------
# Number of payoff degrees of freedom
# -------------------------

# AN; number of payoff degrees of freedom
numPays = numPlayers * numPures

# sum_i A_i; number of harmonic measure dofs
numMeas = sum(skeleton)

# Original notebook cell 9
# Pure actions: list of N lists, each with Ai elements; pure actions of each player

pures_play = [ make_strings(skeleton[i], f'a{i}', shift = 1) for i in players ]

# Original notebook cell 10
# Pure profiles; cartesian product of pure actions of each player
# Returns one list with A = numPures elements; each element is a tuple of strings

pures = list(it.product(*pures_play))

# Original notebook cell 11
assert len(pures) == numPures

# Original notebook cell 12
# Pack mu in list of dicts

mu = [   dict(zip( pures_play[i], mu_values[i] )) for i in players    ]

# Original notebook cell 13
def print_payoff(payoff_dict, payoff_symbol):
    for i in players:
        for a in pures:
            print( f'{payoff_symbol}_{i}{a} = {payoff_dict[i][a]}' )
        print()

# Original notebook cell 15
# Pure profiles of other players
# Make list of N lists; the list pure_minus[i] contains the pure action profiles of players other than i
# Build taking the cartesian product of pure actions of all players other than i
# The size of the list pure_minus[i] is A_{-i} = \prod_{j \neq i} A_j

pures_minus = [ list( it.product( *( pures_play[:i] + pures_play[i+1:] ) ) ) for i in players ]

# Original notebook cell 16
for i in players:
    assert len(pures_minus[i]) == numPuresMinus[i]

# Original notebook cell 18
# Util to make (a_i, a_{-i}) given a_i and a_{-i} as tuple of strings (to be used as key for payoff dictionaries)
def make_pure(ai, a_minus_i, i):
    l = list(a_minus_i)
    return tuple(l[:i] + [ai] + l[i:])

# Original notebook cell 19
def split_pure(a, i):

    ai = a[i]

    l = list(a)
    a_minus_i = tuple( l[:i] + l[i+1:] )
    return ai, a_minus_i

# Original notebook cell 20
# check consistency
for i in players:
    for a in pures:
        ai, a_minus_i = split_pure(a, i)
        assert a == make_pure( ai, a_minus_i, i )

# Original notebook cell 22
# harmonic payoff degrees of freedom; will build harmonic payoff h out of these

# to be determined harmonic payoff of player i (as many as A)

h_sym = [make_symbols(numPures, f'h{i}', shift = 1) for i in players]

# Original notebook cell 23
h = [  dict(zip(pures , h_sym[i] ))  for i in players]

# Original notebook cell 25
# --------------------------------
# KEY: HARMONIC FUNCTION
# --------------------------------
def Hi(a, i):
    ai, a_minus_i = split_pure(a, i)
    return sum(   [   mu[i][bi] * ( h[i][a] - h[i][ make_pure(bi, a_minus_i, i) ] )    for bi in pures_play[i]   ]  )

# Original notebook cell 26
# --------------------------------
# KEY: HARMONIC SYSTEM in degrees of freedom h, that is payoff to be determined, prescribed the harmonic measure
# --------------------------------
system = [  sum( [  Hi(a,i)   for i in players ] )   for a in pures  ]

# Original notebook cell 27
# one equation per pure in pures
assert len(system) == numPures

# Original notebook cell 28
# Unknowns of system to solve: the weights (one per player) + the non-strategic dofs
dofs = list(it.chain(*h_sym))

# Original notebook cell 29
numDofs = len(dofs)

# Original notebook cell 30
# Unknowns of system to solve: harmonic weights
assert numDofs == numPays

# Original notebook cell 31
# Matrix of linear system
A, b = linear_eq_to_matrix(system, *dofs)

# Original notebook cell 32
sol = solve(system, dofs)

# Original notebook cell 33
# Replace sol in dofs

extracted_sol = [e.subs(sol) for e in dofs]

# Original notebook cell 35
# zip solution in dictionary with dofs
sol_dict = dict(zip(dofs, extracted_sol))
sol_dict
