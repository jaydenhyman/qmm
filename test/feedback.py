"""Independent feedback expansion from the public cycle listing."""

from itertools import combinations

import sympy as sp

from qmm.core.helper import get_nodes
from qmm.extensions.paths import get_cycles


def cycle_expansion(G):
    cycles = get_cycles(G)
    n = len(get_nodes(G, "state"))
    feedback = [sp.Integer(-1)] + [sp.Integer(0)] * n
    counts = [1] + [0] * n
    for size in range(1, n + 1):
        for combination in combinations(zip(cycles["Cycle"], cycles["Product"]), size):
            covered = [node for cycle, _ in combination for node in cycle]
            if len(covered) == len(set(covered)):
                feedback[len(covered)] += (-1) ** (size + 1) * sp.prod([product for _, product in combination])
                counts[len(covered)] += 1
    return sp.Matrix(feedback).applyfunc(sp.expand), sp.Matrix(counts)
