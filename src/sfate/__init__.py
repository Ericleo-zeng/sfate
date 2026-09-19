"""scalable-fatesfate fate mapping M1: graphM2: fate """

from sfate.fate import absorption_probabilities
from sfate.graph import (
    MAX_LATENT_DIM,
    absorb_states,
    build_transition_graph,
    split_transient_absorbing,
)

__all__ = [
    "MAX_LATENT_DIM",
    "absorb_states",
    "absorption_probabilities",
    "build_transition_graph",
    "split_transient_absorbing",
]
