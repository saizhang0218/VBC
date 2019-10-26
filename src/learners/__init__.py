from .q_learner_corridor_vbc import QLearner_corridor
from .q_learner_6h_vs_8z_vbc import QLearner_6h_vs_8z

REGISTRY = {}
REGISTRY["q_learner_corridor"] = QLearner_corridor
REGISTRY["q_learner_6h_vs_8z"] = QLearner_6h_vs_8z
