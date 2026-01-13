from agent_arena import Agent
import random

from .random_pick_and_fling import RandomPickAndFling
from .random_pick_and_place import RandomPickAndPlace
from ..human.no_operation import NoOperation


class RandomMultiPrimitive(Agent):

    def __init__(self, config):
        super().__init__(config)

        # 🔁 MUST MATCH HumanMultiPrimitive
        self.primitive_names = [
            "norm-pixel-pick-and-fling",
            "norm-pixel-pick-and-place",
            "no-operation"
        ]

        self.primitive_instances = [
            RandomPickAndFling(config),
            RandomPickAndPlace(config),   # MUST be dual-picker compatible
            NoOperation(config)
        ]

        assert len(self.primitive_names) == len(self.primitive_instances)

    def reset(self, arena_ids):
        self.internal_states = {arena_id: {} for arena_id in arena_ids}

    def init(self, infos):
        pass

    def update(self, infos, actions):
        pass

    def single_act(self, state, update=False):
        """
        Sample a primitive uniformly and delegate action generation.
        """

        pid = random.randrange(len(self.primitive_instances))
        primitive_name = self.primitive_names[pid]
        primitive = self.primitive_instances[pid]

        action = primitive.single_act(state)

        return {
            primitive_name: action
        }

    def act(self, info_list, update=False):
        return [self.single_act(info) for info in info_list]
