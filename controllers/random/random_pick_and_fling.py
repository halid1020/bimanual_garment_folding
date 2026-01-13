from agent_arena import Agent
import numpy as np

class RandomPickAndFling(Agent):

    def __init__(self, config):
        super().__init__(config)
        self.name = "random-pixel-pick-and-fling"

    def act(self, info_list, update=False):
        return [self.single_act(info) for info in info_list]

    def reset(self, arena_ids):
        self.internal_states = {arena_id: {} for arena_id in arena_ids}

    def single_act(self, state, update=False):
        mask = state['observation']['mask']
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        H, W = mask.shape
        mask_coords = np.argwhere(mask > 0)

        def random_norm_xy():
            return np.random.uniform(-1, 1, size=2)

        # ----------------------------
        # Sample picks
        # ----------------------------
        if len(mask_coords) == 0:
            pick0 = random_norm_xy()
            pick1 = random_norm_xy()
        else:
            p0 = mask_coords[np.random.randint(len(mask_coords))]
            p1 = mask_coords[np.random.randint(len(mask_coords))]

            # (row, col) → normalized (x, y)
            pick0 = np.array([
                p0[0] / W * 2 - 1,
                p0[1] / H * 2 - 1
            ], dtype=np.float32)

            pick1 = np.array([
                p1[0] / W * 2 - 1,
                p1[1] / H * 2 - 1
            ], dtype=np.float32)

        # ----------------------------
        # 🔑 FLAT ACTION VECTOR
        # (MUST MATCH HUMAN FLING)
        # ----------------------------
        action = np.concatenate([pick0, pick1], axis=0)

        assert action.shape == (4,)

        return action

    def init(self, state):
        pass

    def update(self, state, action):
        pass
