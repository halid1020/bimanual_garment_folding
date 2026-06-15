from actoris_harena import Agent
import numpy as np

class SingleArmRandomPickAndPlace(Agent):

    def __init__(self, config):
        super().__init__(config)
        self.name = "single-random-pixel-pick-and-place"

    def single_act(self, state, update=False):
        mask = state['observation']['mask']
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        H, W = mask.shape

        mask_coords = np.argwhere(mask > 0)

        def random_norm_xy():
            return np.random.uniform(-1, 1, size=2)

        # ----------------------------
        # Sample pick points
        # ----------------------------
        if len(mask_coords) == 0:
            pick0 = random_norm_xy()
        else:
            p0 = mask_coords[np.random.randint(len(mask_coords))]
            
            # (row, col) → normalized (x, y)
            pick0 = np.array([
                p0[0] / W * 2 - 1,
                p0[1] / H * 2 - 1
            ], dtype=np.float32)

          

        # ----------------------------
        # Random place points
        # ----------------------------
        place0 = random_norm_xy()
        

        # ----------------------------
        # FLAT ACTION VECTOR
        # (MUST MATCH HUMAN POLICY)
        # ----------------------------
        action = np.concatenate([
            pick0,
            
            place0,
           
        ], axis=0)

        assert action.shape == (4,)

        return action

    def init(self, state):
        pass

    def update(self, state, action):
        pass
