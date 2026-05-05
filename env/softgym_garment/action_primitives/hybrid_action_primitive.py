import numpy as np
from gym.spaces import Dict

from .pixel_pick_and_fling import PixelPickAndFling
from .pixel_pick_and_place import PixelPickAndPlace

class HybridActionPrimitive():
    """
    An orchestrator class that maps higher-level hybrid actions to specific 
    underlying primitives (Pick & Place, Pick & Fling, or No Operation).
    """

    def __init__(self, **kwargs):
        self.num_pickers = 2
        self.no_op = {'no-operation': True} # Initialised to prevent attribute errors

        # 1. Standard Pick and Place
        self.np_pnp = PixelPickAndPlace(**kwargs)

        # 2. Pick and Fling
        fling_kwargs = kwargs.copy() # Use .copy() to prevent polluting original kwargs
        if fling_kwargs.get('apply_workspace', False):
            print('[HybridActionPrimitive] Adjusting parameters for workspace!')
            fling_kwargs.update({
                'hang_pos_y': 0,
                'fling_vel': 0.01,
                'stroke': 0.65,
                'hang_height': 0.35,
                'place_height': 0.05,
                'drag_dist': 0.1,
                'drag_vel': 0.005
                
            })
        self.np_pnf = PixelPickAndFling(**fling_kwargs)

        # 3. Pick and Place with specialized height/velocity constraints)
        pnp_kwargs = kwargs.copy()
        pnp_kwargs.update({
            'pregrasp_height': 0.2, 
            'post_pick_height': 0.2,
            'pre_place_height': 0.15,
            'place_height': 0.04,
            'lift_vel': 0.01
        })
        self.np_pnp = PixelPickAndPlace(**pnp_kwargs)
    
    def get_no_op(self):
        return self.no_op
        
    def sample_random_action(self):
        return self.get_action_space().sample()

    def get_action_space(self):
        # Combine the action spaces of the underlying primitives
        pnp_action_space = self.np_pnp.get_action_space()
        
        action_space = Dict({
            'norm-pixel-pick-and-place': pnp_action_space,
            # TODO: add the component of np_pnf
        })
        return action_space
    
    def reset(self, env):
        # Reset the physical pickers out of view
        self.np_pnf.reset(env)
        return env.get_info()
    
    def step(self, env, action):
        """
        Accepts an action dictionary and routes it to the correct primitive.
        If the sub-action is provided as a flat np.ndarray (num_picker, 2, 3), 
        it formats it into the expected key-value pairs before passing it down.
        """
        if 'norm-pixel-pick-and-fling' in action:
            act = action['norm-pixel-pick-and-fling']
            if isinstance(act, np.ndarray):
                act = {'pick_0': act[:2], 'pick_1': act[2:4]}
                
            info = self.np_pnf.step(env, act)
            info['applied_action'] = {'norm-pixel-pick-and-fling': info['applied_action']}

        elif 'norm-pixel-dual-pick-and-place' in action:
            act = action['norm-pixel-dual-pick-and-place']
            if isinstance(act, np.ndarray):
                act = {
                    'pick_0': act[:2],  'pick_1': act[2:4],
                    'place_0': act[4:6], 'place_1': act[6:8]
                }
                
            info = self.np_pnp.step(env, act)
            # Retain original behavior: map back to generic pick-and-place key
            info['applied_action'] = {'norm-pixel-pick-and-place': info['applied_action']}

        elif 'norm-pixel-single-pick-and-place' in action:
            act = action['norm-pixel-single-pick-and-place']
            if isinstance(act, np.ndarray):
                act = {'pick_0': act[:2], 'place_0': act[2:4]}
                
            info = self.np_pnp.step(env, act)
            info['applied_action'] = {'norm-pixel-single-pick-and-place': info['applied_action']}

        elif 'no-operation' in action:
            info = env.get_info(new=True)
            info['applied_action'] = action
            
        else:
            raise ValueError(f'Action not recognized. Received keys: {list(action.keys())}')

        return info