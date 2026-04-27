import numpy as np

class WorldPositionWithVelocityAndGraspingControl():

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.grasping = [False, False]

    def step(self, env, actions):
        total_steps = 0
        info = {}
        
        for action in actions:
            pickers_position = env.get_picker_position()
            target_position = action[:, :3].copy()
            velocity = action[:, 3]

            delta = target_position - pickers_position
            distance = np.linalg.norm(delta, axis=1)
            num_step = np.ceil(np.max(distance / velocity)).astype(int) + 1

            delta /= num_step
            norm_delta = np.linalg.norm(delta, axis=1, keepdims=True)

            curr_pos = pickers_position.copy()

            for i in range(num_step):
                dist = np.linalg.norm(target_position - curr_pos, axis=1, keepdims=True)
                mask = dist < norm_delta
                delta = np.where(mask, target_position - curr_pos, delta)
                
                control_signal = np.hstack([delta, action[:, 4:5]])
                info = env.control_picker(control_signal, process_info=False)
                curr_pos += delta
                total_steps += 1
            
        info['total_control_steps'] = total_steps
        return info

    
    def movep(self, env, pos, vel):
        grasp_sign = 1 if not self.grasping else -1
        grasp_signs = np.where(self.grasping, -1, 1).reshape(2, 1)
        action =  np.concatenate(
            [
                pos, 
                np.array([vel]*2).reshape(2, -1),
                grasp_signs
            ],     
            axis=1
        )
        info = self.step(env, [action])
        return info
    
    def both_grasp(self, env):
        self.grasping[0], self.grasping[1] = True, True
        picker_pos = env.get_picker_position()
        #print('picker_pos before grasp', picker_pos)
        return self.movep(env, picker_pos, 0.1)
    
    def open_both_gripper(self, env):
        self.grasping[0], self.grasping[1] = False, False
        picker_pos = env.get_picker_position()
        return self.movep(env, picker_pos, 0.1)