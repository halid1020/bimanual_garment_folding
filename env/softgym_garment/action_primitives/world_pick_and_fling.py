## This code is highly referenced from https://github.com/real-stanford/cloth-funnels/blob/main/cloth_funnels/environment/simEnv.py#L1523
import numpy as np

from .world_position_with_velocity_and_grasping_control \
    import WorldPositionWithVelocityAndGraspingControl


## This action primitive should be decoupled from the environment.

class WorldPickAndFling():

    def __init__(self, 
        lowest_cloth_height=0.1,
        max_grasp_dist=0.7,
        stretch_increment_dist=0.02,
        fling_vel=8e-3,
        pregrasp_height=0.3,
        pregrasp_vel=0.06,
        tograsp_vel=0.005,
        hang_height=0.7,#
        prefling_vel=6e-3,
        stroke=0.5,
        hang_pos_y=0,
        lift_vel=0.009,
        hang_adjust_vel=0.005,
        stretch_adjust_vel=0.005,
        release_vel=1e-2,
        ready_pos = [[-1, 0, 0.6], [1, 0, 0.6]],
        no_cloth_vel = 0.3,
        adaptive_fling_momentum=1.3,
        place_height=0.015,
        action_horizon=20,

        **kwargs):

        # Only support 2-picker pick-and-fling
        
        ### Environment has to be WorldPickAndFlingWrapper
        self.action_tool = WorldPositionWithVelocityAndGraspingControl(**kwargs)
        

        #### Define the action space
        self.kwargs = kwargs
        self.action_mode = 'world-pick-and-place'
        self.action_horizon = action_horizon
        self.logger_name = 'standard_logger'
        self.stroke = stroke


        self.lowest_cloth_height = lowest_cloth_height
        self.max_grasp_dist = max_grasp_dist
        self.stretch_increment_dist = stretch_increment_dist
        self.fling_vel = fling_vel
        self.pregrasp_height = pregrasp_height
        self.pregrasp_velocity = pregrasp_vel
        self.tograsp_velocity = tograsp_vel
        self.hang_height = hang_height
        self.prefling_vel = prefling_vel
        self.hang_pos_y = hang_pos_y
        self.ready_pos = np.asarray(ready_pos)
        self.lift_vel = lift_vel
        self.hang_adjust_vel = hang_adjust_vel
        self.stretch_adjust_vel = stretch_adjust_vel
        self.no_cloth_vel = no_cloth_vel
        self.adaptive_fling_momentum = adaptive_fling_momentum
        self.place_height = place_height
        self.num_picker = 2
        self.grasping = False
        # self.action_step = 0
    
    def get_no_op(self):
        return self.ready_pos
        
    def sample_random_action(self):
        return self.action_space.sample()

    def get_action_space(self):
        return self.action_space
    
    def get_action_horizon(self):
        return self.action_horizon


    def reset(self, env):
        # self.action_step = 0
        
        
        return self.reset_pickers(env)
        
    
    # def get_step(self):
    #     return self.action_step
    

    def process(self, action):
       
        return {'pick_0_position': action['pick_0_position'],
                'pick_1_position': action['pick_1_position'],
                'stroke': (action['stroke'] if 'stroke' in action.keys() else self.stroke),
                'fling_vel': (action['fling_vel'] if 'fling_vel' in action.keys() else self.fling_vel),
                'pregrasp_height': (action['pregrasp_height'] if 'pregrasp_height' in action.keys() else self.pregrasp_height),
                'pregrasp_vel': (action['pregrasp_vel'] if 'pregrasp_vel' in action.keys() else self.pregrasp_velocity),
                'tograsp_vel': (action['tograsp_vel'] if 'tograsp_vel' in action.keys() else self.tograsp_velocity),
                'hang_height': (action['hang_height'] if 'hang_height' in action.keys() else self.hang_height),
                'prefling_vel': (action['prefling_vel'] if 'prefling_vel' in action.keys() else self.prefling_vel),
                'hang_pos_y': (action['hang_pos_y'] if 'hang_pos_y' in action.keys() else self.hang_pos_y),
                'drag_dist': (action['drag_dist'] if 'drag_dist' in action.keys() else self.drag_dist),
                'lift_vel': (action['lift_vel'] if 'lift_vel' in action.keys() else self.lift_vel),
                'hang_adjust_vel': (action['hang_adjust_vel'] if 'hang_adjust_vel' in action.keys() else self.hang_adjust_vel),
                'stretch_adjust_vel': (action['stretch_adjust_vel'] if 'stretch_adjust_vel' in action.keys() else self.stretch_adjust_vel),
                'release_vel': (action['release_vel'] if 'release_vel' in action.keys() else self.release_vel),
                'drag_vel': (action['drag_vel'] if 'drag_vel' in action.keys() else self.drag_vel),
                'place_height': (action['place_height'] if 'place_height' in action.keys() else self.place_height),
        }
    
    ## expect to recieve action as an dictionary with at least pick_0 and pick_1 positions
    def step(self, env, action):
        action = self.process(action)

        pick_0_position = action['pick_0_position']
        pick_1_position = action['pick_1_position']

        # Fix: np.align → np.linalg
        if np.linalg.norm(pick_0_position - pick_1_position) < 0.1:
            print('[WorldPickAndFling] Reject Pick and Fling, Two picks are too close.')
            return {}

        
        
        
        pick_positions = np.stack([pick_0_position, pick_1_position], axis=0)
        #pick_vels = np.array([action['tograsp_vel']]*self.num_picker).reshape(self.num_picker, -1)
        
        pregrasp_positions = pick_positions.copy()
        pregrasp_positions[:, 2] = action['pregrasp_height']
 
        lift_positions = pick_positions.copy()
        lift_positions[:, 2] = action['hang_height']
      
        grasp_dist = np.linalg.norm(pick_positions[0, :2] - pick_positions[1, :2])
        prefling_positions = lift_positions.copy()
        prefling_positions[:, 1] = action['hang_pos_y']
        prefling_positions[0, 0] = -grasp_dist/2 + env.hard_shift_x
        prefling_positions[1, 0] = grasp_dist/2 + env.hard_shift_x
 
        ## go to pregrasp position
        info = self.action_tool.movep(env, pregrasp_positions, self.no_cloth_vel)
        ## lower the picker
        info = self.action_tool.movep(env, pick_positions, action['tograsp_vel'])
        #print('pick_positions', pick_positions)
        ## pick the object
        info = self.action_tool.both_grasp(env)
        ## lift the cloth to the prefling height
        info = self.action_tool.movep(env, lift_positions, action['lift_vel'])
        ## go to fling position
        info = self.action_tool.movep(env, prefling_positions, action['prefling_vel'])

        #return self._process_info(info)

        
        ## continue to lift until the the lowest partcile of the cloth is above the minumn height
        hang_height, info = self.hang_cloth(env,
            prefling_positions[0, 2], grasp_dist, action['hang_pos_y'], 
            adjust_vel=action['hang_adjust_vel'])
        
        

        ## stretch the cloth until the cloth is stable
        stretch_dist, info = self.stretch_cloth(env, hang_height, grasp_dist, action['hang_pos_y'], 
            adjust_vel=action['stretch_adjust_vel'], max_grasp_dist=self.max_grasp_dist,
            increment_dist=self.stretch_increment_dist)
        
        info = env.wait_until_stable(max_wait_step=50)

        print('WorldPickAndFling', action)
        self.fling(env, stretch_dist, action['hang_pos_y'], hang_height, place_height=action['place_height'],
                   fling_vel=action['fling_vel'], release_vel=action['release_vel'], 
                   drag_vel=action['drag_vel'], stroke=action['stroke'], 
                   drag_dist=action['drag_dist'])

        info = self.reset_pickers(env)

        env.wait_until_stable()

        # self.action_step += 1
        # done = self.action_step >= self.action_horizon
        # info['done'] = done
        return info
    
    def reset_pickers(self, env):
        
        self.action_tool.movep(env, self.ready_pos, self.no_cloth_vel)
        return self.action_tool.open_both_gripper(env)
    

    ## ALERT!!! this cannot be done in real world, as it reqruiest the get the information of the cloth
    ## This mean this is a research direction.
    def fling(self, env, grasp_dist, hang_pos_y, hang_height, place_height, 
              fling_vel=8e-3, release_vel=1e-2, drag_vel=5e-3, 
              stroke=0.65, drag_dist=0.1):
        cloth_positions = env.get_particle_positions()

        # max height - min height
        cloth_height = np.max(cloth_positions[:, 2]) - np.min(cloth_positions[:, 2])

        # Fling the cloth
        # stroke = 0.3
        left_x = -grasp_dist/2 + env.hard_shift_x
        righ_x = grasp_dist/2 + env.hard_shift_x

        back_pre_fling_pos = np.array([
            [left_x, hang_pos_y+drag_dist, hang_height],
            [righ_x, hang_pos_y+drag_dist, hang_height]
        ])

        front_fling_pos = np.array([
            [left_x, hang_pos_y-stroke, hang_height],
            [righ_x, hang_pos_y-stroke, hang_height]
        ])
        back_fling_pos = np.array([
            [left_x, hang_pos_y, place_height], #0.05
            [righ_x, hang_pos_y, place_height]
        ])
        
        #print('fling_vel', fling_vel)
        #info = self.action_tool.movep(env, back_fling_pos, fling_vel/2)
        info = self.action_tool.movep(env, back_pre_fling_pos, fling_vel)
        info = self.action_tool.movep(env, front_fling_pos, fling_vel)
        info = self.action_tool.movep(env, back_fling_pos, fling_vel)
        

        # # FIX: Drop the cloth at 90% of the distance, so there is room to drag it
        # release_y = (stroke-0.2) * self.adaptive_fling_momentum
        # drag_y = (stroke-0.29) * self.adaptive_fling_momentum
        
        # # 1. Lower to the table at the closer release_y
        # self.action_tool.movep(env, [[left_x,  release_y, place_height],
        #             [righ_x, release_y, place_height]], release_vel)
        
        # 2. Drag the cloth horizontally to drag_y to pull it taut and flatten it
        self.action_tool.movep(env,[[left_x, drag_dist, place_height],
                    [righ_x, drag_dist, place_height]], drag_vel)
        
        # release the cloth
        info = self.action_tool.open_both_gripper(env)

        # move up the picker a bit
        self.action_tool.movep(env,[[left_x, drag_dist, place_height+0.1],
                    [righ_x, drag_dist, place_height+0.1]], drag_vel)
        
        return info

    ## ALERT!!! this cannot be done in real world, as it reqruiest the get the information of the cloth
    ## This mean this is a research direction.
    def stretch_cloth(self, env, hang_height, grasp_dist, hang_pos_y, adjust_vel=0.001, increment_dist=0.02, max_grasp_dist=0.7):

        ## get picker positions
        picker_pos = env.get_picker_position()

        mid_pos = (picker_pos[0] + picker_pos[1])/2
        mid_pos[2] = hang_height

        direction = (picker_pos[0] - picker_pos[1])/np.linalg.norm(picker_pos[0] - picker_pos[1])

        left_x = -grasp_dist/2 + env.hard_shift_x
        righ_x = grasp_dist/2 + env.hard_shift_x

        hang_positios = np.array([
            [left_x, hang_pos_y, hang_height],
            [righ_x, hang_pos_y, hang_height]])

        info = self.action_tool.movep(env, hang_positios, adjust_vel)
        picker_mid_pos = (picker_pos[0] + picker_pos[1])/2
        max_steps = 100
        stretch_steps = 0
        stable_steps = 0
        while True:
            cloth_positions = env.get_mesh_particles_positions()  # (N, 3)

            high_cloth_positions = cloth_positions[cloth_positions[:, 2] > hang_height - 0.1]
            
            if len(high_cloth_positions) > 0:
                # compute XY distances to picker midpoint
                xy_dists = np.linalg.norm(high_cloth_positions[:, :2] - picker_mid_pos[:2], axis=1)

                # get indices of the 10 nearest particles in XY
                k = min(10, len(high_cloth_positions))
                nearest_idxs = np.argpartition(xy_dists, k-1)[:k]

                if len(nearest_idxs) > 0:
                    # subset to those particles only
                    candidate_particles = high_cloth_positions[nearest_idxs]

                    # now sort those candidates by 3D distance to picker_mid_pos
                    sorted_cloth_positions = candidate_particles[np.argsort(np.linalg.norm(candidate_particles - picker_mid_pos, axis=1))]

                    # pick the closest one
                    closest_cloth_particle_to_picker_mid = sorted_cloth_positions[0]
                    dist = np.linalg.norm(closest_cloth_particle_to_picker_mid - picker_mid_pos)

                    stable = dist < 0.07  # top is within 7 cm
                    # print('stable', stable)
                    # print('picker_mid_pos', picker_mid_pos)
                    if stable:
                        stable_steps += 1
                    else:
                        stable_steps = 0

                    stretched = stable_steps > 1
                    if stretched:
                        #print('break because of stable')
                        break
                    if stretch_steps > max_steps:
                        break


            stretch_steps += 1
            
            #cloth_mid_pos = new_cloth_mid_pos
            grasp_dist += increment_dist
            #print('streching grasp dist', grasp_dist)

            left_pos = mid_pos + direction*righ_x
            right_pos = mid_pos - direction*righ_x

            info = self.action_tool.movep(env, np.stack([left_pos, right_pos]), adjust_vel)

            if grasp_dist > max_grasp_dist:
                #print('break because reaching max grasp step')
                break

        return grasp_dist, info


    ## ALERT!!! this cannot be done in real world, as it reqruiest the get the information of the cloth
    ## This mean this is a research direction.
    def hang_cloth(self, env, hang_height, grasp_dist, hang_pos_y, adjust_vel=0.001):
        
        left_x = -grasp_dist/2 + env.hard_shift_x
        righ_x = grasp_dist/2 + env.hard_shift_x

        hang_positions = np.array([
            [left_x, hang_pos_y, hang_height],
            [righ_x, hang_pos_y, hang_height]])

        info = self.action_tool.movep(env, hang_positions, adjust_vel)
        max_steps = 100
        hang_steps = 0 
        while hang_steps > max_steps:
            #print('adjusting height')
            positions = env.get_particle_positions()
            min_z = np.min(positions[:, 2])
            #print('min cloth height', min_z)
            if min_z > self.lowest_cloth_height + 0.05:
                hang_height -= 0.05
            elif min_z < self.lowest_cloth_height - 0.05:
                hang_height += 0.05
            else:
                break
            hang_positions[:, 2] = hang_height
            
            info = self.action_tool.movep(env, hang_positions, adjust_vel)
            hang_steps += 1
        
        return hang_height, info