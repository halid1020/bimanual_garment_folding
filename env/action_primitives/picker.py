import abc
import numpy as np
from gym.spaces import Box

import pyflex
from enum import Enum
from scipy.spatial.distance import cdist

class Picker():
    class Status(Enum):
        PICK = 0
        HOLD = 1
        PLACE = 2
    
    def get_action_space(self):
        return self.action_space

    def __init__(self, num_picker=1, picker_radius=0.05, init_pos=(0., -0.1, 0.), 
        picker_threshold=0.007, particle_radius=0.05, picker_low=(-1, 0., -1), 
        picker_high=(1, 1.0, 1), init_particle_pos=None, spring_coef=1.2, save_step_info=False, grasp_mode={'closest': 1.0}, **kwargs):
        
        """
        :param gripper_type:
        :param sphere_radius:
        :param init_pos: By default below the ground before the reset is called
        """

        # super(Picker).__init__()
        # logging.info('[softgym, picker]  picker threshold: {}'.format(picker_threshold))
        
        self.set_save_step_info(save_step_info)
        

        self.picker_radius = picker_radius
        self.picker_threshold = picker_threshold
        self.num_picker = num_picker
        self.picked_particles = [[] for _ in range (self.num_picker)]
        self.picker_low, self.picker_high = np.array(list(picker_low)).astype(np.float32), np.array(list(picker_high)).astype(np.float32)
        self.grasp_mode = grasp_mode
        
        self.init_pos = init_pos
        self.particle_radius = particle_radius
        self.init_particle_pos = init_particle_pos
        self.spring_coef = spring_coef  # Prevent picker to drag two particles too far away

        space_low = np.array([-0.1, -0.1, -0.1, -10] * self.num_picker) * 0.1  # [dx, dy, dz, [-1, 1]]
        space_high = np.array([0.1, 0.1, 0.1, 10] * self.num_picker) * 0.1
        self.action_space = Box(space_low.astype(np.float32), 
                                space_high.astype(np.float32), dtype=np.float32)
    
    def set_save_step_info(self, save_step_info):
        self.save_step_info=save_step_info
        if self.save_step_info:
            self.step_info = {
                'control_signal': [],
                'particle_pos': [],
                'picker_pos': [],
                'rgbd': []
            }

    # def update_picker_boundary(self, picker_low, picker_high):
    #     self.picker_low, self.picker_high = np.array(picker_low).copy(), np.array(picker_high).copy()

    def visualize_picker_boundary(self):
        halfEdge = np.array(self.picker_high - self.picker_low) / 2.
        center = np.array(self.picker_high + self.picker_low) / 2.
        quat = np.array([1., 0., 0., 0.])
        pyflex.add_box(halfEdge, center, quat)

    def _apply_picker_boundary(self, picker_pos):
        
        return np.clip(picker_pos, self.picker_low, self.picker_high)

    def _get_centered_picker_pos(self, center):
        r = np.sqrt(self.num_picker - 1) * self.picker_radius * 2.
        pos = []
        for i in range(self.num_picker):
            x = center[i, 0] + np.sin(2 * np.pi * i / self.num_picker) * r
            y = center[i, 1]
            z = center[i, 2] + np.cos(2 * np.pi * i / self.num_picker) * r
            pos.append([x, y, z])
        return np.array(pos)

    def reset(self, picker_pos):
        
        if self.save_step_info:
            self.clean_step_info()

        # for i in (0, 2):
        #     offset = center[i] - (self.picker_high[:, i] + self.picker_low[:, i]) / 2.
        #     self.picker_low[:, i] += offset
        #     self.picker_high[:, i] += offset
        init_picker_poses = picker_pos #self._get_centered_picker_pos(center)

        for picker_pos in init_picker_poses:
            #print('!!!!add sphere')
            pyflex.add_sphere(self.picker_radius, picker_pos, [1, 0, 0, 0])
        pos = pyflex.get_shape_states()  # Need to call this to update the shape collision
        pyflex.set_shape_states(pos)

        self.picked_particles = [[] for _ in range (self.num_picker)]
        shape_state = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        #centered_picker_pos = self._get_centered_picker_pos(center)
        centered_picker_pos = init_picker_poses
        for (i, centered_picker_pos) in enumerate(centered_picker_pos):
            shape_state[i] = np.hstack([centered_picker_pos, centered_picker_pos, [1, 0, 0, 0], [1, 0, 0, 0]])
        pyflex.set_shape_states(shape_state)
        # pyflex.step() # Remove this as having an additional step here may affect the cloth drop env
        self.particle_inv_mass = pyflex.get_positions().reshape(-1, 4)[:, 3]
        # print('inv_mass_shape after reset:', self.particle_inv_mass.shape)

        self.last_grasp_mode = ['realease' for _ in range(self.num_picker)]
        self.graps_try_step = [0 for _ in range(self.num_picker)]

    # num_pickers * 14
    def get_picker_pos(self):
        #print('get picker pos', pyflex.get_shape_states())
        return np.array(pyflex.get_shape_states()).reshape(-1, 14)

    # num_pickers * 14
    def set_picker_pos(self, picker_pos):
        pyflex.set_shape_states(picker_pos)

    # num_pickers * 3
    def get_picker_position(self):
        return self.get_picker_pos()[:, :3]

    # num_pickers * 3
    def set_picker_position(self, picker_position):
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:, 3:6] = shape_states[:, :3]
        shape_states[:, :3] = picker_position
        self.set_picker_pos(shape_states)

    def get_particle_pos(self):
        return np.array(pyflex.get_positions()).reshape(-1, 4)

    @staticmethod
    def _get_pos():
        """ Get the current pos of the pickers and the particles, along with the inverse mass of each particle """
        picker_pos = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        particle_pos = np.array(pyflex.get_positions()).reshape(-1, 4)
        return picker_pos[:, :3], particle_pos

    @staticmethod
    def _set_pos(picker_pos, particle_pos):
        #print('picker_pos', picker_pos)
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:, 3:6] = shape_states[:, :3]
        shape_states[:, :3] = picker_pos
        pyflex.set_shape_states(shape_states)
        pyflex.set_positions(particle_pos)

    @staticmethod
    def set_picker_pos(picker_pos):
        """ Caution! Should only be called during the reset of the environment. Used only for cloth drop environment. """
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:, 3:6] = picker_pos
        shape_states[:, :3] = picker_pos
        pyflex.set_shape_states(shape_states)

    def clean_step_info(self):
        if self.save_step_info:
            self.step_info = {k: [] for k in self.step_info.keys()}
        
    def get_step_info(self):
        if self.save_step_info:
            return self.step_info.copy()
        else:
            raise NotImplementedError
        
    def step(self, action, arena):
        #print('grasp mode:', self.grasp_mode)
        action = np.reshape(action, (-1, 4))
        grip_flag = action[:, 3] < 0
        release_flag = (0 <= action[:, 3]) & (action[:, 3] <= 1)
        
        picker_pos, particle_pos = self._get_pos()
        mesh_particle_pos = arena.get_mesh_particles_positions()
        mesh_particle_pos[:, [1, 2]] = mesh_particle_pos[:, [2, 1]]

        new_picker_pos = self._apply_picker_boundary(picker_pos + action[:, :3])
        new_particle_pos = particle_pos.copy()

        # Release particles
        release_mask = np.zeros(self.num_picker, dtype=bool)
        for i in np.where(release_flag)[0]:
            if self.picked_particles[i]:
                release_mask[i] = True
                self.last_grasp_mode[i] = 'release'
                self.graps_try_step[i] = 0
                new_particle_pos[self.picked_particles[i], 3] = self.particle_inv_mass[self.picked_particles[i]]
                self.picked_particles[i] = []

        # Pick new particles
        pick_mask = grip_flag & (np.array([len(p) for p in self.picked_particles]) == 0)
        
        if np.any(pick_mask):
            pickers_to_pick = np.where(pick_mask)[0]
            dists = cdist(picker_pos[pickers_to_pick], mesh_particle_pos[:, :3])
            
            threshold = self.picker_threshold + self.picker_radius + self.particle_radius
            #print('threshold:', threshold)
            mask = dists <= threshold
            #print('grasp num', np.sum(mask, axis=1))
            
            for i, picker_idx in enumerate(pickers_to_pick):
                valid_particles = np.where(mask[i])[0]
                if len(valid_particles) > 0:
                    sorted_indices = np.argsort(dists[i, valid_particles])
                    candidate_particles = valid_particles[sorted_indices]
                    
                    mode = np.random.choice(list(self.grasp_mode.keys()), p=list(self.grasp_mode.values()))

                    if self.last_grasp_mode[i] == 'miss' and self.graps_try_step[i] < 40:
                        self.graps_try_step[i] += 1
                    elif mode == 'around':
                        self.picked_particles[picker_idx].extend(candidate_particles)
                        self.last_grasp_mode[i] = 'around'
                    elif mode == 'closest':
                        self.picked_particles[picker_idx].append(candidate_particles[0])
                        self.last_grasp_mode[i] = 'closest'
                    elif mode == 'miss':
                        self.last_grasp_mode[i] = 'miss'
                    else:
                        raise NotImplementedError

        # Update picked particle positions
        for i, particles in enumerate(self.picked_particles):
            if particles:
                displacement = new_picker_pos[i] - picker_pos[i]
                new_particle_pos[particles, :3] += displacement
                new_particle_pos[particles, 3] = 0  # Set mass to infinity
        
        
        self._set_pos(new_picker_pos, new_particle_pos)

        self._detect_over_stretching(arena, new_particle_pos)
       
                            
        return 1

    def _detect_over_stretching(self, arena, new_particle_pos):
        pairwise_dists = []
        overstretch = False

        for i in range(self.num_picker):
            for j in range(i + 1, self.num_picker):
                if self.picked_particles[i] and self.picked_particles[j]:
                    for p1 in self.picked_particles[i]:
                        for p2 in self.picked_particles[j]:
                            dist = np.linalg.norm(new_particle_pos[p1, :3] - new_particle_pos[p2, :3])
                            pairwise_dists.append((p1, p2, dist))

                            # Compare with rest distance from the arena’s distance matrix
                            rest_dist = arena.particle_dist_matrix[p1, p2]
                            #print('dist', dist, 'rest_dist', rest_dist)
                            if dist > 0.05 + rest_dist:  # threshold factor 5
                                overstretch = True
                                #print('over strech!')
                                break

        arena.over_strech = (arena.over_strech or overstretch)
        return pairwise_dists
        


        