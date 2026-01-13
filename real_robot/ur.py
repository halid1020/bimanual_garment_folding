import rtde_control
import rtde_receive
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation

from rg2_gripper import RG2

def path_avoid_singularity(path, radius=0.25, 
        detour_ratio=np.sqrt(2)):
    if np.min(np.linalg.norm(path[:,:2], axis=-1)) < radius:
        raise RuntimeError('Path within singularity radius.')

    result_path = list()
    for i in range(len(path)-1):
        start_pose = path[i]
        end_pose = path[i+1]
        result_path.append(start_pose)

        # decide if linear path intersects with singularity cylindar
        pstart = start_pose[:2]
        pend = end_pose[:2]
        diff = pend - pstart
        length = np.linalg.norm(diff)

        direction = diff / length
        to_origin = -pstart
        proj_dist = np.dot(to_origin, direction)
        if 0 < proj_dist < length:
            nn_point = proj_dist * direction + pstart
            origin_dist = np.linalg.norm(nn_point)
            if origin_dist < radius:
                # need to insert waypoint
                detour_point = nn_point / origin_dist * (detour_ratio * radius)
                pos_interp = interp1d([0, length], [start_pose[:3], end_pose[:3]], axis=0)
                rot_interp = Slerp([0,length], 
                    rotations=Rotation.from_rotvec([
                        start_pose[-3:],
                        end_pose[-3:]
                    ]))
                pos = pos_interp(proj_dist)
                pos[:2] = detour_point
                rot = rot_interp(proj_dist).as_rotvec()
                detour_pose = np.zeros(6)
                detour_pose[:3] = pos
                detour_pose[-3:] = rot
                result_path.append(detour_pose)
        result_path.append(end_pose)
    result_path = np.array(result_path)
    return result_path

class UR_RTDE:
    def __init__(self, ip, gripper=None):
        self.rtde_c = rtde_control.RTDEControlInterface(ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ip)
        self.rtde_i = None

        
        if gripper == 'rg2':
            self.gripper =  RG2(ip,0)

        #self.gripper = gripper
        self.home_joint = [np.pi/2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0]
        self.camera_state_joint = [math.radians(a) for a in [73.4,  -78.7,  0,   -29.7,  -82.3,  79.0]]
        if self.gripper is None:
            self.rtde_c.setTcp([0, 0, 0, 0, 0, 0])
        elif gripper == 'rg2':
            # self.rtde_c.setTcp([0, 0, 0.195, 0, 0, 0])
            # self.rtde_c.setPayload(1.043, [0, 0, 0.08])
            pass
        else:
            self.rtde_c.setTcp(self.gripper.tool_offset)
            self.rtde_c.setPayload(self.gripper.mass, [0, 0, 0.08])
    
    def __del__(self):
        self.disconnect()
    
    def disconnect(self):
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()
        if hasattr(self.gripper, 'disconnect'):
            self.gripper.disconnect()

    def home(self, speed=1.5, acceleration=1, blocking=True):
        return self.rtde_c.moveJ(self.home_joint, speed, acceleration, not blocking)
    
    def camera_state(self, speed=1.5, acceleration=1, blocking=True):
        return self.rtde_c.moveJ(self.camera_state_joint, speed, acceleration, not blocking)

    def movej(self, q, speed=1.5, acceleration=1, blocking=True):
        return self.rtde_c.moveJ(q, speed, acceleration, not blocking)
    
    def movel(self, p, speed=1.5, acceleration=1, blocking=True, avoid_singularity=False):
        # nomralize input format to 2D numpy array
        if not isinstance(p, np.ndarray):
            p = np.array(p)
        if len(p.shape) == 1:
            p = p.reshape(1,-1)
        
        if avoid_singularity:
            path = np.concatenate([
                self.get_tcp_pose().reshape(-1,6),
                p],axis=0)
            new_path = path_avoid_singularity(path)
            p = new_path[1:]

        if p.shape[0] == 1:
            return self.rtde_c.moveL(p[0].tolist(), speed, acceleration, not blocking)
        else:
            p = p.tolist()
            for x in p:
                x.extend([speed, acceleration, 0])
            return self.rtde_c.moveL(p, not blocking)
    
    def movej_ik(self, p, speed=1.5, acceleration=1, blocking=True):
        return self.rtde_c.moveJ_IK(p, speed, acceleration, not blocking)

    def open_gripper(self, sleep_time=1):
        self.gripper.open()
        
    def close_gripper(self, sleep_time=1):
        self.gripper.close()
    
    def start_force_mode(self):
        class ForceModeGuard:
            def __init__(self, rtde_c):
                self.rtde_c = rtde_c
                self.enabled = False
            
            def __enter__(self):
                self.enabled=True
                return self
            
            def __exit__(self, type, value, traceback):
                if value is not None:
                    print(value, traceback)
                self.rtde_c.forceModeStop()
                self.enabled = False
                return True

            def apply_force(self, task_frame, selection_vector, wrench, type, limits):
                if not self.enabled:
                    return False
                return self.rtde_c.forceMode(task_frame, selection_vector, wrench, type, limits)
        return ForceModeGuard(self.rtde_c)

    def get_tcp_pose(self):
        return np.array(self.rtde_r.getActualTCPPose())
    
    def get_tcp_speed(self):
        return np.array(self.rtde_r.getActualTCPSpeed())

    def get_tcp_force(self):
        return np.array(self.rtde_r.getActualTCPForce())