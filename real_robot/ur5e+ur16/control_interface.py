#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import json

from rcc_msgs.msg import NormPixelPnP, Observation, Reset, WorldPnP
from std_msgs.msg import Header
from cv_bridge import CvBridge

from utils import *

class ControlInterface(Node):
    def __init__(self, task, steps=20, name='control_interface',
                 adjust_pick=False, adjust_orien=False, video_device='/dev/video6', save_dir='.'):
        super().__init__(f"{name}_interface")
        self.img_sub = self.create_subscription(Observation, '/observation', self.img_callback, 10)
        self.pnp_pub = self.create_publisher(NormPixelPnP, '/norm_pixel_pnp', 10)
        self.reset_pub = self.create_publisher(Header, '/reset', 10)
        self.resolution = (256, 256)
        self.fix_steps = steps
        self.mask_generator = get_mask_generator()
        self.task = task
        self.save_dir = save_dir
      
        self.step = -1
        self.last_action = None
        self.trj_name = ''
        self.adjust_pick = adjust_pick
        self.adjust_orient = adjust_orien
        self.video_device = video_device
        self.video_process = None

        if 'folding' in task:
            self.collect_demo = False
            self.demo_states = []
        #self.estimate_pick_depth = estimate_pick_dpeth

        print('Finish Init Control Interface')

    def publish_action(self, pnp):
        data = pnp['pick-and-place'] #.reshape(4)
        pnp_msg = NormPixelPnP()
        pnp_msg.header = Header()
        pnp_msg.header.stamp = self.get_clock().now().to_msg()
        pnp_msg.data = (data[0], data[1], data[2], data[3])
        pnp_msg.degree = pnp['orientation']
        pnp_msg.place_height = 0.03
        if 'folding' in self.task:
            pnp_msg.place_height = 0.02

        self.pnp_pub.publish(pnp_msg)


    def publish_reset(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        self.reset_pub.publish(header)
        print('Published reset!')

    def _save_step(self, state, save_dir):
        rgb = state['observation']['rgb']
        raw_rgb = state['observation']['raw_rgb']
        full_mask = state['observation']['full_mask']
        depth = state['observation']['depth']
        mask = state['observation']['mask']

        os.makedirs(save_dir, exist_ok=True)
        print('rgb shape', rgb.shape)
        save_color(rgb, filename='rgb', directory=save_dir)
        save_color(raw_rgb, filename='raw_rgb', directory=save_dir)
        save_depth(depth, filename='depth', directory=save_dir)
        save_depth(depth, filename='colour_depth', directory=save_dir, colour=True)
        save_mask(mask, filename='mask', directory=save_dir)
        save_mask(full_mask, filename='full_mask', directory=save_dir)

        if 'action' in state:
            action = state['action']
            action_image = state['action_image']
            print('action image', action_image.shape)
            save_color(action_image, filename='action_image', directory=save_dir)
            # cv2.imwrite(f'{save_dir}/action_image.png', action_image)
            with open(f'{save_dir}/action.json', "w") as json_file:
                json.dump(action, json_file, indent=4)
            #np.save(f'{save_dir}/action.npy', action)
        
        if 'evaluation' in state:
            evaluation = state['evaluation']
            with open(f'{save_dir}/evaluation.json', "w") as json_file:
                json.dump(evaluation, json_file, indent=4)

    def save_step(self, state):
        
        #color_depth = cv2.applyColorMap(np.uint8(255 * depth), cv2.COLORMAP_AUTUMN)
        save_dir = os.path.join(self.save_dir, self.trj_name, f'step_{str(self.step)}')

        self._save_step(state, save_dir)
        

    def evaluate(self, state):
        if self.task == 'flattening':
            current_mask = state['observation']['full_mask']
            cur_mask_pixels = int(np.sum(current_mask))
            print('current_mask', cur_mask_pixels)

            if self.step == 0:
                self.init_mask_pixels = cur_mask_pixels

            res = {
                'max_coverage': self.max_mask_pixels,
                'init_coverage': self.init_mask_pixels,
                'coverage': cur_mask_pixels,
                'normalised_coverage': 1.0 * cur_mask_pixels/self.max_mask_pixels,
                'normalised_improvement': max(min(1.0*(cur_mask_pixels - self.init_mask_pixels)\
                                                /(self.max_mask_pixels - self.init_mask_pixels), 1), 0),
                'auto success': bool(1.0 * cur_mask_pixels/self.max_mask_pixels > 0.95),
                'human success': False
            }

            return res
        elif 'folding' in self.task:
            IoU, matched_mask = get_IoU(state['observation']['mask'], self.goal_mask)
            save_mask(matched_mask, filename='matched_mask', directory='tmp')
            return {
                'human success': False, 
                'auto success': bool(IoU > 0.9),
                'IoU': IoU}
    
    def setup_evaluation(self, state):
        if self.task == 'flattening':
            current_mask = state['observation']['mask']
            self.max_mask_pixels = int(np.sum(current_mask))
            save_dir = os.path.join(self.save_dir, self.trj_name, 'goals', f'step_0')
            self._save_step(state, save_dir)
            
        elif 'folding' in self.task:
            self.demo_states.append(state)
        
            while True:
                finish_demo = input('[User Attention!] Finish the demonstration? (y/n): ')
                if finish_demo == 'n':
                    self.collect_demo = True
                    break
                elif finish_demo == 'y':
                    self.collect_demo = False
                    break
                else:
                    print('[User Attention!] Invalid input')
                    continue

            if self.collect_demo:
                demo_step = input('[User Attention!] Make a move manually, and enter any key to continue when finshed!')
                self.step = -1
                self.publish_reset()
                return
            else:
                for i, state in enumerate(self.demo_states):
                    save_dir = os.path.join(self.save_dir, self.trj_name, 'goals', f'step_{i}')
                    self._save_step(state, save_dir)
                
                self.goal_mask = self.demo_states[-1]['observation']['mask']
        
        self.setup_init_state()
        self.start_video()
        self.publish_reset()

    def setup_init_state(self):
        is_continue = input('[User Attention!] Please set a random initial state, and enter any keys after setup to continue!')
        

    def clean_up(self, state):
        self.stop_video()

        while True:
            is_success = input('[User Attention!] Is the task successful? (y/n): ')
            if is_success == 'y':
                state['evaluation']['human success'] = True
                break
            elif is_success == 'n':
                state['evaluation']['human success'] = False
                break
            else:
                print('[User Attention!] Invalid input')
                continue
        self.save_step(state)
        self.reset()

    def img_callback(self, data):
        print('Receive observation data')
        crop_rgb = imgmsg_to_cv2_custom(data.crop_rgb, "bgr8")
        crop_depth = imgmsg_to_cv2_custom(data.crop_depth, "64FC1")
        raw_rgb = imgmsg_to_cv2_custom(data.raw_rgb, "bgr8")
        self.real_camera_height = data.camera_height
        input_state = self.post_process(crop_rgb, crop_depth, raw_rgb)

        if self.step == -1:
            self.step += 1
            self.setup_evaluation(input_state)
            
            return

        evaluation = self.evaluate(input_state)

        input_state['evaluation'] = evaluation

        done = (self.step >= self.fix_steps)

        
        if wait_for_user_input():
            print("User signaled finish.")
            done = True

        if done:
            self.clean_up(input_state)
            return
        elif self.step == 0:
            self.init(input_state)
        else:
            self.update(input_state, self.last_action)
    
        
        action = self.act(input_state)
        internal_state = self.get_state()

        save_state = input_state.copy()
        save_state.update(internal_state)

        self.last_action = action
        pick_and_place = action['pick-and-place']
        pixel_actions = ((pick_and_place + 1) / 2 * self.resolution[0]).astype(int).reshape(4)
        action_image = draw_pick_and_place(
            save_state['observation']['rgb'],
            tuple(pixel_actions[:2]),
            tuple(pixel_actions[2:]),
            color=(0, 255, 0)
        ).astype(np.uint8)
        action['pick-and-place'] = action['pick-and-place'].reshape(4).tolist()
        #print('save actio', action)
        save_state['action'] = action
        save_state['action_image'] = action_image
        self.save_step(save_state)
        self.step += 1
        self.publish_action(action)

    def act(self, state):
        pass


    def setup(self):

        if self.task == 'flattening':
            final_state = input('[User Attention!] Please set to the final state for setup evaluation, please enter any key when finsihed!')
            
        elif 'folding' in self.task:
            while True:
                new_demo = input('[User Attention!] Do you want to start a new demonstration? (y/n): ')
                if new_demo == 'n':
                    self.collect_demo = False
                    break
                elif new_demo == 'y':
                    self.collect_demo = True
                    break
                else:
                    print('[User Attention!] Invalid input')
                    continue
            
            if self.collect_demo:
                self.demo_states = []
                init_state = input('[User Attention!] Please set the initial state for setup demonstration, please enter any key when finsihed!')
            else:
                self.setup_init_state()
                self.start_video()
                self.step = 0
                
        
        self.publish_reset()

    def reset(self):
        self.step = -1
        self.last_action = None
        

        while True:
            is_continue = input('[User Attention!] Continue for a new trial? (y/n): ')
            if is_continue == 'n':   
                #rclpy.shutdown()
                raise NotImplementedError
            elif is_continue == 'y':
                self.trj_name = input('[User Attention!] Enter Trial Name: ')
                break
            else:
                print('Invalid input')
                continue
        
        self.setup()
        
        
    
    def stop_video(self):
        if self.video_process is not None:
            print('Stop recording!')
            stop_ffmpeg_recording(self.video_process)

    def start_video(self):
        if self.video_device is not None:
            save_dir = os.path.join(self.save_dir, self.trj_name)
            os.makedirs(save_dir, exist_ok=True)
            output_file = os.path.join(save_dir, 'record.mp4')
            self.video_process = start_ffmpeg_recording(output_file, self.video_device)
        

    def update(self, state, aciton):
        pass

    def init(self, state):
        pass

    def get_state(self):
        return {}

    def run(self):
        self.reset()
        rclpy.spin(self)

    def post_process(self, rgb, depth, raw_rgb=None, pointcloud=None):
        pass