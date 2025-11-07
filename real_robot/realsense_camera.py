#!/usr/bin/env python

import cv2
import numpy as np
import pyrealsense2 as rs
from save_utils import save_depth, save_colour

class RealsenseCamera():

    def __init__(self, debug=False):
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Explicitly select your device (optional but safer if multiple are connected)
        # config.enable_device('239722074129')

        # Use supported resolutions and frame rates
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

        self.pipeline.start(config)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        ### Depth Camera Macros ###
        self.colorizer = rs.colorizer()
        self.hole_filling = rs.hole_filling_filter()
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.2)
        self.temporal.set_option(rs.option.filter_smooth_delta, 24)
        self.debug = debug

        self.take_rgbd()


    def _post_process_depth(self, depth_frame):
        H, W = np.asanyarray(depth_frame.get_data()).shape[:2]
        raw_depth_data = np.asarray(depth_frame.get_data()).astype(float)/1000
        if self.debug:
            save_depth(raw_depth_data, filename='raw_depth', directory='./tmp')
        depth_frame = self.temporal.process(depth_frame)
        #depth_frame = self.disparity_to_depth.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)

        depth_data = np.asarray(depth_frame.get_data()).astype(float)/1000
        depth_data = cv2.resize(depth_data, (W, H))

        
        ## Align depth to color, to fine tune.
        OW = 0 #-14
        OH = 0 #-10
        CH = int(H)
        CW = int(W)
        MH = int(H//2)
        MW = int(W//2)
        depth_data = depth_data[
            max(MH-CH//2+OH, 0): min(MH+CH//2+OH, H), 
            max(MW-CW//2+OW, 0): min(MW+CW//2+OW, W)]
        depth_data = cv2.resize(depth_data, (W, H))

        ## get the blak ones
        blank_mask = (depth_data == 0)

        self.depth_img = depth_data.copy()
        
        return depth_data

    def take_rgbd(self):
        for _ in range(10):
            frames = self.pipeline.wait_for_frames()
        
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            self.intrinsic = color_frame.profile.as_video_stream_profile().intrinsics

            depth_image = self._post_process_depth(depth_frame)
            if self.debug:
                save_depth(depth_image, filename='post_depth', directory='./tmp', colour=True)
            color_image = np.asanyarray(color_frame.get_data())
            if self.debug:
                save_colour(color_image, filename='color', directory='./tmp', rgb2bgr=False)
        return color_image, depth_image

    def get_intrinsic(self):
        return self.intrinsic
    

def main(args=None):
    image_retriever = RealsenseCamera(debug=True)

    try:
        while True:
            command = input("Press Enter to capture images or 'q' to quit: ")
            if command.lower() == 'q':
                break
            image_retriever.take_rgbd()
    finally:
        exit(0)
if __name__ == '__main__':
    main()