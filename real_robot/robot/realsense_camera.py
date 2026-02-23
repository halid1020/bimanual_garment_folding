#!/usr/bin/env python

import cv2
import numpy as np
import pyrealsense2 as rs
import time
from real_robot.utils.save_utils import save_depth, save_colour

class RealsenseCamera():

    def __init__(self, debug=False):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Explicitly select your device (optional but safer if multiple are connected)
        # config.enable_device('239722074129')

        # Use supported resolutions and frame rates
        # Note: 30FPS is generally more stable for synchronization than 6/15
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        self.start_pipeline()

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        ### Depth Camera Macros ###
        self.colorizer = rs.colorizer()
        self.hole_filling = rs.hole_filling_filter()
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.2)
        self.temporal.set_option(rs.option.filter_smooth_delta, 24)
        self.debug = debug
        self.intrinsic = None

        # Warm up
        self.take_rgbd()

    def start_pipeline(self):
        try:
            self.profile = self.pipeline.start(self.config)
        except Exception as e:
            print(f"[Camera] Error starting pipeline: {e}. Attempting Hardware Reset...")
            self.restart()

    def restart(self):
        """
        Stops and restarts the pipeline. useful if the buffer freezes.
        """
        print("[Camera] Restarting pipeline...")
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass # Pipeline might not be running
        
        time.sleep(1.0) # Give the USB controller a moment to clear
        self.start_pipeline()
        
        # Warm up again to ensure auto-exposure settles
        print("[Camera] Warming up after restart...")
        for _ in range(10):
            try:
                self.pipeline.wait_for_frames(timeout_ms=1000)
            except:
                pass

    def _post_process_depth(self, depth_frame):
        H, W = np.asanyarray(depth_frame.get_data()).shape[:2]
        
        # Only save raw depth if strictly necessary (slows down loop)
        if self.debug:
            raw_depth_data = np.asarray(depth_frame.get_data()).astype(float)/1000
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
        # 1. Flush the buffer (CRITICAL FIX)
        # We drain the queue by calling wait_for_frames 5-10 times without processing.
        # This ensures we get the *newest* frame, not an old buffered one.
        for _ in range(5):
            self.pipeline.wait_for_frames()
        
        # 2. Capture the actual frame we want to use
        frames = None
        for _ in range(3):
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                break
            except RuntimeError:
                print("[Camera] Time out waiting for frames...")
                continue
        
        if frames is None:
            print("[Camera] CRITICAL: Camera frozen. Restarting...")
            self.restart()
            return self.take_rgbd()

        # 3. Process ONLY the final frame
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        # Update intrinsics (safe to do every frame)
        self.intrinsic = color_frame.profile.as_video_stream_profile().intrinsics

        depth_image = self._post_process_depth(depth_frame)
        color_image = np.asanyarray(color_frame.get_data())

        if self.debug:
            save_depth(depth_image, filename='post_depth', directory='./tmp', colour=True)
            save_colour(color_image, filename='color', directory='./tmp', rgb2bgr=False)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        return color_image, depth_image

    def get_intrinsic(self):
        if self.intrinsic is None:
            # Force a capture if intrinsics haven't been populated yet
            self.take_rgbd()
        return self.intrinsic
    

def main(args=None):
    image_retriever = RealsenseCamera(debug=True)

    try:
        while True:
            command = input("Press Enter to capture images or 'q' to quit: ")
            if command.lower() == 'q':
                break
            t0 = time.time()
            image_retriever.take_rgbd()
            print(f"Capture took: {time.time()-t0:.3f}s")
    finally:
        exit(0)
        
if __name__ == '__main__':
    main()