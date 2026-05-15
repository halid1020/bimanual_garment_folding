import os
import cv2
import numpy as np

REMOTE = True
ACTUAL_DISPLAY = os.environ.get("DISPLAY", "localhost:10.0")

if REMOTE:
    CV2_DISPLAY = "localhost:10.0" 
    SIM_DISPLAY = ""
else:
    CV2_DISPLAY = ":0"
    SIM_DISPLAY = ":0"


class KeypointGUI:
    def __init__(self, semantics):
        self.semantics = list(semantics)
        # Generate random BGR colors for each keypoint to distinguish them easily
        np.random.seed(42)
        self.colors = np.random.randint(50, 255, (len(self.semantics), 3)).tolist()
        
    def _mouse_callback(self, event, x, y, flags, param):
        """Callback function for OpenCV mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.click_order < len(self.semantics):
                name = self.semantics[self.click_order]
                color = self.colors[self.click_order]
                
                # Because we use the native raw_rgb image, x and y are exactly correct
                self.keypoints[name] = np.array([x, y])
                self.points.append((x, y, color, name))
                self.click_order += 1
            else:
                self.warning_message = "All keypoints assigned! Press 'f' to Finish."

    def run(self, rgb):
        # --- SWITCH TO GUI DISPLAY ---
        print(f"Switching DISPLAY to {CV2_DISPLAY} for GUI...")
        os.environ["DISPLAY"] = CV2_DISPLAY

        try:
            print(f'[KeypointGUI] Please provide keypoints for {self.semantics}')
            
            # Convert RGB directly to BGR without resizing
            self.bgr_base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # Reset state
            self.keypoints = {}
            self.click_order = 0
            self.points = []  
            self.warning_message = None

            window_name = "Keypoint Annotation"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            # Set the initial window size to match the image natively
            h, w = self.bgr_base.shape[:2]
            cv2.resizeWindow(window_name, w, h)
            cv2.setMouseCallback(window_name, self._mouse_callback)

            # Determine dynamic font scale based on image width to keep text readable
            fs = max(0.4, w / 900.0)
            thick_bg = max(2, int(fs * 3))
            thick_fg = max(1, int(fs * 2))

            # Interactive Display Loop
            while True:
                # Create a fresh canvas to draw on every frame
                canvas = self.bgr_base.copy()

                # Draw all assigned points and their labels
                for (px, py, color, name) in self.points:
                    cv2.circle(canvas, (px, py), 5, color, -1)
                    cv2.circle(canvas, (px, py), 6, (255, 255, 255), 1) # White outline
                    cv2.putText(canvas, name, (px + 10, py - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, fs, color, thick_bg)
                    cv2.putText(canvas, name, (px + 10, py - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), thick_fg)

                # Draw UI Text (Instructions)
                if self.click_order < len(self.semantics):
                    prompt_text = f"Next: {self.semantics[self.click_order]}"
                    prompt_color = (0, 0, 255) # Red
                else:
                    prompt_text = "All keypoints assigned! Press 'f' or Enter to Finish."
                    prompt_color = (0, 255, 0) # Green

                # Top prompt
                cv2.putText(canvas, prompt_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.2, (0, 0, 0), thick_bg + 1)
                cv2.putText(canvas, prompt_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.2, prompt_color, thick_fg)
                
                # Bottom instructions pinned dynamically to the image height (h)
                instruction_text = "Shortcuts: [u] Undo  |  [r] Reset  |  [f] Finish"
                cv2.putText(canvas, instruction_text, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), thick_bg)
                cv2.putText(canvas, instruction_text, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), thick_fg)

                # Warnings
                if self.warning_message:
                    cv2.putText(canvas, self.warning_message, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), thick_bg)

                # Render the canvas
                cv2.imshow(window_name, canvas)
                
                # Wait for keyboard input
                key = cv2.waitKey(10) & 0xFF

                # --- Handle Keyboard Shortcuts ---
                if key == ord('f') or key == 13:  # 'f' or Enter key
                    missing = [s for s in self.semantics if s not in self.keypoints]
                    if missing:
                        self.warning_message = "Assign remaining: " + ", ".join(missing)
                    else:
                        print("Finishing annotation...")
                        break
                        
                elif key == ord('u'):  # 'u' to Undo
                    if self.click_order > 0:
                        self.click_order -= 1
                        removed_name = self.semantics[self.click_order]
                        self.keypoints.pop(removed_name, None)
                        self.points.pop()
                        self.warning_message = None
                        print(f"Undid {removed_name}")
                        
                elif key == ord('r'):  # 'r' to Reset
                    self.click_order = 0
                    self.keypoints.clear()
                    self.points.clear()
                    self.warning_message = None
                    print("Reset all keypoints.")

        finally:
            # Clean up the OpenCV window
            cv2.destroyAllWindows()
            
            # --- REVERT TO SIMULATION DISPLAY ---
            print(f"Reverting DISPLAY back to {SIM_DISPLAY} for Simulation...")
            os.environ["DISPLAY"] = SIM_DISPLAY

        return self.keypoints