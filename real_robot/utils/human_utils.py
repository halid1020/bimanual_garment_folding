import cv2
import numpy as np

def draw_ui(img, clicks, mask=None, btn_height=50):
    """
    Helper to composite the image, mask, button bar, and current points.
    Returns the composite image (canvas) to display.
    """
    # 1. Prepare the base image with mask overlay
    clone = img.copy()
    if mask is not None:
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask
        
        mask_color = np.zeros_like(clone)
        mask_color[mask_gray > 0] = (0, 255, 0)  # Green overlay
        alpha = 0.4
        # Blend mask into clone
        clone = cv2.addWeighted(clone, 1 - alpha, mask_color, alpha, 0)

    # 2. Create the UI Header (Button Area)
    h, w, c = clone.shape
    header = np.zeros((btn_height, w, 3), dtype=np.uint8)
    header[:] = (50, 50, 50)  # Dark grey background

    # Draw UNDO button box
    btn_w = 100
    cv2.rectangle(header, (10, 5), (10 + btn_w, btn_height - 5), (100, 100, 200), -1)
    cv2.putText(header, "UNDO", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add instructions text next to button
    cv2.putText(header, f"Points: {len(clicks)} selected", (130, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # 3. Stack Header and Image
    # Canvas coordinates: y=0 to btn_height is header; y>btn_height is image
    canvas = np.vstack((header, clone))

    # 4. Draw existing clicks
    # Note: We must shift y-coordinates down by btn_height
    for pt in clicks:
        # Draw on canvas: (x, y + btn_height)
        display_pt = (pt[0], pt[1] + btn_height)
        cv2.circle(canvas, display_pt, 6, (0, 0, 255), -1) # Red dot
        cv2.circle(canvas, display_pt, 8, (0, 0, 0), 1)    # Black outline

    return canvas

def click_points_pick_and_place(window_name, img, mask=None):
    """
    Select 4 points with an UNDO button in a header bar.
    """
    clicks = []
    BTN_HEIGHT = 50

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if click is in the UI Header area
            if y < BTN_HEIGHT:
                # Check if inside UNDO button (approx x < 110 based on draw_ui)
                if x < 110: 
                    if clicks:
                        clicks.pop()
                        # Redraw
                        canvas = draw_ui(img, clicks, mask, BTN_HEIGHT)
                        cv2.imshow(window_name, canvas)
            else:
                # Click is on the image
                # Convert window y to image y
                img_y = y - BTN_HEIGHT
                
                # Only add if we haven't reached the limit
                if len(clicks) < 4:
                    clicks.append((x, img_y))
                    # Redraw
                    canvas = draw_ui(img, clicks, mask, BTN_HEIGHT)
                    cv2.imshow(window_name, canvas)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720 + BTN_HEIGHT)
    
    # Initial display
    canvas = draw_ui(img, clicks, mask, BTN_HEIGHT)
    cv2.imshow(window_name, canvas)
    cv2.setMouseCallback(window_name, mouse_cb)

    print("--- PICK & PLACE ---")
    print("UI: Click 'UNDO' button to remove last point.")
    print("Order: PICK0 -> PLACE0 -> PICK1 -> PLACE1")
    print("Press 'q' to cancel.")

    while len(clicks) < 4:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyWindow(window_name)

    if len(clicks) < 4:
        raise RuntimeError(f"Expected 4 points, got {len(clicks)}")

    return clicks[0], clicks[1], clicks[2], clicks[3]

def click_points_pick_and_fling(window_name, img, mask=None):
    """
    Select 2 points with an UNDO button in a header bar.
    """
    clicks = []
    BTN_HEIGHT = 50

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check UI area
            if y < BTN_HEIGHT:
                if x < 110: # Inside Undo
                    if clicks:
                        clicks.pop()
                        canvas = draw_ui(img, clicks, mask, BTN_HEIGHT)
                        cv2.imshow(window_name, canvas)
            else:
                # Image area
                img_y = y - BTN_HEIGHT
                if len(clicks) < 2:
                    clicks.append((x, img_y))
                    canvas = draw_ui(img, clicks, mask, BTN_HEIGHT)
                    cv2.imshow(window_name, canvas)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720 + BTN_HEIGHT)
    
    canvas = draw_ui(img, clicks, mask, BTN_HEIGHT)
    cv2.imshow(window_name, canvas)
    cv2.setMouseCallback(window_name, mouse_cb)

    print("--- PICK & FLING ---")
    print("UI: Click 'UNDO' button to remove last point.")
    print("Order: PICK -> PLACE")
    
    while len(clicks) < 2:
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cv2.destroyWindow(window_name)
    if len(clicks) < 2:
        raise RuntimeError("2 points not selected")
    
    return clicks[0], clicks[1]