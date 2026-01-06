import pybullet as p
import numpy as np

IMG_W, IMG_H = 640, 480
FOV = 60
NEAR, FAR = 0.01, 3.0

NEAR, FAR = 0.01, 3.0
TABLE_Z = 0.65  # The height where objects sit

fx = IMG_W / (2 * np.tan(np.deg2rad(FOV / 2)))
fy = fx
cx, cy = IMG_W / 2, IMG_H / 2

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])


CENTER = np.array([0.0, 0.0])  # midpoint of bases
RADIUS = 0.7                  # meters

# Match dimensions to your capture size
W, H = 640, 480 

# Tighten the bounds so the table fills the view
# If the table is 0.8m wide, these bounds should be close to that
X_MIN, X_MAX = -0.5, 0.5 
Y_MIN, Y_MAX = -0.375, 0.375 # Maintains 640/480 aspect ratio

# Calculate PPM based on the desired width and the physical bounds
PPM = W / (X_MAX - X_MIN)

# Re-initialize buffers with correct shape
accum = np.zeros((H, W, 3), np.float32)
weight = np.zeros((H, W), np.float32)

def capture_reference_view():
    cam_pos = [2.0, -2.0, 2.0]   # far & high
    lookat = [0.0, 0.0, 0.6]     # table center
    up = [0, 0, 1]

    view = p.computeViewMatrix(cam_pos, lookat, up)
    proj = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=IMG_W / IMG_H,
        nearVal=0.1,
        farVal=5.0
    )

    img = p.getCameraImage(
        IMG_W, IMG_H,
        view, proj,
        renderer=p.ER_TINY_RENDERER
    )

    rgb = np.reshape(img[2], (IMG_H, IMG_W, 4))[:, :, :3]
    return rgb

# def capture_wrist_camera(body_id):
#     pos, orn = p.getBasePositionAndOrientation(body_id)

#     CAMERA_OFFSET = [0, 0, 0.0]
#     CAMERA_ROT = p.getQuaternionFromEuler([np.pi, 0, 0])  # looking down

#     cam_pos, cam_orn = p.multiplyTransforms(
#         pos, orn,
#         CAMERA_OFFSET, CAMERA_ROT
#     )

#     rot = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
#     forward = rot @ np.array([0, 0, 1])
#     up = rot @ np.array([0, -1, 0])

#     view = p.computeViewMatrix(cam_pos, cam_pos + forward, up)
#     proj = p.computeProjectionMatrixFOV(FOV, IMG_W / IMG_H, NEAR, FAR)

#     img = p.getCameraImage(
#         IMG_W, IMG_H, view, proj,
#         renderer=p.ER_TINY_RENDERER
#     )

#     rgb = np.reshape(img[2], (IMG_H, IMG_W, 4))[:, :, :3]
#     return rgb, view

def capture_wrist_camera(body_id, target=[0, 0, 0.65]):
    # Get the current position of the wrist sphere
    pos, _ = p.getBasePositionAndOrientation(body_id)

    # Automatically compute the rotation to look at the target
    # This handles all tilting angles and heights perfectly
    view = p.computeViewMatrix(
        cameraEyePosition=pos,
        cameraTargetPosition=target,
        cameraUpVector=[0, 0, 1]  # Standard Z-up world
    )

    # Use the standard FOV projection
    proj = p.computeProjectionMatrixFOV(FOV, IMG_W / IMG_H, NEAR, FAR)

    img = p.getCameraImage(
        IMG_W, IMG_H, view, proj,
        renderer=p.ER_TINY_RENDERER
    )

    rgb = np.reshape(img[2], (IMG_H, IMG_W, 4))[:, :, :3]
    return rgb, view

TOP_Z = 1.8          # height above table (safe margin)
LOOKAT_Z = 0.65      # table height

def capture_topdown_groundtruth():
    # Position camera directly above center
    cam_pos = [0, 0, 2.0]
    target = [0, 0, 0]
    up = [0, 1, 0] 

    view = p.computeViewMatrix(cam_pos, target, up)
    
    proj = p.computeProjectionMatrixFOV(FOV, IMG_W / IMG_H, NEAR, FAR)

    img = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER)
    rgb = np.reshape(img[2], (H, W, 4))[:, :, :3]
    return rgb.astype(np.uint8)

def get_perspective_bounds(cam_z, target_z, fov, aspect):
    distance = cam_z - target_z
    visible_height = 2 * distance * np.tan(np.deg2rad(fov / 2))
    visible_width = visible_height * aspect
    return visible_width, visible_height


def fuse(rgb, view_matrix):
    global accum, weight
    # V is World-to-Camera, we need Camera-to-World
    V = np.array(view_matrix).reshape(4, 4).T
    T_world_cam = np.linalg.inv(V)
    cam_pos = T_world_cam[:3, 3]

    # Calculate the boundaries of the Ground Truth view at the table height
    # Camera at 2.0, Table at 0.65 -> distance is 1.35
    vis_w, vis_h = get_perspective_bounds(2.0, TABLE_Z, FOV, W/H)
    
    # Bounds for mapping P_world to pixels
    X_MIN, X_MAX = -vis_w/2, vis_w/2
    Y_MIN, Y_MAX = -vis_h/2, vis_h/2

    h_img, w_img = rgb.shape[:2]
    aspect = w_img / h_img
    tan_half_fov = np.tan(np.deg2rad(FOV / 2))

    for v in range(0, h_img, 2):
        for u in range(0, w_img, 2):
            # 1. Perspective Ray Casting from Wrist Camera
            x_ndc = (2.0 * u / w_img) - 1.0
            y_ndc = 1.0 - (2.0 * v / h_img)
            
            ray_cam = np.array([x_ndc * aspect * tan_half_fov, y_ndc * tan_half_fov, -1])
            ray_world = T_world_cam[:3, :3] @ ray_cam
            
            # 2. Intersection with Table Plane
            if abs(ray_world[2]) < 1e-6: continue
            t = (TABLE_Z - cam_pos[2]) / ray_world[2]
            
            if t > 0:
                P = cam_pos + t * ray_world
                
                # 3. Map to the Ground Truth's Perspective Grid
                # Because the GT is also perspective, we need to map the world 
                # point based on the GT camera's frustum at that specific Z.
                if X_MIN < P[0] < X_MAX and Y_MIN < P[1] < Y_MAX:
                    # Convert world P to pixel coordinates in the 640x480 grid
                    px = int((P[0] - X_MIN) / (X_MAX - X_MIN) * W)
                    py = int((Y_MAX - P[1]) / (Y_MAX - Y_MIN) * H)
                    
                    if 0 <= px < W and 0 <= py < H:
                        accum[py, px] += rgb[v, u]
                        weight[py, px] += 1

# def fuse(rgb, view_matrix):
#     V = np.array(view_matrix).reshape(4, 4).T
#     T_cam_world = np.linalg.inv(V)

#     h, w = rgb.shape[:2]

#     for v in range(0, h, 2):
#         for u in range(0, w, 2):
#             x = (u - cx) / fx
#             y = (v - cy) / fy

#             ray_cam = np.array([x, y, -1, 0])

#             ray_world = T_cam_world @ ray_cam
#             cam_pos = T_cam_world[:3, 3]

#             # Must be pointing down
#             if ray_world[2] >= 0:
#                 continue

#             t = -cam_pos[2] / ray_world[2]
#             if t <= 0:
#                 continue

#             P = cam_pos + t * ray_world[:3]

#             if not (X_MIN < P[0] < X_MAX and Y_MIN < P[1] < Y_MAX):
#                 continue

#             px = int((P[0] - X_MIN) * PPM)
#             py = int((Y_MAX - P[1]) * PPM)

#             accum[py, px] += rgb[v, u]
#             weight[py, px] += 1

# TOP_Z = 1.8          # height above table (safe margin)
# LOOKAT_Z = 0.65      # table height

# X_MIN, X_MAX = -0.7, 0.7
# Y_MIN, Y_MAX = -0.7, 0.7

# GT_W = W
# GT_H = H

# def ortho_projection(left, right, bottom, top, near, far):
#     return [
#         2.0 / (right - left), 0, 0, -(right + left) / (right - left),
#         0, 2.0 / (top - bottom), 0, -(top + bottom) / (top - bottom),
#         0, 0, -2.0 / (far - near), -(far + near) / (far - near),
#         0, 0, 0, 1
#     ]


# def capture_topdown_groundtruth():
#     cam_pos = [CENTER[0], CENTER[1], TOP_Z]
#     target = [CENTER[0], CENTER[1], LOOKAT_Z]

#     # ✅ Safe up vector for nadir view
#     up = [0, 1, 0]

#     view = p.computeViewMatrix(
#         cameraEyePosition=cam_pos,
#         cameraTargetPosition=target,
#         cameraUpVector=up
#     )

#     proj = ortho_projection(
#         left=X_MIN,
#         right=X_MAX,
#         bottom=Y_MIN,
#         top=Y_MAX,
#         near=-2.5,
#         far=2.5
#     )

#     img = p.getCameraImage(
#         W, H,
#         viewMatrix=view,
#         projectionMatrix=proj,
#         renderer=p.ER_TINY_RENDERER
#     )

#     rgb = np.reshape(img[2], (H, W, 4))[:, :, :3]
#     return rgb
