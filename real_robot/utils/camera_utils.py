import cv2
import numpy as np

def get_birdeye_rgb_and_pose(camera_rgb, T_base_cam, K, scale=800, margin=0.1, rotate_ccw=True):
    """
    Convert a perspective RGB image to an orthogonal (bird's-eye) view and
    compute the equivalent top-down camera intrinsics and extrinsics.

    Parameters
    ----------
    camera_rgb : np.ndarray
        RGB image (H x W x 3)
    T_base_cam : np.ndarray
        4x4 base-to-camera transformation
    K : np.ndarray
        3x3 camera intrinsic matrix
    scale : float
        Pixels per meter for output image (controls zoom)
    margin : float
        Padding (as fraction of the image size)
    rotate_ccw : bool
        Rotate 90° counter-clockwise for visualization

    Returns
    -------
    orthogonal_rgb : np.ndarray
        Top-down RGB image
    K_ortho : np.ndarray
        Intrinsics for the orthogonal camera
    T_base_ortho : np.ndarray
        Extrinsics for the orthogonal camera
    """
    print('T', T_base_cam)
    print('K', K)

    # === Step 1: Homography ===
    T_cam_base = np.linalg.inv(T_base_cam)
    R = T_cam_base[:3, :3]
    t = T_cam_base[:3, 3:4]
    n = np.array([[0, 0, 1]]).T
    H = R - (t @ n.T) / (n.T @ t)
    H = K @ H
    H = H / H[2, 2]

    # === Step 2: Project corners to find bounds ===
    h, w = camera_rgb.shape[:2]
    corners = np.array([[0, 0, 1],
                        [w, 0, 1],
                        [0, h, 1],
                        [w, h, 1]]).T
    ground_corners = np.linalg.inv(H) @ corners
    ground_corners /= ground_corners[2]
    gx, gy = ground_corners[0], ground_corners[1]

    # === Step 3: Define output region ===
    margin = 0.1
    xmin, xmax = gx.min(), gx.max()
    ymin, ymax = gy.min(), gy.max()
    xrange = (xmin - margin*(xmax-xmin), xmax + margin*(xmax-xmin))
    yrange = (ymin - margin*(ymax-ymin), ymax + margin*(ymax-ymin))

    scale_pxpm = scale / (xmax - xmin) * (xmax - xmin)
    x_lin = np.linspace(xrange[0], xrange[1], int((xrange[1]-xrange[0])*scale_pxpm))
    y_lin = np.linspace(yrange[1], yrange[0], int((yrange[1]-yrange[0])*scale_pxpm))
    xv, yv = np.meshgrid(x_lin, y_lin)

    pts_ground = np.stack([xv, yv, np.ones_like(xv)], axis=-1).reshape(-1, 3).T
    pts_img = H @ pts_ground
    pts_img /= pts_img[2]

    map_x = pts_img[0].reshape(yv.shape).astype(np.float32)
    map_y = pts_img[1].reshape(yv.shape).astype(np.float32)

    orthogonal_rgb = cv2.remap(
        camera_rgb, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    # === Step 4: Define orthogonal camera pose ===
    # Top-down camera looking straight down from +Z direction
    height = T_base_cam[2, 3]  # camera height above ground (approx)
    R_base_ortho = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    T_base_ortho = np.eye(4)
    T_base_ortho[:3, :3] = R_base_ortho
    T_base_ortho[:3, 3] = np.array([0, 0, height])

    # === Step 5: Define orthogonal camera intrinsics ===
    h_out, w_out = orthogonal_rgb.shape[:2]
    fx = fy = scale_pxpm  # pixels per meter
    cx, cy = w_out / 2, h_out / 2
    K_ortho = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])

    # === Step 6: Optional rotation (CCW) ===
    if rotate_ccw:
        orthogonal_rgb = cv2.rotate(orthogonal_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # When rotating, update intrinsics accordingly
        K_rot = np.array([
            [0, -1, h_out],
            [1,  0, 0],
            [0,  0, 1]
        ])
        K_ortho = K_rot @ K_ortho

    return orthogonal_rgb, map_x, map_y, K_ortho, T_base_ortho

def intrinsic_to_params(intr):
    """Convert pyrealsense2 intrinsics object to fx,fy,ppx,ppy."""
    fx = intr.fx
    fy = intr.fy
    cx = intr.ppx
    cy = intr.ppy
    return fx, fy, cx, cy


def intrinsics_to_matrix(intrinsics):
    """
    Convert a pyrealsense2.intrinsics object to a 3x3 NumPy camera matrix.
    """
    K = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=float)
    return K