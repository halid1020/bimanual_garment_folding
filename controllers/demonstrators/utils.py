import numpy as np

def get_pixel(key, semkey2pid, keypids, key_pixels):
    pid = semkey2pid[key]
    idx = keypids.index(pid)
    pixel = key_pixels[idx].astype(int).copy()
    return pixel

def norm_pixel(pixel, H, W):
    """Normalize pixel coords into [-1,1] range."""
    return pixel / np.asarray([H, W]) * 2 - 1

def random_point_on_line(p1, p2, cloth_mask, offset=10, max_tries=30, on_mask=True):
    """Pick a random point around the line segment p1-p2 that lies on cloth."""
    H, W = cloth_mask.shape[:2]
    for _ in range(max_tries):
        alpha = np.random.rand()
        base = (1 - alpha) * p1 + alpha * p2
        noise = np.random.randint(-offset, offset + 1, size=2)
        candidate = np.clip(base + noise, [0, 0], [H - 1, W - 1]).astype(int)
        if (not on_mask) or (cloth_mask[candidate[0], candidate[1]] > 0):
            return candidate
    ys, xs = np.where(cloth_mask > 0)
    idx = np.random.randint(len(xs))
    return np.array([ys[idx], xs[idx]])

def sample_near_pixel(pixel, cloth_mask, radius=10, max_tries=30, on_mask=True):
    """Sample a pixel near the given pixel that lies on cloth."""
    H, W = cloth_mask.shape[:2]
    for _ in range(max_tries):
        noise = np.random.randint(-radius, radius + 1, size=2)
        candidate = np.clip(pixel + noise, [0, 0], [H - 1, W - 1]).astype(int)
        if (not on_mask) or (cloth_mask[candidate[0], candidate[1]] > 0):
            return candidate
    ys, xs = np.where(cloth_mask > 0)
    idx = np.random.randint(len(xs))
    return np.array([ys[idx], xs[idx]])