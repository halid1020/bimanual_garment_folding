import socket

# utils.py
def resolve_save_root(default_root):
    """
    Returns the machine-specific save root based on the hostname.
    Falls back to default_root if the host is not recognized.
    """
    hostname = socket.gethostname()
    print(f"[tool.utils, resolve_save_root] Detected Host: {socket.gethostname()}")
    if "pc282" in hostname:
        return '/media/hcv530/T7/garment_folding_data'
    elif "thanos" in hostname:
        return '/data/ah390/bimanual_garment_folding'
    elif "viking" in hostname:
        return '/mnt/scratch/users/hcv530/garment_folding_data'
    elif "labruja" in hostname:
        return '/data/ah390/bimanual_garment_folding'
    else:
        raise ValueError
    
    return default_root