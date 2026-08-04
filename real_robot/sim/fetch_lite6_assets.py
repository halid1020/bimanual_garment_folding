"""Fetch the UFACTORY Lite 6 URDF and meshes for the simulation.

Source: https://github.com/hygradme/lite6_urdf (BSD-3-Clause), generated from the
xacro in https://github.com/xArm-Developer/xarm_ros2. Copyright (c) 2021 UFACTORY
Inc. The parameters are genuine -- 0.2435 m base, 0.2002 m link 2, the
0.087 / -0.22761 elbow offsets, and the real joint limits -- and the meshes are
referenced by RELATIVE path (``visual/link1.stl``), not ``package://``, so PyBullet
loads the file as-is.

The assets land in ``assets/lite6_urdf/``, which is already in .gitignore, so ~3.4 MB
of STL never enters the repository. This script is the reproducible step instead.
Everything in real_robot/sim degrades gracefully when the assets are missing, so a
clone without network still runs -- it just draws frames instead of arms.

Usage
    python real_robot/sim/fetch_lite6_assets.py
    python real_robot/sim/fetch_lite6_assets.py --force
"""
import argparse
import os
import sys
import urllib.error
import urllib.request

RAW_BASE = "https://raw.githubusercontent.com/hygradme/lite6_urdf/main"
FILES = [
    "LICENSE",
    "lite6.urdf",
    "visual/base.stl",
    "visual/link1.stl",
    "visual/link2.stl",
    "visual/link3.stl",
    "visual/link4.stl",
    "visual/link5.stl",
    "visual/link6.stl",
    "visual/gripper_lite.stl",
]

ATTRIBUTION = """\
UFACTORY Lite 6 description, vendored for simulation only.

Source : https://github.com/hygradme/lite6_urdf
Upstream: https://github.com/xArm-Developer/xarm_ros2 (generated from its xacro)
Licence: BSD-3-Clause -- see LICENSE in this directory
Copyright (c) 2021 UFACTORY Inc.

Fetched by real_robot/sim/fetch_lite6_assets.py. Not committed: `assets` is in
.gitignore. Re-run that script to restore.
"""


def repo_root():
    """MP_FOLD_PATH when setup.sh has been sourced, else infer from this file."""
    env = os.environ.get('MP_FOLD_PATH')
    if env:
        return env
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def asset_dir():
    return os.path.join(repo_root(), 'assets', 'lite6_urdf')


def urdf_path():
    return os.path.join(asset_dir(), 'lite6.urdf')


def assets_present():
    """Is the description complete enough to load? Checks the meshes too -- a URDF
    with missing STLs loads to an invisible robot, which is worse than no robot
    because it looks like a rendering bug."""
    root = asset_dir()
    return all(os.path.exists(os.path.join(root, name)) for name in FILES
               if name != 'LICENSE')


def fetch(force=False, verbose=True):
    root = asset_dir()
    os.makedirs(os.path.join(root, 'visual'), exist_ok=True)

    total = 0
    for name in FILES:
        dest = os.path.join(root, name)
        if os.path.exists(dest) and not force:
            if verbose:
                print("  have  {}".format(name))
            total += os.path.getsize(dest)
            continue
        url = "{}/{}".format(RAW_BASE, name)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            print("  !! failed {}: {}".format(name, e))
            print("     The simulation still runs without these -- it draws base and")
            print("     TCP frames instead of arm meshes.")
            return False
        with open(dest, 'wb') as f:
            f.write(data)
        total += len(data)
        if verbose:
            print("  got   {}  ({:.0f} kB)".format(name, len(data) / 1024.0))

    with open(os.path.join(root, 'ATTRIBUTION.txt'), 'w') as f:
        f.write(ATTRIBUTION)
    if verbose:
        print("\n{:.1f} MB in {}".format(total / 1e6, root))
        print("BSD-3-Clause, (c) 2021 UFACTORY Inc -- see LICENSE and ATTRIBUTION.txt.")
    return True


def main():
    ap = argparse.ArgumentParser(description="Fetch the Lite 6 URDF for the simulation.")
    ap.add_argument('--force', action='store_true', help="re-download files that exist")
    args = ap.parse_args()
    print("Fetching the Lite 6 description into {}".format(asset_dir()))
    return 0 if fetch(force=args.force) else 1


if __name__ == '__main__':
    sys.exit(main())
