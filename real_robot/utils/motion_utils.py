# motion_utils.py
import time

def safe_movel(robot, pose, speed, acc, blocking=True, dry_run=False, avoid_singularity=False):
    if dry_run:
        print(f"[Dry-run] Moving {pose} (speed={speed}, acc={acc})")
        time.sleep(0.2)
        return True
    return robot.movel(pose, speed, acc, blocking, avoid_singularity=avoid_singularity)


def safe_gripper(robot, action="open", dry_run=False):
    if dry_run:
        print(f"[Dry-run] Gripper {action}")
        time.sleep(0.1)
        return True

    if action == "open":
        return robot.open_gripper()
    elif action == "close":
        return robot.close_gripper()
    else:
        raise ValueError(f"Unknown gripper action: {action}")


def safe_home(robot, speed=1.5, acc=1.0, blocking=True, dry_run=False):
    if dry_run:
        print("[Dry-run] Moving home")
        time.sleep(0.2)
        return True
    return robot.home(speed, acc, blocking)

def safe_out_scene(robot, speed=1.5, acc=1.0, blocking=True, dry_run=False):
    if dry_run:
        print("[Dry-run] Moving home")
        time.sleep(0.2)
        return True
    return robot.out_scene(speed, acc, blocking)
