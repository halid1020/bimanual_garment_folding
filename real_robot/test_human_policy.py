# main.py
from dual_arm_arena import DualArmArena
from human_policy import HumanPolicy

def main():
    config = {
        "ur5e_ip": "192.168.1.10",
        "ur16e_ip": "192.168.1.102",
        "dry_run": False
    }

    arena = DualArmArena(config)
    policy = HumanPolicy(arena)

    info = arena.reset()
    print("Environment ready. Choose skill:")

    while not info['done']:
        action = policy.single_act(info)
        #info = arena.step(action)

if __name__ == "__main__":
    main()
