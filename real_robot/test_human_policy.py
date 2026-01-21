# main.py
from robot.dual_arm_arena import DualArmArena
from human_policy import HumanPolicy
from garment_flattening_task import GarmentFlatteningTask
from dotmap import DotMap

def main():
    anrea_config = {
        "ur5e_ip": "192.168.1.10",
        "ur16e_ip": "192.168.1.102",
        "dry_run": False,
        'horizon': 20,
        "debug": True,
        'measure_time': True
    }
    task_config = {
        'debug': True
    }
    agent_config = {
        'debug': False,
        'measure_time': True
    }

    arena = DualArmArena(DotMap(anrea_config))
    policy = HumanPolicy(DotMap(agent_config))
    policy.reset([0])
    task = GarmentFlatteningTask(DotMap(task_config))
    arena.set_task(task)

    info = arena.reset()
    print("Environment ready. Choose skill:")
    print('info evaluate', info['evaluation'])

    while True:
        action = policy.single_act(info)
        info = arena.step(action)
        print('info evaluate', info['evaluation'])

if __name__ == "__main__":
    main()
