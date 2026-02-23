from dotmap import DotMap

from controllers.human.human_multi_primitive import HumanMultiPrimitive
from controllers.human.real_world_human_policy import RealWordHumanPolicy

from env.softgym_garment.tasks.garment_folding import GarmentFoldingTask
from env.softgym_garment.tasks.garment_flattening import GarmentFlatteningTask
from env.softgym_garment.tasks.canonicalisation_alignment import CanonicalisationAlignmentTask
from real_robot.tasks.garment_flattening_task import RealWorldGarmentFlatteningTask
from real_robot.tasks.garment_folding_task import RealWorldGarmentFoldingTask

def build_task(task_cfg):
    # task
    if task_cfg.task_name == 'centre-sleeve-folding':
        demonstrator = HumanMultiPrimitive({"debug": False})
        task = GarmentFoldingTask(DotMap({**task_cfg, "demonstrator": demonstrator}))
       
    elif task_cfg.task_name == 'waist-leg-alignment-folding':
        from controllers.demonstrators.waist_leg_alignment_folding_stochastic_policy \
            import WaistLegFoldingStochasticPolicy
        demonstrator = WaistLegFoldingStochasticPolicy({"debug": False})
        task = GarmentFoldingTask(DotMap({**task_cfg, "demonstrator": demonstrator}))
       
    elif task_cfg.task_name == 'waist-hem-alignment-folding':
        from controllers.demonstrators.waist_hem_alignment_folding_stochastic_policy \
            import WaistHemAlignmentFoldingStochasticPolicy
        demonstrator = WaistHemAlignmentFoldingStochasticPolicy({"debug": False})
        task = GarmentFoldingTask(DotMap({**task_cfg, "demonstrator": demonstrator}))
        
    elif task_cfg.task_name == 'flattening':
        task = GarmentFlatteningTask(task_cfg)
    
    elif task_cfg.task_name == 'canonicalisation-alignment':
        task = CanonicalisationAlignmentTask(task_cfg)

    elif task_cfg.task_name == 'dummy':
        task = None
    elif task_cfg.task_name == 'real-world-garment-flattening':
        task = RealWorldGarmentFlatteningTask(task_cfg)
    elif task_cfg.task_name == 'real-world-garment-folding':
        demonstrator = RealWordHumanPolicy(DotMap())
        task = RealWorldGarmentFoldingTask(DotMap({**task_cfg, "demonstrator": demonstrator}))
    else:
        raise NotImplementedError(f"Task {task_cfg.task_name} not supported")
    return task
