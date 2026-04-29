# Tutorial: Integrating a New Policy, Control Method, or Agent

In this tutorial, we will discuss how to add or integrate a new policy, control method, or agent into our framework.

**Prerequisites:** Before starting, please ensure you have followed the setup instructions in the `README.md` of the current directory. 

**Note:** The exact code and configuration files shown in this tutorial are already provided within the repository. The purpose of this guide is to walk you through the methodological pipeline so you understand the framework's mechanics and can replicate these steps to integrate other policies into new environments.

---

## 1. Understand the `Agent` Subclasses

All agents must inherit from the abstract classes [`Agent`]([https://github.com/halid1020/actoris_harena/blob/develop/actoris_harena/agent/agent.py](https://github.com/halid1020/actoris_harena/blob/develop/actoris_harena/agent/agent.py)) or [`TrainableAgent`]([https://github.com/halid1020/actoris_harena/blob/develop/actoris_harena/agent/trainable_agent.py](https://github.com/halid1020/actoris_harena/blob/develop/actoris_harena/agent/trainable_agent.py)) within the `Actoris Harena` framework. 

For an example of a human-controlled policy inheriting from `Agent` for multi-primitive bimanual manipulation, please see our [`HumanMultiPrimitive`](../controllers/human/human_multi_primitive.py) class. We primarily implement the constructor (`__init__`), as well as the `reset` and `single_act` functions. 

The environment's action tool, [`HybridActionPrimitive`](../environment/softgym_garment/action_primitives/hybrid_action_primitives.py), within our [`GarmentEnv`](../environment/softgym_garment/garment_env.py), accepts a set of primitives as a dictionary. 

The `HumanMultiPrimitive` agent returns four types of primitives. Values are defined in a normalised pixel space ranging from -1 to 1 (where `[-1, -1]` is the top-left corner, `[-1, 1]` is the top-right, `[1, -1]` is the bottom-left, and `[1, 1]` is the bottom-right). These primitives include:

* **`norm-pixel-pick-and-fling` (4 values - pick 0, pick 1):** Two arms pick the garment simultaneously, hold it parallel to the horizontal central line from the camera, stretch it, fling it forward towards the top of the image, fling it back and down, and finally release it slightly below the horizontal central line.
* **`norm-pixel-dual-pick-and-place` (8 values - pick 0, pick 1, place 0, place 1):** Two arms execute a pick-and-place simultaneously if there is no collision on their path; otherwise, they execute sequentially.
* **`norm-pixel-single-pick-and-place` (4 values - pick 0, place 0):** The robot automatically chooses one arm to execute a single-arm pick-and-place.
* **`no-operation` (0 values):** The robot does nothing.

This class is registered in `registration/agent.py` to hook the initialisation into `Actoris Harena` like so:
```python
athar.register_agent('human-multi-primitive', HumanMultiPrimitive)
```

To run this policy, we define its experiment configuration file, located at `conf/sim_exp/magpie/human_multi_primitive_multi_longsleeve_alignment_random_flattened_goal.yaml`:

```yaml
# @package _global_
defaults:
  - /agent/magpie@agent: human_multi_primitive_multi_longsleeve_alignment
  - /arena/magpie@arena: multi_longsleeve_provide_semkey_pixel_no_success_stop_resol_128_workspace
  - /task@task: central_alignment

exp_name:  human_multi_primitive_multi_longsleeve_alignment_random_flattened_goal
project_name: bimanual_garment_folding
save_root: /mnt/ssd/garment_folding_data
train_and_eval: train_and_evaluate_single
```

The corresponding agent, arena, and task configurations can be found at:
* **Agent config:** [`conf/agent/magpie/human_multi_primitive_multi_longsleeve_alignment.yaml`](conf/agent/magpie/human_multi_primitive_multi_longsleeve_alignment.yaml)
* **Arena config:** [`conf/arena/magpie/multi_longsleeve_provide_semkey_pixel_no_success_stop_resol_128_workspace.yaml`](conf/arena/magpie/multi_longsleeve_provide_semkey_pixel_no_success_stop_resol_128_workspace.yaml)
* **Task config:** [`conf/task/central_alignment.yaml`](conf/task/central_alignment.yaml)

Policy evaluation is executed via the [`tool/hydra_eval.py`](../tool/hydra_eval.py) script. This script builds the agent, arena, and tasks through the framework, then streamlines a 30-trial evaluation using `athar.evaluate`. 

Run the human experiments using:
```bash
python tool/hydra_eval.py --config-name magpie/human_multi_primitive_multi_longsleeve_alignment_random_flattened_goal 
```
Alternatively, use the background script (adding `f` runs it in the foreground):
```bash
./job_scripts/submit_evaluating_locally.sh magpie/human_multi_primitive_multi_longsleeve_alignment_random_flattened_goal f 
```

---

## 2. Understand the `TrainableAgent` Subclasses

The `TrainableAgent` (inheriting from `Agent`) provides functionality for controllers that need to optimise a parameterised model iteratively. It expects the user to provide `train` and `set_data_augmenter` functions, and includes built-in saving and loading functionalities for model parameters. 

A prime example is our SAC implementation, located at [`actoris_harena/agent/drl/sac`](https://github.com/halid1020/actoris_harena/blob/develop/actoris_harena/agent/drl/sac/vanilla_sac.py). For a more complex example, see the `MagpieAgent` class in [`controllers/magpie/magpie_agent`](../controllers/magpie/magpie_agent.py).

Registration and configuration follow the same mechanism as the human policy in Section 1. To start training these agents, run:
```bash
python tool/hydra_train.py --config-name magpie/magpie_align_longsleeve
```
Or via the shell script:
```bash
./job_scripts/submit_training_locally.sh magpie/magpie_align_longsleeve f 
```

This script automatically trains, validates, and evaluates the agent on both the best and last checkpoints (with the best checkpoint selected based on validation performance). 

Common arguments to set in the agent configuration files include:
* `validation_interval`: e.g., 5000.
* `total_update_steps`: e.g., 10000.
* `data_augmenter`: A string that initialises the augmentation class (configured in `conf/data_augmenter` and sourced from the `data_augmentation` folder).

Other arugments uniquely belong to the specific agents construction and running.

---

## 3. Integrate Existing Methods from 3rd-Party Repositories

When integrating existing 3rd-party methods, do not rewrite them from scratch. Instead, use an **adapter pattern** to wrap the construction and action inference processes. Some 3rd-party policies are deeply coupled with their specific environment classes, so you will need to decouple them carefully.

### Environment Observations (`obs` and `info`)
Our `GarmentEnv` automatically provides a rich set of observations via the `observation` and `info` dictionaries returned by the `reset` and `step` functions. 

**The `observation` dictionary in `info` provides:**
* `rgb` / `image` / `raw_rgb`: The visual camera feed (cropped and full resolution).
* `depth`: The depth map.
* `mask` / `cloth_mask`: Boolean masks identifying the garment.
* `robot0_mask` / `robot1_mask`: Masks for the robot arms (if the workspace is applied).
* `particle_positions`: 3D coordinates of the mesh particles.
* `semkey2pid` / `semkey_pos`: Semantic keypoints and their respective positions.
* `visible_point_cloud`: Point cloud data for the visible portions of the mesh.
* `mesh_faces`: The topology of the mesh.
* **Flattened versions:** Features like `flattened_semkey_pos` and `flattened_semkey_norm_pixel` providing data relative to the goal flattened state.
* `action_step`: The current step in the trajectory.

**The `info` dictionary also provides:**
* Environment states: `observation`, `arena_id`, `action_space`, `sim_steps`.
* Termination flags: `out_of_view`, `terminated`, `truncated`, `done`, `success`.
* Tracking data: `overstretch`, `picker_norm_pixel_pos`, `low_level_mesh_particles`.
* Goal data: `goal`, `goals`, `flattened_obs`. They follows the same structure of `observation`

### Task Evaluation and Rewards (`info['evaluation']` and `info['reward']`)
Instead of rewriting task metrics in the wrapper, rely on the values automatically computed by our task classes (such as `GarmentFlatteningTask`, `AlignmentTask`, or `GarmentFoldingTask`).

**The `info['evaluation']` dictionary includes metrics such as:**
* `max_IoU_to_flattened` / `algn_IoU_to_flattened`: Intersection over Union scores against the target.
* `normalised_coverage` / `normalised_improvement`: How well the garment covers the target area.
* `mean_particle_distance` / `semantic_keypoint_distance`: L2 distance metrics for point clouds and specific keypoints.
* `overstretch`: A metric tracking if the cloth is being pulled beyond its limits.
* Trajectory maximums: Variables like `maximum_trj_max_IoU_to_flattened` for tracking the best state reached in an episode.

**The `info['reward']` dictionary includes structured reward signals such as:**
* `coverage_alignment`: Base rewards for flattening and alignment.
* `particle_distance` / `keypoint_distance`: Rewards inversely tied to point cloud distances.
* `multi_stage_reward`: Complex reward formulations for multi-step folding tasks.
* Penalty-adjusted rewards (e.g., `coverage_alignment_with_stretch_and_affordance_penalty` or `particle_distance_with_stretch_penality`), penalising the agent for overstretching the fabric or outputting poor affordances.

*Note: The `info` dictionary also provides a direct reference to the simulation environment through the `info['arena']` field, which is useful for edge cases where absolute decoupling is not possible.*

### Creating the Adapter
You must identify the differences between the 3rd-party method's input/output space and our framework, creating conversions at the beginning and end of the `single_act` function. 

A good practice is to only include the files strictly necessary for action inference inside the adapter folder. You do not need to implement a `train` function for pre-trained adapters, but you must carefully implement the `load_best` function to load the provided pre-trained weights.

For excellent examples of how to integrate 3rd-party methods using the adapter pattern, please review the following classes:
* [`ClothFunnelsAdapter`](../controllers/rl/cloth_funnels/adapter.py)
* [`ClothMateAdapter`](../controllers/rl/cloth_mate/adapter.py)
* [`UnifoldingAdapter`](../controllers/rl/unifolding/adapter.py)

Once integrated, remember to register these new agents and create their experiment configurations (as detailed in Section 1). Document the integration process thoroughly in your PR or README, noting any extra packages required for installation, input/output conversions, and any unavoidable modifications to the 3rd-party source code.

An example of evaluating an integrated 3rd-party method:
```bash
./job_scripts/submit_evaluating_locally.sh magpie/clothmate_longsleeve_canon_flattening f
```