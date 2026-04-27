# MEGPIE: Multi-Primitive Bimanual Garment Folding from Crumpled States with Pixel-based Flow Matching Policies

**Author:** Halid A. Kadi  
**Contributors:** Houdeyfa Ajrou, Lucy Walsh, and Ivan Kapelyukh  
**Supervisors:** Kasim Terzic and John Oyekan  

**Affiliations:**
* Department of Computer Science, University of York, United Kingdom
* Department of Computer Science, University of Sheffield, United Kingdom
* Department of Computing, Imperial College London, United Kingdom
* School of Computer Science, University of St Andrews, United Kingdom
* Department of Computer Science, University of Loughborough, United Kingdom

---

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Simulation Installation](#2-simulation-installation)
3. [Testing the Installation](#3-testing-the-installation)
4. [Adding a New Agent / Controller](#4-adding-a-new-agent--controller)
5. [Configuring Experiments](#5-configuring-experiments)
6. [Simulation Experiments](#6-simulation-experiments)
7. [Real-World Experiments](#7-real-world-experiments)

---

## 1. Prerequisites

Before proceeding, please ensure you have already installed and tested the following:
* The `py3.10` branch of the [`softgym`](https://github.com/halid1020/softgym/tree/py3.10) repository (only required for simulation experiments).
* The `develop` branch of the [`actoris_harena`](https://github.com/halid1020/actoris_harena/tree/develop) repository.

---

## 2. Simulation Installation

### Step 1: Create a virtual environment
```bash
conda create -n magpie python=3.10 -y
conda activate magpie
```

### Step 2: Install the `actoris_harena` package
Navigate to your `actoris_harena` directory and install it with the Torch dependencies:
```bash
cd <path-to-agent-arena>
pip install -e ".[torch]"
```

### Step 3: Install additional required packages
Install the remaining dependencies required for this repository:
```bash
pip install pycurl
pip install segment_anything==1.0
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch_geometric

# For enbale the running of ClothMate
pip installl trimesh
pip install OpenEXR
```

### Step 4: Setup Assets (only for simulation)
Download and unzip the [`assets.zip`]([https://link-to-your-assets.com/assets.zip](https://link-to-your-assets.com/assets.zip)) file into the root directory of this repository (`bimanual_garment_folding`). This folder contains the meshes and goal configurations required for garment folding.

---

## 3. Testing the Installation

Run the following commands to verify your setup. This will execute a random policy on the multi-primitive setup in simulation.

```bash
cd <path-to-bimanual_garment_folding>

# Source the setup script
. ./setup.sh 

# Run the evaluation script
python tool/hydra_eval.py --config-name sim_exp/random_multi_primitive_multi_longsleeve_folding_from_crumpled
```

The evaluation results will be saved in the `./tmp` folder.

---

## 4. Adding a New Agent / Controller

1. **Create the Controller:**
   Add your new controller script inside the `controllers` folder. Ensure it adheres to the `Agent` or `TrainableAgent` interface defined in the `actoris_harena` package. You can refer to the existing controllers in that folder for examples.
2. **Register the Controller:**
   Open `registration/agent.py` and add your new agent to the `register_agents` function. This makes it available to the training script.

---

## 5. Configuring Experiments

All experiment configurations are stored in the `conf` folder. This codebase uses **Hydra** for configuration management.

To create a new experiment, create a YAML file directly under the `conf` folder. This file serves as the entry point and should define the following parameters:

* **`agent`**: The filename of the agent configuration (located in `conf/agent`). Inside that file, ensure the `name` matches the agent you registered in Step 4.
* **`arena`**: The filename of the environment configuration (located in `conf/arena`). Ensure the arena name matches a registered arena.
* **`task`**: The filename of the task configuration (located in `conf/task`), defining objectives like flattening or folding.
* **`exp_name`**: The name of your experiment (this should match your YAML filename).
* **`project_name`**: Used for Weights & Biases (WandB) logging. You can leave this as the default.
* **`save_root`**: The directory where logs and experiment data will be stored.
* **`train_and_eval`**: Specifies which `actoris_harena` API function to use for the experiment.

---

## 6. Simulation Experiments

### Running on a Local or Remote Machine

To start training in the background on a local or remote machine, run the following commands:

```bash
. ./setup.sh

# Example configuration shown below
./job_scripts/submit_training_locally.sh magpie/gc_diff_mp_longsleeve_align_200_demo_one_hot_one2one_pred_sem_key_v1
```

Running the script above is equivalent to executing the following job in the foreground:

```bash
. ./setup.sh

python tool/hydra_train.py --config-name magpie/gc_diff_mp_longsleeve_align_200_demo_one_hot_one2one_pred_sem_key_v1 
```

### Re-evaluating a Job

To re-evaluate an existing experiement in the background, use the evaluation script:

```bash
. ./setup.sh

./job_scripts/submit_evaluating_locally.sh <sim-exp-config>
# Example: magpie/gc_diff_mp_longsleeve_align_200_demo_one_hot_one2one_pred_sem_key_v1
```

> **Note:** Ensure that the log directory does not contain previously generated evaluation results before running this command.

### Submitting to the Viking Cluster (University of York)

If you are on the login node of the University of York's Viking cluster, you can submit a training job by running:

```bash
./job_scripts/generate_and_submit_viking_job.sh <sim-exp-config> -c 6 -m 18G -p gpu -t 48:00:00 # number of cpus, memory usage, partition and time
```

---

## 7. Real-World Experiments

You do not need to install `softgym` for the real-world setup, but `actoris_harena` and the packages listed in Section 2 must be installed. Additionally, real-world experiments require the Segment Anything Model (SAM) weights.

### Step 1: Download SAM Weights
Download `sam_vit_h_4b8939.pth` and place it inside the `sam_vit` directory. *(Note: This is strictly required for real-world vision processing).*

### Step 2: Network Configuration
1. Ensure the control machine (e.g., a GPU laptop) is connected to both robot arms via an Ethernet switch.
2. Set the IP address of the control machine:
   ```bash
   sudo ip addr add 192.168.1.20/24 dev enp45s0
   ```
3. Check the IP addresses of the robots to ensure they are consistent with the information defined in `real_robot/calibration/ur5e.yaml` and `real_robot/calibration/ur16e.yaml`.

### Step 3: Camera and Calibration
4. Launch `realsense-viewer` and verify that the RGB-D images are displaying correctly.
5. **Conduct hand-eye calibration:**
   * Have the UR5e robot arm grasp the printed ChArUco board (provided in `real_robot/calibration/calib.io_charuco_210x300_7x5_40_30_DICT_4X4.pdf`).
   * Navigate to the calibration directory and run:
     ```bash
     cd real_robot/calibration
     python hand_to_eye_calib.py --config ur5e.yaml
     ```
   * Repeat this process for the UR16e robot. It takes approximately 2 minutes for each arm to calibrate.

### Step 4: Testing the Setup
6. **Test with a human policy:**
   From the root directory of this repository, run:
   ```bash
   python tool/eval_real_world.py --config-name real_world_exp/real_world_human_alignment
   ```
   *Instructions:* The program will guide you to provide the garment ID. Next, arrange the garment into its goal state so the system can record it. After that, crumple the garment into its initial state. The script will then guide you through providing primitive actions.

7. **Test neural controllers:**
   To evaluate a trained neural controller, run a command similar to:
   ```bash
   python tool/eval_real_world.py --config-name real_world_exp/eval_megpie_flow_matching_controller
   ```
   *(Alternatively, use the diffusion demo config: `real_world_exp/diffusion_multi_primitive_multi_longsleeve_canonicalisation_alignment_demo_100_snap_one_hot_rgb+goal_mask_predict_semkey`)*

> **Important Note for Neural Controllers:** The configuration files for real-world experiments are located in `conf/real_world_exp` and share a similar structure to the simulation configs. In order to run neural controllers successfully, you must first place the correct network checkpoint file into the corresponding experiment log folder so the agent can load the weights correctly.