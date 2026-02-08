### **Installation & Usage Guide**

**Prerequisites**
Before proceeding, please ensure you have already installed and tested the corresponding `softgym` and `actoris_harena` repositories.

### **1. Installation**

1. **Create a virtual environment:**
```bash
conda create -n mp-fold python=3.10 -y
conda activate mp-fold

```


2. **Install the `actoris_harena` package:**
Navigate to your `actoris_harena` directory and install it with the Torch dependencies.
```bash
cd <path-to-agent-arena>
pip install -e ".[torch]"

```


3. **Setup Assets:**
Unzip the `assets.zip` file into the root directory of this repository (`bimanual_garment_folding`). This folder contains the meshes and goal configurations required for garment folding.

### **2. Testing the Installation**

Run the following commands to verify the setup. This will execute a random policy on the multi-primitive setup.

```bash
cd <path-to-bimanual_garment_folding>

# Source the setup script
. ./setup.sh 

# Run the evaluation script
python train/hydra_eval.py --config-name run_exp/random_multi_primitive_multi_longsleeve_folding_from_crumpled

```

The evaluation results will be saved in the `./tmp` folder.

### **3. Adding a New Agent / Controller**

1. **Create the Controller:**
Add your new controller script inside the `controllers` folder. Ensure it adheres to the `Agent` or `TrainableAgent` interface defined in the `agent-arena` package. You can refer to existing controllers in that folder for examples.
2. **Register the Controller:**
Open `train/utils.py` and add your new agent to the `register_actoris_harena` function to make it available to the training script.

### **4. Configuring Experiments**

All experiment configurations are stored in the `conf` folder. This codebase uses **Hydra** for configuration management.

To create a new experiment, create a YAML file directly under the `conf` folder. This file serves as the entry point and should define the following parameters:

* **`agent`**: The filename of the agent configuration (located in `conf/agent`). Inside that file, ensure the `name` matches the agent you registered in Step 3.
* **`arena`**: The filename of the environment configuration (located in `conf/arena`). Ensure the arena name inside matches a registered arena.
* **`task`**: The filename of the task configuration (located in `conf/task`), defining objectives like flattening or folding.
* **`exp_name`**: The name of your experiment (this should match your YAML filename).
* **`project_name`**: Used for Weights & Biases (WandB) logging. You can leave this as default.
* **`save_root`**: The directory where logs and experiment data will be stored.
* **`train_and_eval`**: Specifies which `agent-arena` API function to use for the experiment.
