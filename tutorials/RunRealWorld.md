
# Tutorial: Setting Up and Running Real-World Experiments

## 1. Robot Arrangements

First, ensure the two robot arms are facing each other, approximately 1.6 metres apart. The camera should be mounted at a height of roughly 1.66 metres, facing top-down to view the midpoint between the two arms. The setup should look as follows:

![image](./figures/robot-setup.jpeg)

Next, use an Ethernet switch and three Ethernet cables to connect both robot arms to the control machine. The camera connects to the control machine via a USB connection.

![image](./figures/robot-machine-connection.jpeg)

## 2. Booting the Robot Arms

First, ensure each robot is connected to its power supply, then press the hard on/off button on its pendant. 

Follow the instruction images below to complete the soft booting, release the brakes, and start the system. This ensures the robot is in a ready state for control.

![image](./figures/robot-hard-boot.jpeg)
![image](./figures/robot-ready-on.jpeg)
![image](./figures/robot-on-1.jpeg)
![image](./figures/robot-on.jpeg)
![image](./figures/robot-start.jpeg)
![image](./figures/robot-finish-booting.jpeg)

![image](./figures/robot-exit-booting-page.jpeg)

## 3. Switching Control Modes

Universal Robots operate in two control modes: (1) local mode and (2) remote mode. In local mode, you control the robot via its pendant. In remote mode, control is handled through the Ethernet connection.

The tab for switching control modes is located at the top right corner of the pendant interface, as illustrated below:

![image](./figures/robot-switch-control-mode.jpeg)
![image](./figures/robot-switch-to-local-control.jpeg)

Ensure you set both robots to `remote` mode before proceeding to the next step.

## 4. Network Connection

Open a terminal on the control machine and set its IP address:
```bash
sudo ip addr add 192.168.1.20/24 dev enp45s0 # Or your preferred IP address.
```
You may need to execute the command above several times to establish a stable connection for the following process.

Check the IP addresses of the robots to ensure they match the configurations defined in `real_robot/calibration/ur5e.yaml` and `real_robot/calibration/ur16e.yaml`. You can check and change the IP address of the robots using their teach pendant by navigating to: `Settings` page → `System` tab → `Network` window. Please see the following figure for the correct configuration. 

![image](./figures/robot-network.jpeg)

You can then test the control connection using the following script:

```bash
. ./setup.sh
cd real_robot/test
python test_send_cmd_ur5e.py
```

The robot should move briefly and then stop.

Next, test whether the gripper is functioning correctly:

```bash
. ./setup.sh
cd real_robot/robot
python rg2_gripper.py
```

The gripper should open and close.

## 5. Hand-Eye Calibration

Once a stable connection is confirmed, perform a hand-eye calibration for both arms to determine the positions of the robot bases relative to the camera location.

On the control machine, type and launch `realsense-viewer` and verify that the RGB-D images are displaying correctly. The two robot arms should be visible on the sides, with the midpoint between the arms at the centre of the camera view. Ensure the depth colour map is evenly distributed. Please see the figure below:

![image](./figures/realsense.png)

You may need to step on the table to adjust the camera position. Once verified, close the RealSense viewer.

Then, have the UR5e robot arm grasp the printed ChArUco board (provided in `real_robot/calibration/calib.io_charuco_210x300_7x5_40_30_DICT_4X4.pdf`). 

![image](./figures/printed-chaorocu.jpeg)

Ensure the ChArUco board faces the right side of the robot arm, as shown below:

![image](./figures/robot-grasp-chauroco.jpeg)


Make sure the robot grasp it firmly, so that it will not move during the calibration process; if it does moves, you need to recalibrate the arm.

You can control the gripper manually through the `OnRobot RG` plug-in on the top-right corner of the teach pendant.

![image](./figures/robot-control-grasping-plugin.jpeg)
![image](./figures/robot-control-grasping-open.jpeg)
![image](./figures/robot-control-grasping-close.jpeg)

Navigate to the calibration directory and run:
```bash
. ./setup.sh
cd real_robot/calibration
python hand_to_eye_calib.py --config ur5e.yaml
```
*(Note: If the camera pipeline throws an error, try replugging all USB connections along the camera's cable.)*

Repeat this process for the UR16e robot. Calibration takes approximately two minutes per arm, with each moving through roughly 50 positions.

## 6. System-Level Testing with Human Control

Following calibration, verify its accuracy using a human-controlled multi-primitive policy. This procedure also ensures the primitives are functioning correctly.

From the root directory of this repository, run:
```bash
. ./setup.sh
python tool/eval_real_world.py --config-name real_world_exp/real_world_human_alignment
```

*Instructions:* The program will prompt you to provide the garment ID (e.g., `teen-brown-top`). Next, arrange the garment into its goal state so the system can record it (if this garment's goal was collected previously, it will skip this step). Afterwards, crumple the garment into its initial state. The script will then guide you through providing primitive actions. All instructions from the program that require user input will be highlighted in <span style="color: green;">green</span>.

The target alignment position adjustment window should look like the following for `alignment` tasks:

![image](./figures/set-align-target.png)

In this figure, the red and blue arcs represent the workspace of each arm, and the white garment mask is the detected mask of the testing garment in its flattened state.

Try to complete the entire episode using all different types of primitives. The results will be saved to `~/project/garment_folding_data/real_world_human_alignment`.

## 7. Control with Neural Controllers

Once the system is verified with a human-controlled episode, you are ready to run our neural controller, Magpie.

In the training machine's logging directory, zip the folder with its checkppints. For example, for the experiment example `magpie_ctr_align_longsleeve_p4_v10`.

```
zip -r magpie_ctr_align_longsleeve_p4_v10.zip magpie_ctr_align_longsleeve_p4_v10/checkpoints
```

Then, try to transfer this zip file to the robot control machine.

**Test neural controllers:**
> **TODO:** This step will be polished in the near future.
   
To evaluate a trained neural controller, run a command similar to:
```bash
python tool/eval_real_world.py --config-name real_world_exp/magpie_align_longsleeve_p4_final
```
   
> **Important Note for Neural Controllers:** The configuration files for real-world experiments are located in `conf/real_world_exp` and share a similar structure to the simulation configs. To run neural controllers successfully, you must first place the correct network checkpoint file into the corresponding experiment log folder so the agent can load the weights correctly.

Ensure you securely save the evaluation results once finished.

## 8. Shutting Down the Robots

When you finish your work, shutting down the robots is straightforward. Simply press the hard on/off button on the pendant, then press the `Discard Changes` virtual button on the interface. Ensure the robots are completely shut down whenever left unattended.