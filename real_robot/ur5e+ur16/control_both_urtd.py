import rtde_control
import rtde_receive
import time
import math

UR5E_IP = "192.168.1.10"
UR16E_IP = "192.168.1.102"

SPEED = 0.5       # rad/s
ACCEL = 1.0       # rad/s^2
AMPLITUDE = 0.2   # rad
PERIOD = 5.0      # seconds for full back-and-forth
DT = 0.2          # seconds between trajectory points

def main():
    ur5e_ctrl = rtde_control.RTDEControlInterface(UR5E_IP)
    ur16e_ctrl = rtde_control.RTDEControlInterface(UR16E_IP)

    ur5e_recv = rtde_receive.RTDEReceiveInterface(UR5E_IP)
    ur16e_recv = rtde_receive.RTDEReceiveInterface(UR16E_IP)

    q0_ur5e = ur5e_recv.getActualQ()
    q0_ur16e = ur16e_recv.getActualQ()

    print("Starting smooth motion...")

    try:
        while True:
            t0 = time.time()
            # Generate a small trajectory for the next DT seconds
            steps = 5
            for i in range(steps):
                t = i / steps * DT
                target_ur5e = [q + AMPLITUDE * math.sin(2 * math.pi * (t + t0) / PERIOD) for q in q0_ur5e]
                target_ur16e = [q - AMPLITUDE * math.sin(2 * math.pi * (t + t0) / PERIOD) for q in q0_ur16e]

                # Blocking move (async=False) ensures smooth execution
                ur5e_ctrl.moveJ(target_ur5e, speed=SPEED, acceleration=ACCEL, asynchronous=False)
                ur16e_ctrl.moveJ(target_ur16e, speed=SPEED, acceleration=ACCEL, asynchronous=False)

            # Wait until next DT period
            time.sleep(DT)

    except KeyboardInterrupt:
        print("Stopping robots safely...")
        ur5e_ctrl.stopJ(2.0)
        ur16e_ctrl.stopJ(2.0)
        del ur5e_ctrl, ur16e_ctrl, ur5e_recv, ur16e_recv
        print("Stopped.")

if __name__ == "__main__":
    main()
