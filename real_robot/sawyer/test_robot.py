import rospy
import intera_interface

def main():
    rospy.init_node('sawyer_test')
    limb = intera_interface.Limb('right')

    # Get joint angles
    print("Current Joint Angles:", limb.joint_angles())

    # Simple joint command (move a bit)
    joint_command = limb.joint_angles()
    joint_command['right_j0'] += 0.1
    limb.move_to_joint_positions(joint_command)

if __name__ == '__main__':
    main()
