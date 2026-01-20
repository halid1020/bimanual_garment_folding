import socket

HOST = "192.168.1.10"  # UR5e IP
PORT = 30002            # Secondary client port

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

# Move robot to a safe position
command = "movej([0, -1.57, 0, -1.57, 0, 0], a=1.2, v=0.25)\n"
s.sendall(command.encode('utf-8'))

s.close()
