import pycurl
import xmlrpc.client
from io import BytesIO


class RG2:
    def __init__(self, robot_ip, rg_id):
        self.rg_id = rg_id
        self.robot_ip = robot_ip
        self.port = 41414
        self.connected = True  # 🔧 Track connection state
        self.open_width = 30

    def get_rg_width(self):
        if not self.connected:
            raise RuntimeError("Gripper not connected.")
            
        xml_request = f"""<?xml version="1.0"?>
        <methodCall>
            <methodName>rg_get_width</methodName>
            <params>
                <param><value><int>{self.rg_id}</int></value></param>
            </params>
        </methodCall>"""

        headers = ["Content-Type: application/x-www-form-urlencoded"]
        data = xml_request.replace('\r\n', '').encode()

        curl = pycurl.Curl()
        curl.setopt(curl.URL, f'http://{self.robot_ip}:{self.port}')
        curl.setopt(curl.HTTPHEADER, headers)
        curl.setopt(curl.POSTFIELDS, data)

        buffer = BytesIO()
        curl.setopt(curl.WRITEDATA, buffer)
        curl.perform()
        response = buffer.getvalue()
        curl.close()

        xml_response = xmlrpc.client.loads(response.decode('utf-8'))
        rg_width = float(xml_response[0][0])
        return rg_width

    def rg_grip(self, target_width: float = 100, target_force: float = 10) -> bool:
        if not self.connected:
            raise RuntimeError("Gripper not connected.")
            
        xml_request = f"""<?xml version="1.0"?>
        <methodCall>
            <methodName>rg_grip</methodName>
            <params>
                <param><value><int>{self.rg_id}</int></value></param>
                <param><value><double>{target_width}</double></value></param>
                <param><value><double>{target_force}</double></value></param>
            </params>
        </methodCall>"""

        headers = ["Content-Type: application/x-www-form-urlencoded"]
        data = xml_request.replace('\r\n', '').encode()

        curl = pycurl.Curl()
        curl.setopt(curl.URL, f'http://{self.robot_ip}:{self.port}')
        curl.setopt(curl.HTTPHEADER, headers)
        curl.setopt(curl.POSTFIELDS, data)

        buffer = BytesIO()
        curl.setopt(curl.WRITEDATA, buffer)
        curl.perform()
        curl.close()
        return True

    def open(self):
        """Open gripper to max width"""
        return self.rg_grip(self.open_width, 40)

    def close(self):
        """Close gripper fully"""
        return self.rg_grip(0.0, 40.0)

    def disconnect(self):
        """🔧 Gracefully disconnect gripper (cleanup hook)."""
        if not self.connected:
            print("RG2 already disconnected.")
            return
        try:
            # Optionally open gripper before disconnecting
            # self.open()
            print("Disconnecting RG2 gripper connection...")
            self.connected = False
        except Exception as e:
            print(f"Error while disconnecting RG2: {e}")
        finally:
            print("RG2 gripper disconnected.")


def main():
    print("Main")
    rg_id = 0
    ip = "192.168.1.10"
    rg_gripper = RG2(ip, rg_id)

    try:
        rg_width = rg_gripper.get_rg_width()
        print("rg_width:", rg_width)

        rg_gripper.open()
        rg_gripper.close()
    finally:
        rg_gripper.disconnect()


if __name__ == "__main__":
    main()
