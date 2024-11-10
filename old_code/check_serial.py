import serial

def check_serial_port(port='/dev/ttyUSB0'):
    try:
        ser = serial.Serial(port, baudrate=460800, timeout=1)  # Change baudrate and timeout as needed
        ser.close()  # Close the port after checking
        print(f"Successfully opened {port}")
    except serial.SerialException as e:
        print(f"Failed to open {port}: {e}")

if __name__ == "__main__":
    check_serial_port()
