import numpy as np
import struct
import time
import threading
from collections import namedtuple
import serial
from crc import Calculator, Configuration

class ForceSensor:
    BOTA_PRODUCT_CODE = 123456
    BAUDERATE = 460800
    SINC_LENGTH = 512
    CHOP_ENABLE = 0
    FAST_ENABLE = 0
    FIR_DISABLE = 1
    TEMP_COMPENSATION = 0  # 0: Disabled (recommended), 1: Enabled
    USE_CALIBRATION = 1  # 1: calibration matrix active, 0: raw measurements
    DATA_FORMAT = 0  # 0: binary, 1: CSV
    BAUDERATE_CONFIG = 4  # 0: 9600, 1: 57600, 2: 115200, 3: 230400, 4: 460800
    FRAME_HEADER = b"\xAA"
    time_step = 0.01

    def __init__(self, port, force_offset, torque_offset):
        self._port = port
        self._ser = serial.Serial()
        self._pd_thread_stop_event = threading.Event()
        DeviceSet = namedtuple("DeviceSet", "name product_code config_func")
        self._expected_device_layout = {
            0: DeviceSet("BFT-SENS-SER-M8", self.BOTA_PRODUCT_CODE, self.force_sensor_setup)
        }

        self.prev_force = np.zeros(3)
        self.prev_torque = np.zeros(3)
        self.force_offset = force_offset  # Force offset for calibration
        self.torque_offset = torque_offset  # Torque offset for calibration

    def force_sensor_setup(self):
        # Serial communication settings
        self._ser.baudrate = self.BAUDERATE
        self._ser.port = self._port
        self._ser.timeout = 10

        try:
            self._ser.open()
        except Exception as e:
            print(f"Could not open port: {e}")
            return

        if not self._ser.is_open:
            print("Port could not be opened.")

        # Wait for initialization
        self._ser.read_until(bytes("App Init", "ascii"))
        time.sleep(0.5)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

        # Enter CONFIG mode
        self._ser.write(b"C")
        self._ser.read_until(bytes("r,0,C,0", "ascii"))

        # Communication setup
        comm_setup = f"c,{self.TEMP_COMPENSATION},{self.USE_CALIBRATION},{self.DATA_FORMAT},{self.BAUDERATE_CONFIG}"
        print("Communication setup:", comm_setup)
        self._ser.write(bytes(comm_setup, "ascii"))
        self._ser.read_until(bytes("r,0,c,0", "ascii"))

        # Filter setup
        filter_setup = f"f,{self.SINC_LENGTH},{self.CHOP_ENABLE},{self.FAST_ENABLE},{self.FIR_DISABLE}"
        print("Filter setup:", filter_setup)
        self._ser.write(bytes(filter_setup, "ascii"))
        self._ser.read_until(bytes("r,0,f,0", "ascii"))

        # Enter RUN mode
        self._ser.write(b"R")
        self._ser.read_until(bytes("r,0,R,0", "ascii"))
        print("Force sensor setup complete, running mode activated.")

    def get_force_obs(self):
        # Start data sync and CRC checks
        self._ser.flushInput()
        self._ser.flushOutput()
        frame_synced = False
        crc16X25Configuration = Configuration(16, 0x1021, 0xFFFF, 0xFFFF, True, True)
        crc_calculator = Calculator(crc16X25Configuration)

        # Attempt to synchronize with the frame header
        while not frame_synced and not self._pd_thread_stop_event.is_set():
            possible_header = self._ser.read(1)
            if self.FRAME_HEADER == possible_header:
                data_frame = self._ser.read(34)
                crc16_ccitt_frame = self._ser.read(2)
                crc16_ccitt = struct.unpack_from('H', crc16_ccitt_frame, 0)[0]
                checksum = crc_calculator.checksum(data_frame)
                if checksum == crc16_ccitt:
                    frame_synced = True
                else:
                    print("CRC mismatch - resyncing.")
                    self._ser.read(1)  # Skip a byte and try again
            else:
                # print("Incorrect header byte received, resyncing.")
                self._ser.read(1)  # Skip a byte and retry

        # If synchronized, read the data frame
        if not frame_synced:
            print("Failed to sync with sensor data frame.")
            return self.prev_force, np.zeros(3)

        # Extract force and torque values
        try:
            Fx = struct.unpack_from('f', data_frame, 2)[0]
            Fy = struct.unpack_from('f', data_frame, 6)[0]
            Fz = struct.unpack_from('f', data_frame, 10)[0]
            Mx = struct.unpack_from('f', data_frame, 14)[0]
            My = struct.unpack_from('f', data_frame, 18)[0]
            Mz = struct.unpack_from('f', data_frame, 22)[0]

            # Apply offsets to force and torque readings
            self.prev_force = np.array([Fx, Fy, Fz]) - self.force_offset
            self.prev_torque = np.array([Mx, My, Mz]) - self.torque_offset

            # Print adjusted values to confirm offsets are applied
            # print(f"Raw force: {[Fx, Fy, Fz]}, Offset: {self.force_offset}, Adjusted force: {self.prev_force}")
            # print(f"Raw torque: {[Mx, My, Mz]}, Offset: {self.torque_offset}, Adjusted torque: {self.prev_torque}")

            return self.prev_force, self.prev_torque

        except struct.error as e:
            print(f"Data unpacking error: {e}")
            return self.prev_force, np.zeros(3)


    @staticmethod
    def _sleep(duration, get_now=time.perf_counter):
        now = get_now()
        end = now + duration
        while now < end:
            now = get_now()
