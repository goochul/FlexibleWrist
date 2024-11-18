from ForceSensor import ForceSensor
import numpy as np
import time
import matplotlib.pyplot as plt

# Calibration function
def calibrate_force_sensor(sensor, num_samples=100, sleep_time=0.01):
    readings = []
    print("Starting calibration...")
    for _ in range(num_samples):
        try:
            force, torque = sensor.get_force_obs()
            readings.append((force, torque))
            time.sleep(sleep_time)
        except Exception as e:
            print(f"Error during calibration: {e}")
            return None, None

    force_offset = np.mean([r[0] for r in readings], axis=0)
    torque_offset = np.mean([r[1] for r in readings], axis=0)
    print(f"Calibration complete.\nForce Offset: {force_offset}, Torque Offset: {torque_offset}")
    return force_offset, torque_offset

# Real-time data reading and plotting
def read_force_sensor(sensor, force_offset, torque_offset, duration=10):
    plt.ion()  # Enable interactive mode
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].set_title("Real-time Force Data (Adjusted)")
    axs[0].set_ylabel("Force (N)")
    axs[0].set_xlabel("Time Step")
    axs[0].legend(["Fx", "Fy", "Fz"], loc="upper right")
    axs[0].grid()

    axs[1].set_title("Real-time Torque Data (Adjusted)")
    axs[1].set_ylabel("Torque (Nm)")
    axs[1].set_xlabel("Time Step")
    axs[1].legend(["Tx", "Ty", "Tz"], loc="upper right")
    axs[1].grid()

    force_data = []
    torque_data = []

    print("\nCollecting and visualizing sensor data:")
    start_time = time.time()
    while time.time() - start_time < duration:
        try:
            raw_force, raw_torque = sensor.get_force_obs()
            adjusted_force = raw_force - force_offset
            adjusted_torque = raw_torque - torque_offset

            force_data.append(adjusted_force)
            torque_data.append(adjusted_torque)

            # Update the plots
            axs[0].cla()
            axs[0].set_title("Real-time Force Data (Adjusted)")
            axs[0].set_ylabel("Force (N)")
            axs[0].set_xlabel("Time Step")
            axs[0].grid()
            axs[0].plot(np.array(force_data), label=["Fx", "Fy", "Fz"])
            axs[0].legend(loc="upper right")

            axs[1].cla()
            axs[1].set_title("Real-time Torque Data (Adjusted)")
            axs[1].set_ylabel("Torque (Nm)")
            axs[1].set_xlabel("Time Step")
            axs[1].grid()
            axs[1].plot(np.array(torque_data), label=["Tx", "Ty", "Tz"])
            axs[1].legend(loc="upper right")

            plt.pause(0.01)  # Refresh plot
            print(f"Force: {adjusted_force}, Torque: {adjusted_torque}")

        except Exception as e:
            print(f"Error during data collection: {e}")
            break

    plt.ioff()  # Disable interactive mode
    plt.show()

# Main function
def main():
    sensor = ForceSensor("/dev/ttyUSB0", np.zeros(3), np.zeros(3))
    sensor.force_sensor_setup()

    # Calibration phase
    force_offset, torque_offset = calibrate_force_sensor(sensor)
    if force_offset is None or torque_offset is None:
        print("Calibration failed. Exiting...")
        return

    # Data reading and visualization phase
    read_force_sensor(sensor, force_offset, torque_offset, duration=20)

if __name__ == "__main__":
    main()
