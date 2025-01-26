import os
from datetime import datetime
from ForceSensor import ForceSensor
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

# Real-time plotting
def setup_realtime_plot():
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Real-Time Force and Torque Magnitudes")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Magnitude (N, Nm)")
    ax.grid(True)
    return fig, ax

def update_plot(ax, time_data, force_magnitudes, torque_magnitudes):
    ax.clear()
    ax.plot(time_data, force_magnitudes, label="Force Magnitude (N)", color="blue")
    ax.plot(time_data, torque_magnitudes, label="Torque Magnitude (Nm)", color="red")
    ax.set_title("Real-Time Force and Torque Magnitudes")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Magnitude (N, Nm)")
    ax.grid(True)
    ax.legend()
    plt.pause(0.01)

# Data reading function with real-time plotting
def read_force_sensor(sensor, force_offset, torque_offset, max_samples=10000, plot_realtime=True):
    force_data = []
    torque_data = []
    time_data = []
    start_time = time.time()

    # Setup real-time plot
    fig, ax = None, None
    if plot_realtime:
        fig, ax = setup_realtime_plot()

    print(f"\nCollecting {max_samples} sensor data points:")

    for i in range(max_samples):
        try:
            raw_force, raw_torque = sensor.get_force_obs()
            adjusted_force = raw_force - force_offset
            adjusted_torque = raw_torque - torque_offset

            # Calculate magnitudes
            force_magnitude = np.linalg.norm(adjusted_force)
            torque_magnitude = np.linalg.norm(adjusted_torque)

            # Append data
            force_data.append(list(adjusted_force) + [force_magnitude])
            torque_data.append(list(adjusted_torque) + [torque_magnitude])
            time_data.append(time.time() - start_time)

            # Update plot in real-time
            if plot_realtime and i % 10 == 0:  # Update plot every 10 samples
                update_plot(ax, time_data, [f[3] for f in force_data], [t[3] for t in torque_data])

        except Exception as e:
            print(f"Error during data collection: {e}")
            break

    if plot_realtime:
        plt.ioff()  # Disable interactive mode after collection
        plt.show()

    return force_data, torque_data

# Save data to CSV
def save_to_csv(force_data, torque_data, base_path="data"):
    # Get today's date and current time
    today_date = datetime.now().strftime("%Y%m%d")
    current_time = datetime.now().strftime("%H%M%S")

    # Create folder structure: data/YYYYMMDD/HHMMSS
    save_dir = os.path.join(base_path, today_date, current_time)
    os.makedirs(save_dir, exist_ok=True)

    # Define file path
    file_path = os.path.join(save_dir, "ft_sensor_data.csv")

    # Save data to CSV
    df = pd.DataFrame(
        np.hstack((force_data, torque_data)),
        columns=["Fx", "Fy", "Fz", "F_magnitude", "Tx", "Ty", "Tz", "T_magnitude"]
    )
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

# Main function
def main():
    sensor = ForceSensor("/dev/ttyUSB0", np.zeros(3), np.zeros(3))
    sensor.force_sensor_setup()

    # Calibration phase
    force_offset, torque_offset = calibrate_force_sensor(sensor)
    if force_offset is None or torque_offset is None:
        print("Calibration failed. Exiting...")
        return

    # Data reading phase with real-time plot
    force_data, torque_data = read_force_sensor(sensor, force_offset, torque_offset, max_samples=10000, plot_realtime=True)

    # Save data to CSV
    save_to_csv(force_data, torque_data)

if __name__ == "__main__":
    main()
