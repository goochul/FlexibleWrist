import os
import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ForceSensor import ForceSensor
import pyrealsense2 as rs
import cv2


# Global variables
force_data = []
torque_data = []
timestamps = []
global_start_time = None
force_sensor = None
max_samples = 2000

# Force-Torque Sensor Functions
def calibrate_force_sensor(sensor, num_samples=100, sleep_time=0.01):
    """
    Calibrate the force sensor by calculating force and torque offsets.
    """
    try:
        readings = []
        print("Performing calibration...")
        for _ in range(num_samples):
            force, torque = sensor.get_force_obs()
            readings.append((force, torque))
            time.sleep(sleep_time)

        # Calculate offsets
        force_offset = np.mean([r[0] for r in readings], axis=0)
        torque_offset = np.mean([r[1] for r in readings], axis=0)
        print(f"Calibration complete. Force offset: {force_offset}, Torque offset: {torque_offset}")
        return force_offset, torque_offset
    except Exception as e:
        print(f"Error during calibration: {e}")
        return None, None

def initialize_force_sensor_for_calibration():
    """
    Initialize the sensor for calibration without applying offsets.
    """
    try:
        sensor = ForceSensor("/dev/ttyUSB0", np.zeros(3), np.zeros(3))
        sensor.force_sensor_setup()
        print("Sensor initialized for calibration.")
        return sensor
    except Exception as e:
        print(f"Error initializing sensor: {e}")
        return None

def monitor_ft_sensor(sensor, force_offset, torque_offset):
    """
    Monitor the force-torque sensor with calibration offsets and record data.
    """
    global force_data, torque_data, timestamps, global_start_time
    print("Starting FT sensor monitoring thread.")

    try:
        while len(force_data) < max_samples:
            # Read adjusted force-torque data
            raw_force, raw_torque = sensor.get_force_obs()
            adjusted_force = raw_force - force_offset
            adjusted_torque = raw_torque - torque_offset

            elapsed_time = time.time() - global_start_time
            force_magnitude = np.linalg.norm(adjusted_force)
            torque_magnitude = np.linalg.norm(adjusted_torque)

            # Log the data
            timestamps.append(elapsed_time)
            force_data.append(force_magnitude)
            torque_data.append(torque_magnitude)

            time.sleep(0.01)

    except Exception as e:
        print(f"Error in monitor_ft_sensor: {e}")

# Video recording function using RealSense camera
def record_video(output_path, duration=60, fps=30):
    print("Recording video...")

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, fps)

    # Start streaming
    pipeline.start(config)

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (960, 540))

    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # Write the frame to the video file
            out.write(color_image)

    finally:
        pipeline.stop()
        out.release()
        print(f"Video saved to {output_path}")


# Save Data to CSV
def save_data_to_csv():
    date_folder = time.strftime("%Y%m%d")
    time_folder = time.strftime("%H%M%S")
    data_folder = os.path.join("data", date_folder, time_folder)
    os.makedirs(data_folder, exist_ok=True)

    force_df = pd.DataFrame({"Timestamp": timestamps, "Force Magnitude": force_data})
    force_df.to_csv(os.path.join(data_folder, "force_data.csv"), index=False)

    torque_df = pd.DataFrame({"Timestamp": timestamps, "Torque Magnitude": torque_data})
    torque_df.to_csv(os.path.join(data_folder, "torque_data.csv"), index=False)

    print(f"Data saved to folder: {data_folder}")
    return data_folder


# Plot Data
def plot_force_and_torque(data_folder):
    plt.figure(figsize=(10, 6))

    # Plot force magnitude
    plt.plot(timestamps, force_data, label="Force Magnitude (N)", color="blue")

    # Plot torque magnitude
    plt.plot(timestamps, torque_data, label="Torque Magnitude (Nm)", color="orange")

    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude")
    plt.title("Force and Torque Magnitudes Over Time")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(data_folder, "force_torque_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")


def main():
    global global_start_time

    # Initialize the sensor for calibration
    sensor = initialize_force_sensor_for_calibration()
    if sensor is None:
        print("Sensor initialization failed. Exiting...")
        return

    # Perform calibration
    force_offset, torque_offset = calibrate_force_sensor(sensor)
    if force_offset is None or torque_offset is None:
        print("Calibration failed. Exiting...")
        return

    global_start_time = time.time()

    # Create data folder path
    date_folder = time.strftime("%Y%m%d")
    time_folder = time.strftime("%H%M%S")
    data_folder = os.path.join("data", date_folder, time_folder)
    os.makedirs(data_folder, exist_ok=True)

    # Start video recording thread
    video_output_path = os.path.join(data_folder, "realsense_recording.mp4")
    video_thread = threading.Thread(target=record_video, args=(video_output_path, 60, 30), daemon=True)
    video_thread.start()

    # Start monitoring thread
    monitoring_thread = threading.Thread(
        target=monitor_ft_sensor,
        args=(sensor, force_offset, torque_offset),
        daemon=True
    )
    monitoring_thread.start()

    # Wait for threads to finish
    monitoring_thread.join()
    video_thread.join()

    # Save and plot data after threads finish
    data_folder = save_data_to_csv()
    plot_force_and_torque(data_folder)

    print("Process complete.")


if __name__ == "__main__":
    main()
