from ForceSensor import ForceSensor
import numpy as np
import pandas as pd
import time

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

# Data reading function
def read_force_sensor(sensor, force_offset, torque_offset, max_samples=3000):
    force_data = []
    torque_data = []
    print(f"\nCollecting {max_samples} sensor data points:")

    for _ in range(max_samples):
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

            print(f"Force: {adjusted_force}, Magnitude: {force_magnitude}")
            print(f"Torque: {adjusted_torque}, Magnitude: {torque_magnitude}")

        except Exception as e:
            print(f"Error during data collection: {e}")
            break

    return force_data, torque_data

# Save data to CSV
def save_to_csv(force_data, torque_data, file_path="ft_sensor_data.csv"):
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

    # Data reading phase
    force_data, torque_data = read_force_sensor(sensor, force_offset, torque_offset, max_samples=3000)

    # Save data to CSV
    save_to_csv(force_data, torque_data)

if __name__ == "__main__":
    main()
