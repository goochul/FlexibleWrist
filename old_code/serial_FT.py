import os
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas to save data as CSV
from ForceSensor import ForceSensor

# Function to collect initial force readings and compute the average offset
def calculate_force_offset(sensor, num_samples=20, sleep_time=0.001):
    readings = []
    for _ in range(num_samples):
        force = sensor.get_force_obs()
        readings.append(force)
        time.sleep(sleep_time)  # Reduced sleep time to make it faster
    return np.mean(readings, axis=0)

# New function to handle the calibration process based on a flag
def initialize_force_sensor(calibrate=True, predefined_bias=np.zeros(3)):
    if calibrate:
        # Initialize the force sensor (without offset initially)
        initial_sensor = ForceSensor("/dev/ttyUSB0", np.zeros(3))
        initial_sensor.force_sensor_setup()

        # Calculate the offset based on the first 20 readings
        force_offset = calculate_force_offset(initial_sensor)
        print("Calculated force offset:", force_offset)
    else:
        # Use the predefined bias
        force_offset = predefined_bias
        print("Using predefined force offset:", force_offset)

    # Initialize the force sensor with the calculated or predefined offset
    sensor = ForceSensor("/dev/ttyUSB0", force_offset)
    sensor.force_sensor_setup()

    return sensor

# Global variables to store the force data and test count
force_data = []
data_collection_done = threading.Event()
test_count = 0

# Create folders for saving figures and data
figure_folder = "figure/"
data_folder = "data/"
os.makedirs(figure_folder, exist_ok=True)
os.makedirs(data_folder, exist_ok=True)

# Function to continuously read force sensor data
def read_force_sensor():
    global force_data
    force_data = []  # Reset force data for each test
    while not data_collection_done.is_set():
        force = force_sensor.get_force_obs()
        force_data.append(force)
        if len(force_data) > 400:  # Keep the last 400 readings
            force_data.pop(0)
        time.sleep(0.001)  # Reduced sleep time to make it faster

    # Signal that data collection is done
    data_collection_done.set()

# Function to update the plot in real-time and save it when done
def update_plot():
    global test_count
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    ax.set_title("Flexible Wrist Test")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Force (N)")
    
    while not data_collection_done.is_set():
        if force_data:
            ax.clear()
            ax.plot(np.array(force_data)[:, 0], label="Fx")
            ax.plot(np.array(force_data)[:, 1], label="Fy")
            ax.plot(np.array(force_data)[:, 2], label="Fz")
            ax.legend()
            plt.pause(0.001)  # Reduced pause time to make it faster
        time.sleep(0.001)

    # After data collection is done, save the figure and data with unique filenames
    test_count += 1
    figure_filename = os.path.join(figure_folder, f"flexible_wrist_test_{test_count}.png")
    data_filename = os.path.join(data_folder, f"force_data_{test_count}.csv")
    
    # Save the figure
    ax.clear()
    ax.plot(np.array(force_data)[:, 0], label="Fx")
    ax.plot(np.array(force_data)[:, 1], label="Fy")
    ax.plot(np.array(force_data)[:, 2], label="Fz")
    ax.legend()
    ax.set_title("Flexible Wrist Test")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Force (N)")
    plt.savefig(figure_filename)
    plt.ioff()
    plt.show()  # Show the final plot with all data

    # Save the data as a CSV file
    force_df = pd.DataFrame(force_data, columns=["Fx", "Fy", "Fz"])
    force_df.to_csv(data_filename, index=False)

    print(f"Test {test_count} completed. Figure saved as {figure_filename}. Data saved as {data_filename}.")

# Function to start a new test
def start_test():
    global data_collection_done, force_thread, plot_thread
    data_collection_done.clear()  # Clear the event to start data collection

    # Start a new thread for reading force sensor data
    force_thread = threading.Thread(target=read_force_sensor, name="ForceSensorThread")
    force_thread.daemon = True  # Set as daemon so it won't block the program from exiting
    force_thread.start()

    # Start a new thread for updating the plot
    plot_thread = threading.Thread(target=update_plot, name="PlotThread")
    plot_thread.daemon = True
    plot_thread.start()

def show_active_threads():
    """Function to display active threads."""
    print("\nActive Threads:")
    for thread in threading.enumerate():
        print(f"Thread Name: {thread.name}, Thread ID: {thread.ident}, Daemon: {thread.daemon}")

if __name__ == "__main__":
    # Set the calibration flag (True to calibrate, False to use predefined bias)
    calibration_flag = True  # Change this to False if you want to skip calibration

    # Predefined bias value
    predefined_bias = np.array([10.06580024, 5.71097629, 4.49855628]
)

    # Initialize the force sensor with or without calibration
    force_sensor = initialize_force_sensor(calibrate=calibration_flag, predefined_bias=predefined_bias)

    # Wait for user input to start the tests
    for i in range(3):  # Adjust this range to run more tests
        input(f"Press Enter to start collecting data and plotting for Test {i+1}...")

        print(f"Starting test {i+1}")
        start_test()
        time.sleep(10)  # Simulate test duration (e.g., 10 seconds)
        data_collection_done.set()  # Stop the data collection for this test
        plot_thread.join()  # Wait for the plot to be saved before starting a new test

    print("All tests completed.")
