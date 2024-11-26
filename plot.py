import pandas as pd
import matplotlib.pyplot as plt

# Function to plot FT sensor data
def plot_ft_sensor_data(file_path):
    try:
        # Load data from CSV
        data = pd.read_csv(file_path)
        print("Data Preview:")
        print(data.head())

        # Debug: Print column names
        print("Column Names:", data.columns)

        # Extract columns
        Fx = data["Fx"]
        Fy = data["Fy"]
        Fz = data["Fz"]
        F_magnitude = data["F_magnitude"]
        Tx = data["Tx"]
        Ty = data["Ty"]
        Tz = data["Tz"]
        T_magnitude = data["T_magnitude"]

        # Debug: Check if data is being read correctly
        print("Sample Data:")
        print("Fx:", Fx[:5].values)
        print("Fy:", Fy[:5].values)
        print("Fz:", Fz[:5].values)
        print("F_magnitude:", F_magnitude[:5].values)
        print("Tx:", Tx[:5].values)
        print("Ty:", Ty[:5].values)
        print("Tz:", Tz[:5].values)
        print("T_magnitude:", T_magnitude[:5].values)

        # Create subplots for forces and torques
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))

        # Plot force data
        # axs[0].plot(Fx, label="Fx (N)")
        # axs[0].plot(Fy, label="Fy (N)")
        # axs[0].plot(Fz, label="Fz (N)")
        axs[0].plot(F_magnitude, label="F Magnitude (N)", linestyle="--")
        axs[0].set_title("Force Data")
        axs[0].set_xlabel("Sample")
        axs[0].set_ylabel("Force (N)")
        axs[0].legend()
        axs[0].grid()

        # Plot torque data
        # axs[1].plot(Tx, label="Tx (Nm)")
        # axs[1].plot(Ty, label="Ty (Nm)")
        # axs[1].plot(Tz, label="Tz (Nm)")
        axs[1].plot(T_magnitude, label="T Magnitude (Nm)", linestyle="--")
        axs[1].set_title("Torque Data")
        axs[1].set_xlabel("Sample")
        axs[1].set_ylabel("Torque (Nm)")
        axs[1].legend()
        axs[1].grid()

        # Adjust layout and display
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except KeyError as e:
        print(f"Missing expected column in data: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")

# Main function to plot
if __name__ == "__main__":
    # file_path = "ft_sensor_data.csv"  # Replace with your file path
    file_path = "data/20241123/195358/ft_sensor_data.csv"  # Replace with your file path
    plot_ft_sensor_data(file_path)
