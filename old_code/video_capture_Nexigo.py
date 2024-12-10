import cv2
import time
import os

# Specify the camera index for Nexigo (update this based on your system, e.g., 6 or 7)
camera_index = 6  # Replace with the correct index for your Nexigo camera

# Set up the video capture for Nexigo camera
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"Failed to open the Nexigo camera at /dev/video{camera_index}.")
    exit()

# cap.set(cv2.CAP_PROP_BRIGHTNESS, -1)  # Reset brightness to default
# cap.set(cv2.CAP_PROP_CONTRAST, -1)    # Reset contrast to default
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Enable auto exposure


# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Set auto exposure
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # Disable auto exposure (0.25 is manual mode)
# cap.set(cv2.CAP_PROP_EXPOSURE, -10)  # Set a lower exposure value (experiment with this)

# cap.set(cv2.CAP_PROP_BRIGHTNESS, 10)  # Adjust brightness (0–100 scale; adjust as needed)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = 30  # Desired frame rate

# Set up the codec and output file name
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_filename = 'nexigo_camera_recording.mp4'
out = cv2.VideoWriter(output_filename, fourcc, frame_rate, (frame_width, frame_height))

if not out.isOpened():
    print("Failed to initialize the VideoWriter.")
    cap.release()
    exit()

# Start time of the recording
start_time = time.time()
record_duration = 170  # Duration in seconds

print(f"Recording from Nexigo camera (/dev/video{camera_index})... Press 'q' to stop early.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Write the frame to the output file
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('Recording', frame)

    # Check if the recording time has reached the limit or if 'q' is pressed
    if time.time() - start_time > record_duration or cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

# Ask the user if they want to save the file
save = input("Recording complete. Do you want to save the recording? (yes/no): ").strip().lower()
if save != 'yes':
    # Delete the file if the user doesn't want to save it
    try:
        os.remove(output_filename)
        print("Recording deleted.")
    except OSError as e:
        print(f"Error: {e.strerror}")
else:
    print(f"Recording saved as {output_filename}.")
