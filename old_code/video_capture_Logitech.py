import cv2
import time
import os

# Specify the camera index for Logitech (update this based on your system, e.g., 6 or 7)
camera_index = 4  # Replace with the correct index for your Logitech camera

# Set up the video capture for Logitech camera
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"Failed to open the Logitech camera at /dev/video{camera_index}.")
    exit()

cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
new_af = cap.get(cv2.CAP_PROP_AUTOFOCUS)
print("New autofocus setting:", new_af)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = 30  # Desired frame rate


# Set up the codec and output file name
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_filename = 'Logitech_camera_recording.mp4'
out = cv2.VideoWriter(output_filename, fourcc, frame_rate, (frame_width, frame_height))

if not out.isOpened():
    print("Failed to initialize the VideoWriter.")
    cap.release()
    exit()

# Start time of the recording
start_time = time.time()
record_duration = 1700  # Duration in seconds

print(f"Recording from Logitech camera (/dev/video{camera_index})... Press 'q' to stop early.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break


    # Flip the frame horizontally + vertically
    # flipped_frame = cv2.flip(frame, 1)

    # Write the frame to the output file
    # out.write(flipped_frame)
    out.write(frame)
    flipped_frame = frame

    # Display the frame (optional)
    cv2.imshow('Recording', flipped_frame)

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
