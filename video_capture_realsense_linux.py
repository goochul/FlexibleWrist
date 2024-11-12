import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

# Initialize RealSense pipeline with retries
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Adjust resolution and framerate as needed

max_retries = 5
retry_count = 0
while retry_count < max_retries:
    try:
        # Start streaming
        pipeline.start(config)
        print("RealSense camera initialized.")
        break
    except RuntimeError as e:
        print(f"Error initializing RealSense camera: {e}")
        retry_count += 1
        time.sleep(1)  # Wait before retrying
else:
    print("Failed to initialize RealSense camera after multiple attempts.")
    exit(1)

# Set up video writer
output_filename = 'realsense_recording.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_rate = 30  # Set the desired frame rate
out = cv2.VideoWriter(output_filename, fourcc, frame_rate, (640, 480))

# Set recording duration
record_duration = 50  # Set the recording duration in seconds
start_time = time.time()

print("Recording... Press 'q' to stop early.")

try:
    while True:
        # Wait for a coherent frame with retries
        try:
            frames = pipeline.wait_for_frames()
        except RuntimeError as e:
            print(f"Frame didn't arrive: {e}")
            continue  # Retry getting frames
        
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Flip the image vertically
        flipped_image = cv2.flip(color_image, 0)  # 0 means flipping vertically

        # Write flipped frame to video file
        out.write(flipped_image)

        # Display the flipped frame
        cv2.imshow('RealSense Recording (Flipped)', flipped_image)

        # Check if recording time is reached or if 'q' is pressed
        if time.time() - start_time > record_duration or cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming and release resources
    pipeline.stop()
    out.release()
    cv2.destroyAllWindows()

# Ask the user if they want to save the file
save = input("Recording complete. Do you want to save the recording? (yes/no): ").strip().lower()
if save != 'yes':
    try:
        if os.path.exists(output_filename):
            os.remove(output_filename)
            print("Recording deleted.")
        else:
            print("Recording file not found for deletion.")
    except OSError as e:
        print(f"Error: {e.strerror}")
else:
    print(f"Recording saved as {output_filename}.")
