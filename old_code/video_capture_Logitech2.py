import cv2
import time
import os

camera_index = 0  # Use the correct index for your Logitech camera
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"Failed to open the Logitech camera at /dev/video{camera_index}.")
    exit()

# Disable auto exposure (if supported) and set manual exposure
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # This value might vary depending on your camera/OS
cap.set(cv2.CAP_PROP_EXPOSURE, -6)         # Try adjusting this value to reduce brightness

# Optionally, adjust brightness directly (if supported)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.3)
print("Brightness set to:", cap.get(cv2.CAP_PROP_BRIGHTNESS))
print("Exposure set to:", cap.get(cv2.CAP_PROP_EXPOSURE))

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_filename = 'Logitech_camera_recording.mp4'
out = cv2.VideoWriter(output_filename, fourcc, frame_rate, (frame_width, frame_height))
if not out.isOpened():
    print("Failed to initialize the VideoWriter.")
    cap.release()
    exit()

start_time = time.time()
record_duration = 1700  # Duration in seconds

print(f"Recording from Logitech camera (/dev/video{camera_index})... Press 'q' to stop early.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Flip the frame horizontally and vertically
    flipped_frame = cv2.flip(frame, -2)
    out.write(flipped_frame)
    cv2.imshow('Recording', flipped_frame)
    
    if time.time() - start_time > record_duration or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

save = input("Recording complete. Do you want to save the recording? (yes/no): ").strip().lower()
if save != 'yes':
    try:
        os.remove(output_filename)
        print("Recording deleted.")
    except OSError as e:
        print(f"Error: {e.strerror}")
else:
    print(f"Recording saved as {output_filename}.")
