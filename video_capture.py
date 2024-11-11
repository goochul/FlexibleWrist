import cv2
import time

# Set up the video capture
cap = cv2.VideoCapture(0)  # Use the default webcam
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = 30  # Desired frame rate

# Set up the codec and output file name
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_filename = 'webcam_recording.avi'
out = cv2.VideoWriter(output_filename, fourcc, frame_rate, (frame_width, frame_height))

# Start time of the recording
start_time = time.time()
record_duration = 30  # Duration in seconds

print("Recording... Press 'q' to stop early.")

while True:
    ret, frame = cap.read()
    if not ret:
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
