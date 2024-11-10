import cv2

# Open a connection to the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was not captured correctly, ret is False
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()