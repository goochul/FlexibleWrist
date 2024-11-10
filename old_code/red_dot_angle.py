import cv2
import numpy as np
import math

def detect_red_dots(frame):
    # Define the red color range in BGR (since OpenCV uses BGR by default)
    lower_red = np.array([0, 0, 160])  # Adjust this value as needed
    upper_red = np.array([90, 90, 255])  # Adjust this value as needed

    # Create a mask for the red color using BGR thresholds
    mask = cv2.inRange(frame, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    red_dots = []

    for contour in contours:
        # Calculate the area of the contour to filter out noise
        area = cv2.contourArea(contour)
        if area > 50:  # Adjust this threshold as needed to filter small areas
            # Get the center of the contour
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                red_dots.append((cX, cY))

    # Sort the dots by their position on the screen for consistent processing
    red_dots.sort(key=lambda x: x[0])

    return red_dots[:3]  # Return the first three red dots found

def calculate_angle(p1, p2, p3):
    # Calculate the vectors between points
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    # Calculate the angle between vectors
    angle_rad = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
    angle_deg = math.degrees(angle_rad)

    # Make sure the angle is positive
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg

def main():
    # Open the webcam (index 0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Detect the red dots
        red_dots = detect_red_dots(frame)

        if len(red_dots) == 3:
            # If exactly three dots are found, calculate the angle between them
            angle = calculate_angle(red_dots[0], red_dots[1], red_dots[2])
            print(f"Angle between dots: {angle:.2f} degrees")

            # Draw the red dots on the frame
            for i, dot_position in enumerate(red_dots):
                cv2.circle(frame, dot_position, 10, (0, 255, 0), -1)  # Draw a green circle on each red dot
                cv2.putText(frame, f"Dot {i+1}", (dot_position[0] + 10, dot_position[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display the angle on the frame
            cv2.putText(frame, f"Angle: {angle:.2f} degrees", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow('Red Dot Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
