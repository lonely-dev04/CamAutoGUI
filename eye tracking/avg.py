import cv2
import numpy as np
import pyautogui

pyautogui.FAILSAFE = False
# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Set up video capture device (change index as needed)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Example: Assuming iris_center and eye_boundary are already detected in the frame
    iris_center = (100, 100)  # Example coordinates
    eye_boundary = (200, 200)  # Example coordinates

    # Calculate distance between iris center and eye boundary
    distance = calculate_distance(iris_center, eye_boundary)

    # Map distance to cursor movement
    sensitivity_factor = 1  # Adjust as needed
    cursor_movement = (distance * sensitivity_factor, distance * sensitivity_factor)

    # Apply cursor movement to actual cursor position
    # Example: Use library to move cursor (e.g., pyautogui for desktop cursor movement)
    pyautogui.move(cursor_movement[0], cursor_movement[1])

    # Display the frame with marked iris center and eye boundary
    cv2.circle(frame, iris_center, 2, (0, 255, 0), thickness=2)  # Mark iris center
    cv2.circle(frame, eye_boundary, 2, (0, 0, 255), thickness=2)  # Mark eye boundary
    cv2.imshow("Eye Tracking", frame)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
