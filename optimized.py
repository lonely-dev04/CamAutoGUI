import cv2
import mediapipe as mp
import pyautogui as gui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set window size
window_width = 1930
window_height = 1090
cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)
cv2.resizeWindow('MediaPipe Hands', window_width, window_height)

# Before the while loop, get the webcam's resolution
ret, frame = cap.read()
if ret:
    webcam_width = frame.shape[1]
    webcam_height = frame.shape[0]

# Before the while loop
prev_screen_x, prev_screen_y = 0, 0  # Initialize previous screen positions
prev_middle_y = 0  # Initialize previous middle finger y position
smoothing_factor = 0.5  # Increase for more smoothing
movement_threshold = 10  # Minimum pixels the cursor must move to update its position
scroll_threshold = 0.1  # Threshold for detecting scroll gesture (adjust as needed)
left_click_down = False  # Initialize a variable to track the state of the left mouse button
gesture_listening_activated = False  # Add a new variable to track if gesture listening is activated

# Define a function to detect thumbs-up gesture

while cap.isOpened():
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # Draw hand landmarks on frame and move cursor
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate centroid of hand landmarks
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            centroid_x = sum(x_coords) / len(x_coords)
            centroid_y = sum(y_coords) / len(y_coords)
            
            # Map the centroid coordinates to the screen resolution, inverting the x-axis
            screen_x = int(((webcam_width - centroid_x * webcam_width) / webcam_width) * window_width)
            screen_y = int((centroid_y * webcam_height / webcam_height) * window_height)
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                
            # Apply smoothing
            screen_x_smoothed = int(prev_screen_x + smoothing_factor * (screen_x - prev_screen_x))
            screen_y_smoothed = int(prev_screen_y + smoothing_factor * (screen_y - prev_screen_y))

            # Apply the movement threshold
            movement_x = abs(screen_x_smoothed - prev_screen_x)
            movement_y = abs(screen_y_smoothed - prev_screen_y)

            if movement_x > movement_threshold or movement_y > movement_threshold:
                # Move cursor using smoothed coordinates
                # print(f'Smoothed screen:{screen_x_smoothed, screen_y_smoothed}')
                gui.moveTo(screen_x_smoothed, screen_y_smoothed)
                # Update previous screen positions
                prev_screen_x, prev_screen_y = screen_x_smoothed, screen_y_smoothed
            else:
                # If movement is below threshold, don't update cursor position
                # Optionally, you can slightly adjust prev_screen_x and prev_screen_y towards current position for smoother recovery
                prev_screen_x += int(movement_x * smoothing_factor)
                prev_screen_y += int(movement_y * smoothing_factor)
            
            landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
            
            # Calculate distance between thumb and index finger landmarks
            wrist = landmarks[0]
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]
            distance_1 = np.linalg.norm(thumb_tip - index_tip)
            distance_2 = np.linalg.norm(thumb_tip - pinky_tip)
            distance_3 = np.linalg.norm(wrist - middle_tip)
            distance_4 = np.linalg.norm(thumb_tip - middle_tip)

            gesture = ""
            #Predict gesture based on distance change
            if distance_1 < 0.05 and not left_click_down:
                gesture = "Left Click Hold"
                gui.mouseDown(button="left")
                left_click_down = True  # Update the state to indicate the left button is held down
                print("gui.mouseDown(button='left')")
            elif distance_1 > 0.1 and left_click_down:
                gesture = "Left Click Release"
                gui.mouseUp(button="left")
                left_click_down = False  # Update the state to indicate the left button is released
                print("gui.mouseUp(button='left')")
            elif distance_2 < 0.05:
                gesture = "Right Click"
                gui.click(button="right")
                print("gui.mouseClick(button=right)")
            if distance_3 < 0.150:
                gesture = "Scroll Down"
                print("Scroll Down")
                gui.scroll(-100)
            elif distance_4 < 0.050:
                gesture = "Scroll Up"
                print("Scroll Up")
                gui.scroll(100)
            else:
                gesture = "Nothing"
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display frame
    cv2.imshow('MediaPipe Hands', frame)

    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
