import cv2
import mediapipe as mp
import pyautogui as gui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)
prev_landmarks = None

while cap.isOpened():
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # frame = cv2.flip(frame, 1)
    # Convert BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # Draw hand landmarks on frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate centroid (center position) of hand landmarks
            centroid_x = sum([lm.x for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark)
            centroid_y = sum([lm.y for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark)
            centroid_x = int(centroid_x * frame.shape[1])
            centroid_y = int(centroid_y * frame.shape[0])
            # Draw centroid on frame
            cv2.circle(frame, (centroid_x, centroid_y), 5, (255, 0, 0), -1)
            print(centroid_x,centroid_y)
            gui.moveTo(centroid_x,centroid_y)
            
            # landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
            #
            # # Calculate distance between thumb and index finger landmarks
            # thumb_tip = landmarks[4]
            # index_tip = landmarks[8]
            # middle_tip = landmarks[12]
            # distance_1 = np.linalg.norm(thumb_tip - index_tip)
            # distance_2 = np.linalg.norm(thumb_tip - middle_tip)
            # # print(f"Distance: {distance_1}")
            # # print(f"Distance: {distance_2}")
            # x_axis = int(thumb_tip[0] * frame.shape[1])
            # y_axis = int(thumb_tip[1] * frame.shape[0])
            # print(thumb_tip)

            # gesture = ""
            # # Predict gesture based on distance change
            # if distance_1 < 0.05:
            #     gesture = "Left Click"
            #     gui.mouseDown(button="left")
            #     print("gui.mouseDown(button=left)")
            # if distance_2 < 0.05:
            #     gesture = "Right Click"
            #     gui.click(button="right")
            #     print("gui.mouseDown(button=right)")
            # if distance_1 > 0.1:
            #     gesture = "Nothing"
            #     gui.mouseUp(button="left")
            #     print("gui.mouseUp(button=left)")
            # cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Store current landmarks for next iteration
            
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Display frame
    # mirrored_frame = cv2.flip(frame, 1)
    cv2.imshow('MediaPipe Hands', frame)

    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
