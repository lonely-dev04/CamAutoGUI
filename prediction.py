import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,min_tracking_confidence=0.5)
gen_points = ["Wrist", "Thumb CMC", "Thumb MCP", "Thumb IP", "Thumb Tip","Index Finger MCP", "Index Finger PIP", "Index Finger DIP", "Index Finger Tip","Middle Finger MCP", "Middle Finger PIP", "Middle Finger DIP", "Middle Finger Tip", "Ring Finger MCP", "Ring Finger PIP", "Ring Finger DIP", "Ring Finger Tip", "Pinky MCP", "Pinky PIP", "Pinky DIP", "Pinky Tip"]
dataset = []

# Process uploaded image
image = cv2.imread('dataset/5.png')
frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = hands.process(frame_rgb)

# Perform hand sign prediction based on results
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Calculate centroid (center position) of hand landmarks
        centroid_x = sum([lm.x for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark)
        centroid_y = sum([lm.y for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark)
        centroid_x = int(centroid_x * image.shape[1])
        centroid_y = int(centroid_y * image.shape[0])
        # Draw centroid on frame
        cv2.circle(image, (centroid_x, centroid_y), 5, (255, 0, 0), -1)
        # gui.moveTo(centroid_x, centroid_y)

        landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])

        # Calculate distance between thumb and index finger landmarks
        wrist = landmarks[0]
        thumb_cmc = landmarks[1]
        thumb_mcp = landmarks[2]
        thumb_ip = landmarks[3]
        thumb_tip = landmarks[4]
        index_finger_mcp = landmarks[5]
        index_finger_pip = landmarks[6]
        index_finger_dip = landmarks[7]
        index_finger_tip = landmarks[8]
        middle_finger_mcp = landmarks[9]
        middle_finger_pip = landmarks[10]
        middle_finger_dip = landmarks[11]
        middle_finger_tip = landmarks[12]
        ring_finger_mcp = landmarks[13]
        ring_finger_pip = landmarks[14]
        ring_finger_dip = landmarks[15]
        ring_finger_tip = landmarks[16]
        pinky_mcp = landmarks[17]
        pinky_pip = landmarks[18]
        pinky_dip = landmarks[19]
        pinky_tip = landmarks[20]
        
        # for i, label in enumerate(gen_points):
        #     print(f'{label}: {landmarks[i]}')

        # for i in range(len(landmarks) - 1):
        #     dist = np.linalg.norm(landmarks[i] - landmarks[i + 1])
        #     print(f"{gen_points[i]} - {gen_points[i+1]}: {dist}")
        
        for i in range(len(landmarks) - 1):
            dist = np.linalg.norm(landmarks[i] - landmarks[i + 1])
            dataset.append(dist)

        print(dataset)        
        # print(distance_1)
        # dist.append(distance_1)
        # if distance_1 < 0.075:
        #     gesture = "This is super"
        # else:
        #     gesture = "Normal"
        # cv2.putText(image, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,cv2.LINE_AA)
        # Store current landmarks for next iteration
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

# Display the uploaded image
cv2.imshow('Uploaded Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
