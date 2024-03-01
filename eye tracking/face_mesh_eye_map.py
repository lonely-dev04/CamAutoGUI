import cv2
import mediapipe as mp
import numpy as np
import pyautogui as gui

# Initialize MediaPipe FaceMesh model
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
frame_w, frame_h = gui.size()
# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face landmarks
    results = mp_face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            right_eye_indices = [i for i in range(263, 274)]
            right_eye_points = [(int(face_landmarks.landmark[i].x * frame.shape[1]), 
                                 int(face_landmarks.landmark[i].y * frame.shape[0])) 
                                for i in right_eye_indices]
            cv2.circle(frame, [np.array(right_eye_points)], 3, (255, 0, 0))

    cv2.imshow('Face Mesh', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
