import cv2
import math
import mediapipe as mp
import numpy as np

hands = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

cam = cv2.VideoCapture(0)
data = []
subdata = []

# bias and config
Xbias = 24
Ybias = 20

def predict_eye_boundary(eye_west_point, eye_width, eye_height):
    # Calculate the eye boundary based on the west point and dimensions
    left = int(eye_west_point[0] - eye_width / 2)
    right = int(eye_west_point[0] + eye_width / 2)
    top = int(eye_west_point[1] - eye_height / 2)
    bottom = int(eye_west_point[1] + eye_height / 2)
    
    return left, right, top, bottom

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hands.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    
    if landmark_points:
        landmarks = landmark_points[0].landmark
        west = landmarks[414]  # Using the west point of the eye
        
        # Calculate eye boundary based on the west point and bias
        eye_west_point = (int(west.x * frame_w), int(west.y * frame_h))
        eye_left, eye_right, eye_top, eye_bottom = predict_eye_boundary(eye_west_point, Xbias * 2, Ybias * 2)
        
        # Draw rectangle around predicted eye boundary
        cv2.rectangle(frame, (eye_left, eye_top), (eye_right, eye_bottom), (0, 255, 0), 2)
        
    cv2.imshow("Eye Boundary Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()
