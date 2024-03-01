import cv2
import mediapipe as mp

hands = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

cam = cv2.VideoCapture(0)

differ_bias = 2  # Threshold value for change in position

previous_point = (0, 0)
moved = False
text = "Not moved"  # Initialize text outside the loop

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hands.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    if landmark_points:
        landmarks = landmark_points[0].landmark
        point = landmarks[5]
        x = int(point.x * frame_w)
        y = int(point.y * frame_h)
        cv2.circle(frame, (x, y), 3, (0, 255, 0))
        if abs(x - previous_point[0]) > differ_bias or abs(y - previous_point[1]) > differ_bias:
            if not moved:  # Check if state has changed
                print("moved")
                text = "Moved"
                moved = True
        else:
            if moved:  # Check if state has changed
                print("not moved")
                text = "Not moved"
                moved = False
        previous_point = (x, y)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Move text outside the if condition
    cv2.imshow("Center of the pupil", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()
