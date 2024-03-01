import cv2
import mediapipe as mp
import pyautogui

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
smooth_factor = 0.5  # Smoothing factor for mouse movement
data = []
subdata = []

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark
        # print(f'landmarks[474]{landmarks[474].x} , {landmarks[474].y}')
        # print(f'landmarks[475]{landmarks[475].x} , {landmarks[475].y}')
        # print(f'landmarks[476]{landmarks[476].x} , {landmarks[476].y}')
        # print(f'landmarks[477]{landmarks[477].x} , {landmarks[477].y}')
        for id, landmark in enumerate(landmarks[474:478]):            
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)  
            # cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if id == 1:
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                subdata.append([screen_x,screen_y])
                current_x, current_y = pyautogui.position()
                smooth_x = current_x + (screen_x - current_x) * smooth_factor
                smooth_y = current_y + (screen_y - current_y) * smooth_factor
                # pyautogui.moveTo(smooth_x, smooth_y)
        #         print(smooth_x, smooth_y)
        # print(subdata)

        # left = [landmarks[145], landmarks[159]]
        # for landmark in left:
        #     x = int(landmark.x * frame_w)
        #     y = int(landmark.y * frame_h)
        #     cv2.circle(frame, (x, y), 3, (0, 255, 255))

        right_eye_boundary = [landmarks[382],landmarks[384], landmarks[385], landmarks[386], landmarks[387], landmarks[388], landmarks[390], landmarks[374]]
        for landmark in right_eye_boundary:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 1, (255, 0, 0))

        # if (left[0].y - left[1].y) < 0.004:
        #     pyautogui.click()
        #     pyautogui.sleep(1)

    cv2.imshow('Eye Controlled Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()
print(data)