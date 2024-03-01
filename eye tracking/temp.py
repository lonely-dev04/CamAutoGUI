import cv2
import mediapipe as mp
import pyautogui as gui
import numpy as np

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = gui.size()
smooth_factor = 0.5
data = []
dist = {}
subdata = []

def find_circle_center(points):
    # Extract coordinates of the four points
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]
    
    # Equations of two circles formed by three points each
    circle1 = np.array([[2 * (x2 - x1), 2 * (y2 - y1)],
                        [2 * (x3 - x1), 2 * (y3 - y1)]])
    circle2 = np.array([[2 * (x3 - x2), 2 * (y3 - y2)],
                        [2 * (x4 - x2), 2 * (y4 - y2)]])
    
    # Calculate the constants for the equations
    constants1 = np.array([x2**2 - x1**2 + y2**2 - y1**2,
                           x3**2 - x1**2 + y3**2 - y1**2])
    constants2 = np.array([x3**2 - x2**2 + y3**2 - y2**2,
                           x4**2 - x2**2 + y4**2 - y2**2])
    
    # Solve the equations to find the center of the circles
    center1 = np.linalg.solve(circle1, constants1)
    center2 = np.linalg.solve(circle2, constants2)
    
    # Return the average of the centers as the estimated center of the circle
    center = np.mean([center1, center2], axis=0)
    
    return center

cnt = 0
while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    
    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            # print(f'point-{id}: x:{x} , y:{y}')
            subdata.append([landmark.x,landmark.y])
            # cv2.circle(frame, (x,y), 3, (0, 255, 0))
        # Example points on the circumference of a circle
        points = subdata
        # Find the center of the circle
        center = find_circle_center(points)
        centX = center[0]
        centY = center[1]
        center_x = int(center[0] * frame_w)
        center_y = int(center[1] * frame_h)
        # print(f"Center of the circle: x:{centX} , y:{centY}")
        cv2.circle(frame, (center_x,center_y), 3, (0, 0, 255))
        # gui.moveTo(center_x/2,center_y/2)
        subdata = []
        
        # # right_eye_boundary = [landmarks[382],landmarks[384], landmarks[385], landmarks[386], landmarks[387], landmarks[388], landmarks[390], landmarks[374]]
        # for id, landmark_id in enumerate([382, 384, 385, 386, 387, 388, 390, 374]):
        #     landmark = landmarks[landmark_id]
        #     x = int(landmark.x * frame_w)
        #     y = int(landmark.y * frame_h)
        #     diffX = np.abs(centX - landmark.x)
        #     diffY = np.abs(centY - landmark.y)
        #     # Define thresholds for near and far
        #     near_threshold = 0.01
        #     far_threshold = 0.02
        #     # Calculate Euclidean distance
        #     distance = np.linalg.norm([diffX, diffY])
        #     # Determine accuracy based on distance
        #     if distance < near_threshold:
        #         accuracy = "High"
        #     elif distance < far_threshold:
        #         accuracy = "Medium"
        #     else:
        #         accuracy = "Low"
        #     print(f"Difference of landmark[{landmark_id}] from center: x: {diffX}, y: {diffY}, Accuracy: {accuracy}")
        #     cv2.circle(frame, (x, y), 3, (255, 0, 0))
        #     dist[landmark_id] = [(x,y)]
            
        # all_data_points = []
        # for data_points in dist.values():
        #     all_data_points.extend(data_points)

        # # Calculate the mean (average) of x and y coordinates for all data points
        # all_x_coordinates = [point[0] for point in all_data_points]
        # all_y_coordinates = [point[1] for point in all_data_points]
        # overall_mean_x = sum(all_x_coordinates) / len(all_x_coordinates)
        # overall_mean_y = sum(all_y_coordinates) / len(all_y_coordinates)

        # print(f"Overall Mean (x, y) for all landmarks: ({overall_mean_x:.4f}, {overall_mean_y:.4f})")
        
        # distance = []
        # print("--------------------------------------------------------------------\n")
        # if cnt == 5:
        #     break
        # cnt += 1
        
    cv2.imshow("Center of the pupil", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
cam.release
print(data)