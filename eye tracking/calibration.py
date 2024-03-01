import cv2
import mediapipe as mp
import numpy as np

hands = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

cam = cv2.VideoCapture(0)
data = []
subdata = []

#bias and config
Xbias = 50
Ybias = 50

previous_points = {}

def update_positions(center_x, center_y):
    #max NESW pivot
    n = (center_x,center_y - Ybias)
    s = (center_x,center_y + Ybias)
    e = (center_x + Xbias,center_y)
    w = (center_x - Xbias,center_y)
    
    #boundary
    e_n = (e[0],n[1])
    e_s = (e[0],s[1])
    w_s = (w[0],s[1])
    w_n = (w[0],n[1])
    
    return {'n':n, 's':s, 'e':e, 'w':w, 'e_n':e_n, 'e_s':e_s, 'w_n':w_n, 'w_s':w_s}

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

while True:
    _ , frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hands.process(rgb_frame)
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
        direct = update_positions(center_x,center_y)
        print(direct)
        cv2.circle(frame, direct['n'], 3 , (0,255,0))
        cv2.circle(frame, direct['s'], 3 , (0,255,0))
        cv2.circle(frame, direct['e'], 3 , (0,255,0))
        cv2.circle(frame, direct['w'], 3 , (0,255,0))
        # cv2.line(frame, pt1=n, pt2=s ,color=(255,0,0),thickness=1)
        # cv2.line(frame, pt1=e, pt2=w ,color=(255,0,0),thickness=1)
        cv2.line(frame,pt1= direct['w_n'],pt2=direct['e_n'],color=(255,0,0),thickness=1)
        cv2.line(frame,pt1=direct['w_s'],pt2=direct['e_s'],color=(255,0,0),thickness=1)
        cv2.line(frame,pt1=direct['w_n'],pt2=direct['w_s'],color=(255,0,0),thickness=1)
        cv2.line(frame,pt1=direct['e_n'],pt2=direct['e_s'],color=(255,0,0),thickness=1)
        # gui.moveTo(center_x/2,center_y/2)
        subdata = []
        
    cv2.imshow("Center of the pupil", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
cam.release
    
    
    
    

