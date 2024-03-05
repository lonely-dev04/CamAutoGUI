import cv2
import math
import mediapipe as mp
import numpy as np
import pyautogui as gui

hands = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

cam = cv2.VideoCapture(0)
data = []
subdata = []

#bias and config
Xbias = 20
Ybias = 15

previous_points = {}

# Function to calculate Euclidean distance between two points
def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def find_h_w(tl, tr, br, bl):
    # Calculate distances between opposite corners
    distance_1 = euclidean_distance(tl[0], tl[1], tr[0], tr[1])  # Distance between top-left and top-right corners
    distance_2 = euclidean_distance(br[0], br[1], bl[0], bl[1])  # Distance between bottom-right and bottom-left corners
    
    # Width is the larger of the two distances
    width = max(distance_1, distance_2)

    # Height is the smaller of the two distances
    height = min(distance_1, distance_2)

    return width, height

def predict(c,n,s,e,w, threshold=15):
    dist_1 = math.sqrt((n[0]-c[0])**2 + (n[1]-c[1])**2)
    dist_2 = math.sqrt((s[0]-c[0])**2 + (s[1]-c[1])**2)
    dist_3 = math.sqrt((e[0]-c[0])**2 + (e[1]-c[1])**2)
    dist_4 = math.sqrt((w[0]-c[0])**2 + (w[1]-c[1])**2)
    
    min_dist = min(dist_1, dist_2, dist_3, dist_4)
    
    if min_dist <= threshold:
        if min_dist == dist_1:
            return 'near south'
        elif min_dist == dist_2:
            return 'near north'
        elif min_dist == dist_3:
            return 'near east'
        elif min_dist == dist_4:
            return 'near west'
    else:
        return 'center'

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
        north = landmarks[475]
        south = landmarks[374]
        east = landmarks[466]
        west = landmarks[414]
        
        #West based prediction
        w = (int(west.x * frame_w), int(west.y * frame_h))
        n = (int(w[0]+Xbias), int(w[1]+Ybias))
        s = (int(w[0]+Xbias), int(w[1]-Ybias))
        e = (int(w[0]+(2*Xbias)), int(w[1]))
        tl = (w[0], n[1])
        tr = (e[0], n[1])
        br = (e[0], s[1])
        bl = (w[0], s[1])

        #Basic Distance
        cv2.circle(frame, n, 3 , (0,255,0))
        cv2.circle(frame, s, 3 , (0,255,0))
        cv2.circle(frame, e, 3 , (0,255,0))
        cv2.circle(frame, w, 3 , (0,255,0))
        
        #borders and points
        cv2.circle(frame, tl, 3 , (255,0,0))
        cv2.circle(frame, tr, 3 , (255,0,0))
        cv2.circle(frame, br, 3 , (255,0,0))
        cv2.circle(frame, bl, 3 , (255,0,0))
        
        cv2.rectangle(frame, tl, br, (255,0,255), 1)
        p_w, p_h = find_h_w(tl, tr, br, bl)
        
        #Finding center         
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            subdata.append([landmark.x,landmark.y])
        points = subdata
        center = find_circle_center(points)
        centX = center[0]
        centY = center[1]
        center_x = int(center[0] * frame_w)
        center_y = int(center[1] * frame_h)
        cv2.circle(frame, (center_x,center_y), 3, (0, 0, 255))
        p_x = int(centX * p_w)
        p_y = int(centY * p_h)
        gui.moveTo(p_x, p_y)
        print(f'Predicted x and y coordinates {p_x} and {p_y}')
        # cv2.addText(frame, point, (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
         
        
        #lines to center
        cv2.line(frame, pt1=(center_x,center_y), pt2=n, color=(255,0,0), thickness=1)
        cv2.line(frame, pt1=(center_x,center_y), pt2=s, color=(255,0,0), thickness=1)
        cv2.line(frame, pt1=(center_x,center_y), pt2=e, color=(255,0,0), thickness=1)
        cv2.line(frame, pt1=(center_x,center_y), pt2=w, color=(255,0,0), thickness=1)
        subdata = []
        
        #Predict the place
        text = predict((center_x,center_y), n, s, e, w)
        cv2.putText(frame, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    cv2.imshow("Center of the pupil", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
cam.release()
