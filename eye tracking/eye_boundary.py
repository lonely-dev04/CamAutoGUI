import cv2
import mediapipe as mp
import numpy as np

hands = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

cam = cv2.VideoCapture(0)
data = []
subdata = []

#bias and config
Xbias = 20
Ybias = 20

previous_points = {}

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
        # print(int(north.x * frame_w), int(north.y * frame_h))
        # n = (int(north.x * frame_w), int(north.y * frame_h))
        # s = (int(south.x * frame_w), int(south.y * frame_h))
        # e = (int(east.x * frame_w), int(east.y * frame_h))
        w = (int(west.x * frame_w), int(west.y * frame_h))
        n = (int(w[0]+Xbias), int(w[1]+Ybias))
        s = (int(w[0]+Xbias), int(w[1]-Ybias))
        e = (int(w[0]+(2*Xbias)), int(w[1]))
        
        # for landmark in [n,s,e,w]:
        # #     x = int(landmark.x * frame_w)
        # #     y = int(landmark.y * frame_h)
        # #     # print(f'point-{id}: x:{x} , y:{y}')
        # #     subdata.append([landmark.x,landmark.y])
        # #     cv2.circle(frame, (x,y), 3, (0, 255, 0))
        cv2.circle(frame, n, 3 , (0,255,0))
        cv2.circle(frame, s, 3 , (0,255,0))
        cv2.circle(frame, e, 3 , (0,255,0))
        cv2.circle(frame, w, 3 , (0,255,0))
        # cv2.line(frame, pt1=n, pt2=s ,color=(255,0,0),thickness=1)
        # cv2.line(frame, pt1=e, pt2=w ,color=(255,0,0),thickness=1)
        # cv2.line(frame,pt1= direct['w_n'],pt2=direct['e_n'],color=(255,0,0),thickness=1)
        # cv2.line(frame,pt1=direct['w_s'],pt2=direct['e_s'],color=(255,0,0),thickness=1)
        # cv2.line(frame,pt1=direct['w_n'],pt2=direct['w_s'],color=(255,0,0),thickness=1)
        # cv2.line(frame,pt1=direct['e_n'],pt2=direct['e_s'],color=(255,0,0),thickness=1)
        # gui.moveTo(center_x/2,center_y/2)
        # subdata = []
        
    cv2.imshow("Center of the pupil", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
cam.release
    
    
    
    

