import pyautogui as gui
import imutils
import cv2

greenLower = (0,88,139)
greenUpper = (76,255,209)

camera = cv2.VideoCapture(0)

while True:
    _,frame = camera.read()

    frame = imutils.resize(frame,width=600)
    blurred = cv2.GaussianBlur(frame,(11,11),0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x,y),radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c) #To find center 
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 10:
            cv2.circle(frame, (int(x),int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center , 5 , (0,0,255), -1)
            if radius > 250:
                print("Stop :",int(center[0]),int(center[1]))
            else:
                if(center[0]<150):
                    print("left:",int(center[0]),int(center[1]))
                    gui.moveTo(center[0],center[1])
                elif(center[0]>450):
                    print("Right :",int(center[0]),int(center[1]))
                    gui.moveTo(center[0],center[1])
                elif(center[1]>300):
                    print("lower :",int(center[0]),int(center[1]))
                    gui.moveTo(center[0],center[1])
                elif(center[1]<200):
                    print("Upper :",int(center[0]),int(center[1]))
                    gui.moveTo(center[0],center[1])
                elif(radius<250):
                    print("Front :",int(center[0]),int(center[1]))
                    gui.moveTo(center[0],center[1])
                else:
                    print("Stop :",int(center[0]),int(center[1]))
                    gui.moveTo(center[0],center[1])
                    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
camera.release
cv2.destroyAllWindows()