import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python import python
from mediapipe.python.solution_base import BaseOptions
from mediapipe.python.solutions import gesture_recognizer as vision

# Load the model.
base_options = BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer(options)

# Initialize the video capture object.
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video capture object.
    ret, frame = cap.read()

    # Convert the frame to RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create an image object from the RGB frame.
    image = mp.solutions.drawing_utils._image_pyramid(image_frame=frame_rgb)

    # Recognize gestures in the input image.
    recognition_result = recognizer.process(image)

    # Process the result. In this case, visualize it.
    annotated_image = frame_rgb.copy()

    if recognition_result.hand_landmarks is not None:
        for hand_landmarks in recognition_result.hand_landmarks:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            mp.solutions.drawing_utils._draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks_proto,
                connections=mp.solutions.hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style())

    if recognition_result.gestures:
        top_gesture = recognition_result.gestures[0]
        cv2.putText(annotated_image, f'{top_gesture.classification} ({top_gesture.score:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the annotated image.
    cv2.imshow('Gesture Recognition', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    # Press 'q' to quit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object.
cap.release()

# Destroy all windows.
cv2.destroyAllWindows()
