import cv2
import numpy as np


# Assuming center coordinates are available as center_x and center_y
center_x = 100
center_y = 100

# Assuming frame dimensions are available as frame_w and frame_h
frame_w = 640
frame_h = 480

# Define landmarks dictionary with landmark coordinates
landmarks = {
    382: np.array([50, 80]),  # Example landmark coordinates (replace with actual values)
    384: np.array([60, 90]),
    385: np.array([70, 100]),
    386: np.array([80, 110]),
    387: np.array([90, 120]),
    388: np.array([100, 130]),
    390: np.array([110, 140]),
    374: np.array([120, 150])
}

for id, landmark_id in enumerate([382, 384, 385, 386, 387, 388, 390, 374]):
    landmark = landmarks[landmark_id]
    x = int(landmark[0] * frame_w)
    y = int(landmark[1] * frame_h)
    coord1 = np.array([center_x, center_y])
    coord2 = np.array([x, y])
    print(f"difference of landmark[{landmark_id}] - center: {coord2 - coord1}")
