import math

# Define boundary points (NESW) and center point of gaze
boundary_points = {'N': (0, 100), 'E': (100, 50), 'S': (0, 0), 'W': (-100, 50)}
center_gaze = (90, 90)  # Example: center point of gaze (x, y) near NE boundary

# Function to calculate distance between two points
def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to check if center gaze is near boundary points
def check_near_boundary(center_gaze, boundary_points, threshold_distance=20):
    for direction, boundary_point in boundary_points.items():
        dist = distance(center_gaze, boundary_point)
        if dist <= threshold_distance:
            print(f"Center of gaze is near boundary point {direction}.")
        else:
            print(f"Center of gaze is not near boundary point {direction}.")

# Call the function to check if center gaze is near boundary points
check_near_boundary(center_gaze, boundary_points)
