import math

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

    print("Width:", width)
    print("Height:", height)

# Coordinates of the four corners of the rectangle
tl = (10, 20)  # Top-left corner
tr = (50, 20)  # Top-right corner
br = (50, 80)  # Bottom-right corner
bl = (10, 80)  # Bottom-left corner

find_h_w(tl, tr, br, bl)