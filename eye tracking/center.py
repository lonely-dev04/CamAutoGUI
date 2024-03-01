import numpy as np

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

# Example points on the circumference of a circle
points = [(0.4643688499927521 , 0.5692043304443359), (0.4520490765571594 , 0.554023027420044), (0.43896928429603577 , 0.5673460364341736), (0.45125263929367065 , 0.5827410817146301)]

# Find the center of the circle
center = find_circle_center(points)
print("Center of the circle:", center)

# landmarks[474]0.4643688499927521 , 0.5692043304443359
# landmarks[475]0.4520490765571594 , 0.554023027420044
# landmarks[476]0.43896928429603577 , 0.5673460364341736
# landmarks[477]0.45125263929367065 , 0.5827410817146301