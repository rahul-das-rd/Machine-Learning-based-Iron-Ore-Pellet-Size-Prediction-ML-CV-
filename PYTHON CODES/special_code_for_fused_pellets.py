import cv2 as cv
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# function to rescale image
def rescaleframe(frame, scale=1):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# function to detect the circles
def detect_circles(image, min_radius, max_radius, threshold):
    # Perform Hough Circle Transform
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 1, 25, param1=50 , param2=threshold, minRadius=min_radius, maxRadius=max_radius) #1,12,50
    # If circles are detected, extract and return the circles
    if circles is not None:
        num_circles = len(circles[0])
        print("Number of circles detected:", num_circles)
        circles = np.round(circles[0, :]).astype(int)
        return circles
    else:
        return []

# Set the parameters for circle detection
threshold = 6  #param1 = 4,param2 = 6
min_radius = 8 #8
max_radius = 15 #15


# Create a Tkinter root window
root = Tk()
root.withdraw()

# Prompt the user to select an image file
print("Select an image file:")
image_path = askopenfilename()

img = cv.imread(image_path)
cv.imshow('actual', img)

# rescale the image
imgrz = rescaleframe(img)

# converting rescaled image to grayscale
gray = cv.cvtColor(imgrz, cv.COLOR_BGR2GRAY)

# blur the gray image
blur = cv.GaussianBlur(gray, (11, 11), sigmaX = 8, sigmaY= 8) #(11,11),20,20
cv.imshow('blur',blur)

# adaptive thresholding
ad_th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 55, 4) #255, 47, 4
cv.imshow('Threshold',ad_th)

# canny edge detector
canny = cv.Canny(ad_th, 680, 1500) #0,50
cv.imshow('canny edges detected', canny)

# thiker & visible.
dilated = cv.dilate(canny, (1, 1), iterations=0)
cv.imshow('dilated thiker & visible edges', dilated)

# Detect circles using CHT[Hough Circle Detection]
detected_circles = detect_circles(dilated, min_radius, max_radius, threshold)

# Define the size categories
small_circles = []
medium_circles = []
large_circles = []

# Categorize the detected circles based on their radii
for (x, y, r) in detected_circles:
    if r < 10:
        small_circles.append((x, y, r))
    elif r >= 10 and r < 13:
        medium_circles.append((x, y, r))
    else:
        large_circles.append((x, y, r))

# Draw detected circles on the copy of the original image, with different colors for each category
imgrz_copy = np.copy(imgrz)
for (x, y, r) in small_circles:
    cv.circle(imgrz_copy, (x, y), r, (0, 255, 0), 2)
for (x, y, r) in medium_circles:
    cv.circle(imgrz_copy, (x, y), r, (0, 0, 255), 2)
for (x, y, r) in large_circles:
    cv.circle(imgrz_copy, (x, y), r, (255, 0, 0), 2)

# Display the number of circles in each category
small_circle_text = f"Small Circles: {len(small_circles)}"
medium_circle_text = f"Medium Circles: {len(medium_circles)}"
large_circle_text = f"Large Circles: {len(large_circles)}"
total_circle_text = f"Total Circles: {len(detected_circles)}"
cv.putText(imgrz_copy, small_circle_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv.putText(imgrz_copy, medium_circle_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv.putText(imgrz_copy, large_circle_text, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
cv.putText(imgrz_copy, total_circle_text, (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Display the image with detected circles
cv.imshow("Detected Circles", imgrz_copy)

cv.waitKey(0)

# Perform Hough Circle Transform
# dp: Inverse ratio of the accumulator resolution to the image resolution.
#     A smaller value of dp means a higher resolution of the accumulator.
#     Typically, dp = 1 for the same resolution, or dp = 2 for half the resolution.
# minDist: Minimum distance between the centers of detected circles.
#           If the distance is too small, multiple circles may be detected for a single circle.
# param1: Upper threshold for the internal Canny edge detection.
#         This threshold is used in the edge detection stage of the Hough Circle Transform.
#         It determines the sensitivity of the edge detection.
# param2: Accumulator threshold for circle detection.
#         It determines the minimum number of votes required for a circle to be detected.
#         Increasing this threshold can help reduce false positive detections.
# minRadius: Minimum radius of the circles to be detected.
# maxRadius: Maximum radius of the circles to be detected.
# circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, dp=1, minDist=15, param1=50, param2=8,minRadius=8, maxRadius=15)
