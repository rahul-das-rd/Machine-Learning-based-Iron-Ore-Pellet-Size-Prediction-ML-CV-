import cv2 as cv
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import csv
import matplotlib.pyplot as plt

# function to rescale image
def rescaleframe(frame, scale=1):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# function to detect the circles
def detect_circles(image, min_radius, max_radius, threshold):
    # Perform Hough Circle Transform
    circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, 1, 8, param1=4, param2=threshold,minRadius=min_radius, maxRadius=max_radius)
    # If circles are detected, extract and return the circles
    if circles is not None:
        num_circles = len(circles[0])
        print("Number of circles detected:", num_circles)
        circles = np.round(circles[0, :]).astype(int)
        return circles
    else:
        return []

# Create a Tkinter root window
root = Tk()
root.withdraw()

# Prompt the user to select an image file
print("Select an image file:")
image_path = askopenfilename()

# Prompt the user to specify the CSV file
print("Specify the CSV file:")
csv_file_path = asksaveasfilename(initialdir="/", title="Select CSV File", filetypes=[("CSV Files", "*.csv")])

img = cv.imread(image_path)

# rescale the image
imgrz = rescaleframe(img)

# converting rescaled image to grayscale
gray = cv.cvtColor(imgrz, cv.COLOR_BGR2GRAY)
cv.imshow('B&W', gray)

# blur the gray image
blur = cv.GaussianBlur(gray, (11, 11), 20)

# adaptive thresholding
ad_th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 47, 4)

# canny edge detector
canny = cv.Canny(ad_th, 0, 50)

# thiker & visible.
dilated = cv.dilate(canny, (1, 1), iterations=0)
cv.imshow('thiker & visible.', dilated)

# Set the parameters for circle detection
min_radius = 8
max_radius = 15
threshold = 6

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

# Export circles data to CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y', 'diameter'])
    for (x, y, r) in detected_circles:
        writer.writerow([x, y, r * 2])

print(f"CSV file saved to: {csv_file_path}")

# Calculate histogram data
all_radii = [r * 2 for (_, _, r) in detected_circles]
hist, bins = np.histogram(all_radii, bins=10, range=(min_radius*2, max_radius*2))

# Plot the histogram
plt.bar(bins[:-1], hist, width=(max_radius*2 - min_radius*2) / 10, align='edge')
plt.xlabel('Diameter')
plt.ylabel('Frequency')
plt.title('Circle Diameter Distribution')
plt.show()

# Display the image with detected circles
cv.imshow("Detected Circles", imgrz_copy)

cv.waitKey(0)