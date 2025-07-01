# Object Dimension Measurement using Computer Vision
# This script measures the dimensions of objects in an image using a reference object

# Import required libraries
from scipy.spatial import distance as dist  # For calculating Euclidean distances
from imutils import perspective               # For perspective correction and point ordering  # type: ignore
from imutils import contours                 # For contour sorting utilities  # type: ignore
import numpy as np                           # For numerical operations and array handling
import imutils                               # Additional computer vision utilities  # type: ignore
import cv2                                   # OpenCV for image processing


def midpoint(ptA, ptB):
	"""
	Calculate the midpoint between two points
	Args:
		ptA: First point (x, y)
		ptB: Second point (x, y)
	Returns:
		Tuple containing the midpoint coordinates
	"""
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Configuration parameters
image_path = 'imgs/test1.png'      # Path to the input image
ref_obj_width = 25            # Width of the reference object in millimeters (used for scale calibration)

# Load and preprocess the image
image = cv2.imread(image_path)

# Get original image dimensions
height = image.shape[0]
width = image.shape[1]

# Resize image to 1/5 of original size for faster processing
image = cv2.resize(image, (int(width/5),int(height/5)))

print(image.shape)  # Display the new image dimensions

# Convert image to grayscale for edge detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply Gaussian blur to reduce noise
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# Edge detection using Canny algorithm
edged = cv2.Canny(gray, 50, 100)
# Morphological operations to clean up the edges
# Dilation: expand white regions (edges)
edged = cv2.dilate(edged, None, iterations=1)  # type: ignore
# Erosion: shrink white regions back to original size, removing noise
edged = cv2.erode(edged, None, iterations=1)  # type: ignore

# Find contours in the edge-detected image
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
# Extract contours from the result (handles different OpenCV versions)
cnts = imutils.grab_contours(cnts)

# Sort contours from left-to-right for consistent processing
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None  # Variable to store the pixel-to-metric ratio

# Process each contour found in the image
for c in cnts:
	# Filter out small contours (noise) - skip contours smaller than 100 pixels
	if cv2.contourArea(c) < 100:
		continue

	# Create a copy of the original image for drawing measurements
	orig = image.copy()
	
	# Find the minimum area rectangle that encloses the contour
	box = cv2.minAreaRect(c)
	# Get the corner points of the bounding box
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)  # type: ignore
	box = np.array(box, dtype="int")

	# Order the points in a consistent manner (top-left, top-right, bottom-right, bottom-left)
	box = perspective.order_points(box)
	# Draw the bounding box on the image
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

	# Draw circles at each corner of the bounding box
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

	# Unpack the ordered bounding box points
	(tl, tr, br, bl) = box
	
	# Calculate midpoints of the top and bottom edges
	(tltrX, tltrY) = midpoint(tl, tr)    # Top edge midpoint
	(blbrX, blbrY) = midpoint(bl, br)    # Bottom edge midpoint

	# Calculate midpoints of the left and right edges
	(tlblX, tlblY) = midpoint(tl, bl)    # Left edge midpoint
	(trbrX, trbrY) = midpoint(tr, br)    # Right edge midpoint

	# Draw circles at the midpoints for visualization
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

	# Draw lines connecting the midpoints to show the measurement lines
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)  # Vertical measurement line
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)  # Horizontal measurement line

	# Calculate the Euclidean distances between midpoints (in pixels)
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))  # Height in pixels
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))  # Width in pixels

	# Initialize the pixels-per-metric ratio using the first object as reference
	if pixelsPerMetric is None:
		pixelsPerMetric = dB / ref_obj_width

	# Convert pixel measurements to real-world measurements (millimeters)
	dimA = dA / pixelsPerMetric  # Height in millimeters
	dimB = dB / pixelsPerMetric  # Width in millimeters

	# Draw the measurement text on the image
	cv2.putText(orig, "{:.1f}mm".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)  # Height measurement
	cv2.putText(orig, "{:.1f}mm".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)  # Width measurement

	# Display the result image with measurements
	cv2.imshow("Image", orig)
	cv2.waitKey(0)  # Wait for key press to continue to next object