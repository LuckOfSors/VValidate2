import cv2

filename = 'VVImages/BoxImages/9.png'

# Load the image
image = cv2.imread(filename)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to binarize the image
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by y-coordinate
contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[1])

# Calculate spacing between lines
line_spacing = []
for i in range(1, len(contours)):
    current_contour = cv2.boundingRect(contours[i])
    previous_contour = cv2.boundingRect(contours[i - 1])
    spacing = current_contour[1] - (previous_contour[1] + previous_contour[3])
    line_spacing.append(spacing)

# Print the individual line spacing
print("Individual line spacings:")
for spacing in line_spacing:
    print(spacing)

# Visualize the contours (optional)
# image_with_contours = image.copy()
# cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
# cv2.imshow("Contours", image_with_contours)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
