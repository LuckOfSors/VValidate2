import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
image = cv2.imread(r'.\VVImages\BoxImages\9.png')

# Upscale the image
scale_factor = 2
upscaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

# Convert the image to grayscale
grayscale = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)

# Thresholding
_, thresholded = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)

# Perform OCR to detect text
custom_config = r'--oem 3 --psm 6'  # Tesseract OCR configuration
text_data = pytesseract.image_to_data(thresholded, config=custom_config, output_type=pytesseract.Output.DICT)

# Extract bounding boxes for words
boxes = []
lengths = []
for i in range(len(text_data['text'])):
    if int(text_data['conf'][i]) > 0:
        x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
        boxes.append((x, y, w, h))

# Draw lines between boxes and display length as text
valid_distances = []
for i in range(len(boxes) - 1):
    x1, y1, w1, h1 = boxes[i]
    x2, y2, w2, h2 = boxes[i + 1]
    # Calculate bottom-right corner of first box
    bottom_right_x1, bottom_right_y1 = x1 + w1, y1 + h1
    # Calculate bottom-left corner of second box
    bottom_left_x2, bottom_left_y2 = x2, y2 + h2
    # Calculate distance between bottom-right corner of first box and bottom-left corner of second box
    distance = np.sqrt((bottom_left_x2 - bottom_right_x1) ** 2 + (bottom_left_y2 - bottom_right_y1) ** 2)
    if distance <= 100:  # Check if distance is less than or equal to 100
        # Draw line between bottom-right corner of first box and bottom-left corner of second box
        cv2.line(upscaled_image, (bottom_right_x1, bottom_right_y1), (bottom_left_x2, bottom_left_y2), (255, 0, 0), 1)
        # Display distance between boxes as text under each line
        cv2.putText(upscaled_image, f"{distance:.2f}", (int((bottom_right_x1 + bottom_left_x2) / 2), int((bottom_right_y1 + bottom_left_y2) / 2) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        lengths.append(distance)
        valid_distances.append(distance)

# Calculate average length
average_length = np.mean(valid_distances)

# Display the average length as text on the image
cv2.putText(upscaled_image, f"Average Length: {average_length:.2f}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Define a threshold for significant difference
threshold_difference = average_length * .2  # You can adjust this threshold as needed

# Check each word spacing against the average length
for distance in valid_distances:
    if abs(distance - average_length) > threshold_difference:
        print("Significant difference in word spacing detected: Likely fake")
        break

# Display the result
cv2.imwrite("word_spacing.png", upscaled_image)
