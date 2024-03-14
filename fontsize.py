import cv2
import pytesseract

# Path to the image file
image_path = 'image.jpg'

# Read the image using OpenCV
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use pytesseract to do OCR on the image
text = pytesseract.image_to_string(gray)

# Split the text into lines
lines = text.split('\n')

# Calculate the bounding box of the detected text
bounding_boxes = pytesseract.image_to_boxes(gray)

# Extract font size information from bounding boxes
font_sizes = []
for box in bounding_boxes.splitlines():
    box = box.split()
    font_size = abs(int(box[3]) - int(box[1]))  # Height of the bounding box
    font_sizes.append(font_size)

# Calculate the average font size
if font_sizes:
    average_font_size = sum(font_sizes) / len(font_sizes)
    print("Average font size:", average_font_size)
else:
    print("No text detected.")