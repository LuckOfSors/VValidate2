import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
image = cv2.imread(r"C:\Users\kaide\OneDrive\Documents\Python\Test\VVImages\BoxImages\9.png")

# Upscale the image
scale_factor = 2

upscaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

# Convert the image to grayscale
grayscale = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)

thresholded = cv2.inRange(blurred, 0, 180)
thresholded = cv2.bitwise_not(thresholded)
cv2.imwrite("threshheld.png", thresholded)

final = thresholded

# Perform OCR to detect text
custom_config = r'--oem 3 --psm 6'  # Tesseract OCR configuration
text = pytesseract.image_to_string(final, config=custom_config)

# Draw boxes around individual letters based on the detected text
for char_index, char in enumerate(text):
    boxes = pytesseract.image_to_boxes(final)
    #print(boxes)
    #input()
    for box in boxes.splitlines():
        b = box.split()
        if len(b) == 6 and b[0] == char:
            # Adjust coordinates for upscaled image
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            cv2.rectangle(upscaled_image, (x, upscaled_image.shape[0] - y), (w, upscaled_image.shape[0] - h), (0, 255, 0), 1)

# Display the result
cv2.imshow("Text with Boxes", upscaled_image)
cv2.imwrite("output_image_with_boxes.png", upscaled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()