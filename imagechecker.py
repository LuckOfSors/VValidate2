import cv2
import numpy as np

# Load the given image
given_image = cv2.imread("output0.jpg")
if given_image is None:
    print("Error: Could not load image.")
else:
    pass

given_gray = cv2.cvtColor(given_image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(given_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
ocr_engine = cv2.text.OCRTesseract_create()
text, _ = ocr_engine.run(thresh, 0)  

text_length = len(text)
whitespace_count = text.count(' ')
expected_text_length = 100  # text length
expected_whitespace_count = 10  # whitespace count

if text_length != expected_text_length:
    print('Text length differs: Expected - {}, Detected - {}'.format(expected_text_length, text_length))

if whitespace_count != expected_whitespace_count:
    print('Whitespace count differs: Expected - {}, Detected - {}')