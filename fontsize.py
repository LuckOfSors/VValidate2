import cv2
import pytesseract
import os

#three parts of the standard deviation
def mean(numbers):
    return sum(numbers) / len(numbers)

def variance(numbers):
    mean_val = mean(numbers)
    return sum((x - mean_val) ** 2 for x in numbers) / len(numbers)

def standard_deviation(numbers):
    return variance(numbers) ** 0.5


def calculate_std_font_size(image_paths):
    total_font_sizes = []
    
    for image_path in image_paths:
        # Read the image using OpenCV
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use pytesseract to do OCR on the image
        text_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

        # Extract font size information from text lines
        font_sizes = []

        for i, text in enumerate(text_data['text']):
            # Skip empty text
            if text.strip():
                # Extract bounding box coordinates
                x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]

                # Estimate font size based on bounding box height
                font_size = h
                font_sizes.append(font_size)

        # Calculate the standard deviation font size for this image
        
        print(font_sizes)
        print(total_font_sizes)
        
        if font_sizes:
            mn = mean(font_sizes)
            total_font_sizes.append(mn)
        else:
            print(f"No text detected in {image_path}.")
        


    # Calculate the overall average font size across all images
    if total_font_sizes:
        overall_std_font_size = standard_deviation(total_font_sizes)
        return overall_std_font_size
    else:
        print("No text detected in any image.")
        return None

# List of image paths
image_paths = ['/Users/kaliyahsolomon/VValidate2/VVImages/BoxImages/3.png', '/Users/kaliyahsolomon/VValidate2/VVImages/BoxImages/6.png', '/Users/kaliyahsolomon/VValidate2/VVImages/BoxImages/9.png']

# Calculate average font size for all images
std_font_size = calculate_std_font_size(image_paths)

print(std_font_size)

'''
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
#three parts of the standard deviation
def mean(numbers):
    return sum(numbers) / len(numbers)

def variance(numbers):
    mean_val = mean(numbers)
    return sum((x - mean_val) ** 2 for x in numbers) / len(numbers)

def standard_deviation(numbers):
    return variance(numbers) ** 0.5

# Calculate the standard deviation
std_dev = standard_deviation(font_sizes)
print("Standard deviation:", std_dev)

# Calculate the average font size
if font_sizes:
    average_font_size = sum(font_sizes) / len(font_sizes)
    print("Average font size:", average_font_size)
else:
    print("No text detected.")
'''
