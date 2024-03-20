import cv2
import pytesseract
import os
from PIL import Image
import re

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
        if font_sizes:
            mn = mean(font_sizes)
            total_font_sizes.append(mn)
        else:
            print(f"No text detected in {image_path}.")
    print(total_font_sizes)
        


    # Calculate the overall average font size across all images
    if total_font_sizes:
        overall_std_font_size = standard_deviation(total_font_sizes)
        return overall_std_font_size
    else:
        print("No text detected in any image.")
        return None

def std2_font_size(image_paths):
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

        # Calculate the standard deviation
        std_dev = standard_deviation(font_sizes)
        return std_dev

# List of image paths

def textChecker(directory, text_list):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            image_path = os.path.join(root, filename)
            if imageContainsText(image_path, text_list):
                image_paths.append(image_path)
    #return image_paths
    length = len(image_paths)
    if length > 1:
        std_font_size = calculate_std_font_size(image_paths)
        print("Standard deviation of more than one image", std_font_size)
    else:
        std_font_size = std2_font_size(image_paths)
        print("Standard deviation of one image", std_font_size)


def imageContainsText(image_path, text_list):
    # Check if an image file contains one of the specified strings of text.
    extracted_text = pytesseract.image_to_string(Image.open(image_path))
    for text in text_list:
        if text in extracted_text:
            return True
    return False
#Input image path and out folder

textList = ["Receipt Number", "Receipt", "Beneficiary", "Petitioner","Page", "Notice Date", "Case Type"]
textChecker ("./VVImages/BoxImages", textList)
textList2 = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",  "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR","PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
textChecker ("./VVImages/BoxImages", textList2)
