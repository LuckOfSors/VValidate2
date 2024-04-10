import cv2
import pytesseract
import numpy as np

def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def detect_text(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    # Use pytesseract to extract text and bounding boxes
    boxes = pytesseract.image_to_data(preprocessed_image, output_type=pytesseract.Output.DICT)
    return boxes

def analyze_text_regions(boxes):
    # Extract relevant information from text boxes
    texts = []
    lefts = []
    tops = []
    widths = []
    heights = []
    confidences = []

    for i, text in enumerate(boxes['text']):
        if int(boxes['conf'][i]) > 0:
            texts.append(text)
            lefts.append(boxes['left'][i])
            tops.append(boxes['top'][i])
            widths.append(boxes['width'][i])
            heights.append(boxes['height'][i])
            confidences.append(boxes['conf'][i])

    # Analyze text regions for consistency
    num_regions = 5  # Number of regions to divide the image into
    region_height = len(texts) // num_regions  # Approximate height of each region

    # Initialize consistency flags for each region
    region_consistencies = [False] * num_regions

    # Check consistency in each region
    for i in range(num_regions):
        start_index = i * region_height
        end_index = (i + 1) * region_height

        region_texts = texts[start_index:end_index]
        region_lefts = lefts[start_index:end_index]
        region_tops = tops[start_index:end_index]
        region_widths = widths[start_index:end_index]
        region_heights = heights[start_index:end_index]

        # Check consistency of left margins within the region
        left_margin_deviation = np.std(region_lefts)
        
        # Check consistency of top margins within the region
        top_margin_deviation = np.std(region_tops)
        
        # Check consistency of font size (heights) within the region
        font_height_deviation = np.std(region_heights)

        # Overall consistency check for the region
        region_consistencies[i] = (left_margin_deviation < 5) and (top_margin_deviation < 5) and (font_height_deviation < 5)
    
    # Overall consistency check based on region consistencies
    overall_consistency = all(region_consistencies)

    return overall_consistency

def main():
    image_path = "output0.png" 
    boxes = detect_text(image_path)
    is_consistent = analyze_text_regions(boxes)
    if is_consistent:
        print("Text in the image is consistent.")
    else:
        print("Text in the image is not consistent.")

if __name__ == "_main_":
    main()