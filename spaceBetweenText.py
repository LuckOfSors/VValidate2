import cv2
import pytesseract


def find_line_spacing(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use Tesseract to perform OCR and get bounding boxes
    custom_config = r'--oem 3 --psm 6'  # Page segmentation mode 6 (Assume a single uniform block of text)
    text_boxes = pytesseract.image_to_boxes(gray, config=custom_config).split('\n')
    
    # Extract Y-coordinates of each bounding box
    y_coordinates = []
    for box in text_boxes:
        try:
            y_coordinates.append(int(box.split()[2]))
        except IndexError:
            # Skip boxes with missing or insufficient information
            pass
    
    # If no text boxes are found, return None
    if not y_coordinates:
        return None
    
    # Calculate line spacing
    line_spacing = sum(y_coordinates[i] - y_coordinates[i-1] for i in range(1, len(y_coordinates))) / (len(y_coordinates) - 1)
    
    return line_spacing

# Example usage
image_path = "./VVImages/BoxImages/9.png"  # Replace with your image file path
line_spacing = find_line_spacing(image_path)
print("Line spacing:", line_spacing)
