import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    return edges

def find_text_boxes(image):
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area or other criteria
    text_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter contours based on size (adjust threshold as needed)
        if cv2.contourArea(contour) > 100:
            text_boxes.append([(x, y), (x + w, y + h)])
    
    # Sort text boxes based on their y-coordinate (top-left corner)
    text_boxes.sort(key=lambda box: box[0][1])
    
    return text_boxes

def calculate_distances(text_boxes):
    # Calculate vertical distances between text boxes (in pixels)
    num_boxes = len(text_boxes)
    distances = []
    
    for i in range(1, num_boxes):
        # Calculate vertical distance between the bottom of the previous box and top of current box
        prev_bottom = text_boxes[i-1][1][1]  # y-coordinate of bottom of previous box
        current_top = text_boxes[i][0][1]    # y-coordinate of top of current box
        distance = current_top - prev_bottom
        distances.append(distance)
    
    return distances

def plot_distances(distances):
    # Plot distances as a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(distances)), distances, color='skyblue')
    plt.title('Vertical Distances Between Text Lines')
    plt.xlabel('Text Line Index')
    plt.ylabel('Distance (pixels)')
    plt.savefig('distances.png')
    plt.show()

if __name__ == "__main__":
    # Path to input image containing text
    input_image_path = 'VVImages/BoxImages/9.png'
    
    # Preprocess the image (e.g., edge detection)
    edges = preprocess_image(input_image_path)
    
    # Find potential text bounding boxes
    text_boxes = find_text_boxes(edges)
    
    # Calculate vertical distances between text lines
    distances = calculate_distances(text_boxes)
    
    # Plot and save distances as a bar chart
    pos_nos = [num for num in distances if num >= 0]
 
    print("Line Spacing: ", *pos_nos)