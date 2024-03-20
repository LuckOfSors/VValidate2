import cv2

# Load image
image = cv2.imread('./VVImages/BoxImages/9.png')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area
min_area = 50
max_area = 10000
filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

total_length = 0
num_characters = 0
character_lengths = []

# Iterate through contours
for i, contour in enumerate(filtered_contours):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calculate the average length (width or height) of each character
    total_length += w + h  # You can choose either width or height, or their average
    num_characters += 1
    
    # Extract individual character
    character = gray[y:y+h, x:x+w]
    
    # Save character image
    cv2.imwrite('./SeperatedChar/' + f'character_{i}.png', character)
    
    # Store character length
    character_lengths.append(w + h)

# Calculate the average length of characters
average_length = total_length / num_characters
print(f"Average length of each character: {average_length}")

# Define a threshold for significant difference
threshold = 0.1 * average_length  # Adjust the factor as needed

# Search for characters with lengths significantly different from the average
for i, length in enumerate(character_lengths):
    if abs(length - average_length) > threshold:
        print(f"Character {i+1} has a significantly different length ({length}) compared to the average length ({average_length})")
print("Done")
# Display segmented characters
cv2.imshow('Segmented Characters', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
