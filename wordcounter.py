import cv2
import pytesseract

# Function to find the word after specific words from a list in the extracted text
def find_word_after_specific_words(extracted_text, specific_words):
    words = extracted_text.split()
    word_after_specific_words = []
    
    for i, word in enumerate(words):
        if word in specific_words and i < len(words):
            next_word = words[i + 1]
            word_after_specific_words.append(next_word)
    
    return word_after_specific_words

# Path to the image file
image_path = 'VVImages/rotated0.jpg'

# List of specific words to search for
specific_words_to_search = ['Number', 'NAME', 'Petitioner']  # Example list

# Read the image using OpenCV
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use pytesseract to extract text from the image
extracted_text = pytesseract.image_to_string(gray)

# Find the word after specific words from the list in the extracted text
word_after_specific_words = find_word_after_specific_words(extracted_text, specific_words_to_search)
print(word_after_specific_words)


def count_word_occurrences(extracted_text, word_list):
    word_counts = {word: 0 for word in word_list}  # Initialize counts to 0 for each word in the list
    
    # Split the extracted text into words
    words = extracted_text.split()
    
    # Count the occurrences of each word from the list
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
    
    return list(word_counts.values())

word_occurrences = count_word_occurrences(extracted_text, word_after_specific_words)
print(word_occurrences)

def compare_lists(list1, list2):
    # Ensure both lists have the same length
    
    # Create a new list to store comparison results
    comparison_result = []
    
    # Iterate through corresponding elements in the lists and compare them
    for num1, num2 in zip(list1, list2):
        comparison_result.append(num1 == num2)
    
    return comparison_result

# Example lists
list2 = [4, 3, 3]  

# Compare the lists
result = compare_lists(word_occurrences, list2)

# Print the result
print("Comparison result:", result)
