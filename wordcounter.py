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

def word_to_phrase(list):
    lines = extracted_text.split('\n')

    # Find the line starting with "TECH"
    desired_line = None
    for value in word_after_specific_words:
        for line in lines:
            if line.strip().startswith(value):
                desired_line = line.strip()
                break

    modified_string = desired_line.replace(',', '')

    return modified_string

def count_word_occurrences(extracted_text, word_list):
    word_counts = {word: 0 for word in word_list}  # Initialize counts to 0 for each word in the list
    
    # Split the extracted text into words
    words = extracted_text.split()
    
    # Count the occurrences of each word from the list
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
    
    return list(word_counts.values())

def count_phrase_occurrences(text, phrase):
    return text.count(phrase)


def compare_lists(list1, list2):
    # Ensure both lists have the same length
    
    # Create a new list to store comparison results
    comparison_result = []
    
    # Iterate through corresponding elements in the lists and compare them
    for num1, num2 in zip(list1, list2):
        comparison_result.append(num1 == num2)
    
    return comparison_result

# Path to the image file
image_path = 'VVImages/rotated0.jpg'

# List of specific words to search for
specific_words_to_search = ['NAME', 'Petitioner']  # Example list

# Read the image using OpenCV
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use pytesseract to extract text from the image
extracted_text = pytesseract.image_to_string(gray)

# Find the word after specific words from the list in the extracted text
word_after_specific_words = find_word_after_specific_words(extracted_text, specific_words_to_search)
print(word_after_specific_words)

#turn words to Phrases
phrases = word_to_phrase(word_after_specific_words)
print(phrases)

#Make it one List
word_after_specific_words[0] = phrases
print(word_after_specific_words)

# Count word occurences
word_occurrences = count_word_occurrences(extracted_text, word_after_specific_words)
print(word_occurrences)

# Count phrase occurences
phrase_occurrences = count_phrase_occurrences(extracted_text, phrases)
print(phrase_occurrences)

#Make it one list
word_occurrences[0] = phrase_occurrences
print(word_occurrences)

# Base lists
list2 = [4, 3]  

# Compare the lists
result = compare_lists(word_occurrences, list2)

# Print the result
print("Comparison result:", result)
   

