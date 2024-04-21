import numpy as np
import cv2
import pdf2image
import os
import pytesseract
import re
from skimage.transform import radon
from PIL import Image
from numpy import asarray, mean, array, blackman
from numpy.fft import rfft
import math
try:
    from parabolic import parabolic

    def argmax(x):
        return parabolic(x, np.argmax(x))[0]
except ImportError:
    from numpy import argmax


class PDFChecker:
    def __init__(self, pdfPath):
        self.valid = True
        self.pdfPath = pdfPath
        self.images = pdf2image.convert_from_path(self.pdfPath)
        self.topImages = []
        self.bottomImages = []
        self.rotatePdf(self.images)
        self.box_extraction("VVImages/rotated0.jpg", "./VVImages/BoxImages/")
        self.textList = ["Receipt Number", "Receipt", "Beneficiary", "Petitioner","Page", "Notice Date" "Case Type", "Date of Birth"]
        self.LineSpacing()
        self.textChecker("./VVImages/BoxImages/", self.textList)
        self.wordCounter()
        self.wordSpacing()
        self.calculateSpacing("./VVImages/BoxImages/9.png")
        for root, dirs, files in os.walk("./VVImages/BoxImages/"):
            for filename in files:
                image_path = os.path.join(root, filename)
                self.angleOfText(image_path)
        if self.valid:
            print("All tests passed.")
        else:
            print("Some tests failed. Look in terminal logs for more information.")
        
                

    def wordSpacing(self):
        image = cv2.imread(r'.\VVImages\BoxImages\9.png')

        # Upscale the image
        scale_factor = 2
        upscaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # Convert the image to grayscale
        grayscale = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)

        # Thresholding
        _, thresholded = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)

        # Perform OCR to detect text
        custom_config = r'--oem 3 --psm 6'  # Tesseract OCR configuration
        text_data = pytesseract.image_to_data(thresholded, config=custom_config, output_type=pytesseract.Output.DICT)

        # Extract bounding boxes for words
        boxes = []
        lengths = []
        for i in range(len(text_data['text'])):
            if int(text_data['conf'][i]) > 0:
                x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
                boxes.append((x, y, w, h))

        # Draw lines between boxes and display length as text
        valid_distances = []
        for i in range(len(boxes) - 1):
            x1, y1, w1, h1 = boxes[i]
            x2, y2, w2, h2 = boxes[i + 1]
            # Calculate bottom-right corner of first box
            bottom_right_x1, bottom_right_y1 = x1 + w1, y1 + h1
            # Calculate bottom-left corner of second box
            bottom_left_x2, bottom_left_y2 = x2, y2 + h2
            # Calculate distance between bottom-right corner of first box and bottom-left corner of second box
            distance = np.sqrt((bottom_left_x2 - bottom_right_x1) ** 2 + (bottom_left_y2 - bottom_right_y1) ** 2)
            if distance <= 100:  # Check if distance is less than or equal to 100
                # Draw line between bottom-right corner of first box and bottom-left corner of second box
                cv2.line(upscaled_image, (bottom_right_x1, bottom_right_y1), (bottom_left_x2, bottom_left_y2), (255, 0, 0), 1)
                # Display distance between boxes as text under each line
                cv2.putText(upscaled_image, f"{distance:.2f}", (int((bottom_right_x1 + bottom_left_x2) / 2), int((bottom_right_y1 + bottom_left_y2) / 2) + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                lengths.append(distance)
                valid_distances.append(distance)

        # Calculate average length
        average_length = np.mean(valid_distances)

        # Display the average length as text on the image
        cv2.putText(upscaled_image, f"Average Length: {average_length:.2f}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Define a threshold for significant difference
        threshold_difference = average_length * .2  # You can adjust this threshold as needed

        # Check each word spacing against the average length
        for distance in valid_distances:
            if abs(distance - average_length) > threshold_difference:
                print("Significant difference in word spacing detected: Likely fake")
                self.valid = False
                break


# Display the result
    def rotatePdf(self, images):
        for x, img in enumerate(images):
            img.save('VVImages/output'+str(x)+'.jpg', 'JPEG')
            # load file, converting to grayscale
            img = cv2.imread('VVImages/output'+str(x)+'.jpg')
            I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = I.shape
            # resize image to reduce processing time
            if (w > 640):
                I = cv2.resize(I, (640, int((h / w) * 640)))
            I = I - np.mean(I)  # Demean; make the brightness extend above and below zero

            # Do the radon transform
            sinogram = radon(I)
            # Find the RMS value of each row and find "busiest" rotation,
            # where the transform is lined up perfectly with the alternating dark
            # text and white lines
            r = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.transpose()])
            rotation = np.argmax(r)
            print('Rotation: {:.2f} degrees'.format(90 - rotation))

            # Rotate and save with the original resolution
            M = cv2.getRotationMatrix2D((w/2,h/2),90 - rotation,1)
            dst = cv2.warpAffine(img,M,(w,h))
            cv2.imwrite('VVImages/rotated'+str(x)+'.jpg', dst)
    def sort_contours(self, cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0

        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True

        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1

        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))

        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)
    
    def wordCounter(self):
        image_path = 'VVImages/rotated0.jpg'

        # List of specific words to search for
        specific_words_to_search = ['NAME', 'Petitioner']  # Example list

        # Read the image using OpenCV
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use pytesseract to extract text from the image
        self.extracted_textt = pytesseract.image_to_string(gray)

        # Find the word after specific words from the list in the extracted text
        self.word_after_specific_words = self.find_word_after_specific_words(self.extracted_textt, specific_words_to_search)


        #turn words to Phrases
        phrases = self.word_to_phrase(self.word_after_specific_words)


        #Make it one List
        self.word_after_specific_words[0] = phrases


        # Count word occurences
        word_occurrences = self.count_word_occurrences(self.extracted_textt, self.word_after_specific_words)


        # Count phrase occurences
        phrase_occurrences = self.count_phrase_occurrences(self.extracted_textt, phrases)


        #Make it one list
        word_occurrences[0] = phrase_occurrences


        # Base lists
        list2 = [4, 3]  

        # Compare the lists
        result = self.compare_lists(word_occurrences, list2)
        if False in result:
            self.valid = False
            print("Word occurences are incorrect.")
        else:
            print("Word occurences are correct.")
        

    def find_word_after_specific_words(self,extracted_text, specific_words):
        words = extracted_text.split()
        self.word_after_specific_words = []
        
        for i, word in enumerate(words):
            if word in specific_words and i < len(words):
                next_word = words[i + 1]
                self.word_after_specific_words.append(next_word)
        
        return self.word_after_specific_words

    def word_to_phrase(self,list):
        lines = self.extracted_textt.split('\n')

        # Find the line starting with "TECH"
        desired_line = None
        for value in self.word_after_specific_words:
            for line in lines:
                if line.strip().startswith(value):
                    desired_line = line.strip()
                    break

        modified_string = desired_line.replace(',', '')

        return modified_string

    def count_word_occurrences(self,extracted_text, word_list):
        word_counts = {word: 0 for word in word_list}  # Initialize counts to 0 for each word in the list
        
        # Split the extracted text into words
        words = extracted_text.split()
        
        # Count the occurrences of each word from the list
        for word in words:
            if word in word_counts:
                word_counts[word] += 1
        
        return list(word_counts.values())

    def count_phrase_occurrences(self,text, phrase):
        return text.count(phrase)


    def compare_lists(self,list1, list2):
        # Ensure both lists have the same length
        
        # Create a new list to store comparison results
        comparison_result = []
        
        # Iterate through corresponding elements in the lists and compare them
        for num1, num2 in zip(list1, list2):
            comparison_result.append(num1 == num2)
        
        return comparison_result
    def preprocess_image(self, image_path):
        # Load the image
        img = cv2.imread(image_path)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian Blur
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def detect_text(self, image_path):
        # Preprocess the image
        preprocessed_image = self.preprocess_image(image_path)
        # Use pytesseract to extract text and bounding boxes
        boxes = pytesseract.image_to_data(preprocessed_image, output_type=pytesseract.Output.DICT)
        return boxes

    def analyze_text_regions(self, boxes):
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
        #Functon for extracting the box
    def box_extraction(self, img_for_box_extraction_path, cropped_dir_path):

 
        img = cv2.imread(img_for_box_extraction_path, 0)  # Read the image
        (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
        img_bin = 255-img_bin  # Invert the image

 
        cv2.imwrite("Images/Image_bin.jpg",img_bin)

 
        # Defining a kernel length
        kernel_length = np.array(img).shape[1]//40
        
        # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
        verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
        hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        # A kernel of (3 X 3) ones.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Morphological operation to detect verticle lines from an image
        img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
        verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
        cv2.imwrite("Images/verticle_lines.jpg",verticle_lines_img)

        # Morphological operation to detect horizontal lines from an image
        img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
        horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
        cv2.imwrite("Images/horizontal_lines.jpg",horizontal_lines_img)

        # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
        alpha = 0.5
        beta = 1.0 - alpha
        # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
        img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
        img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
        (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        cv2.imwrite("Images/img_final_bin.jpg",img_final_bin)
        # Find contours for image, which will detect all the boxes
        contours, hierarchy = cv2.findContours(
            img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Sort all the contours by top to bottom.
        (contours, boundingBoxes) = self.sort_contours(contours, method="top-to-bottom")



        idx = 0
        for c in contours:
            # Returns the location and width,height for every contour
            x, y, w, h = cv2.boundingRect(c)

            # If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
            if (w > 80 and h > 20) and w > 3*h:
                idx += 1
                new_img = img[y:y+h, x:x+w]
                cv2.imwrite(cropped_dir_path+str(idx) + '.png', new_img)
    def FontSize(self):
        image_path = "output0.png" 
        boxes = self.detect_text(image_path)
        is_consistent = self.analyze_text_regions(boxes)
        if is_consistent:
            print("Size in the image is consistent.")
        else:
            print("Size in the image is not consistent.")
            self.valid = False

    def textChecker(self, directory, text_list):
        """Delete image files in directory and its subdirectories that don't contain any of the specified strings of text."""
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith('.png'):  # Adjust file extensions as needed
                    image_path = os.path.join(root, filename)
                    if not self.imageContainsText(image_path, text_list):
                        os.remove(image_path)

    def imageContainsText(self, image_path, text_list):
        # Check if an image file contains one of the specified strings of text.
        extracted_text = pytesseract.image_to_string(Image.open(image_path))
        pattern = r'[A-Z]{2}\s\d{5}'
        for text in text_list:
            if text in extracted_text:
                if text != "Date of Birth":
                    self.topImages.append(image_path)
                return True
            if re.search(pattern, extracted_text):
                return True
        return False
    def rms_flat(self, a):
        return np.sqrt(np.mean(np.abs(a) ** 2))
    def calculateSpacing(self, filename):
        I = asarray(Image.open(filename))
        I = I - mean(I)  # Demean; make the brightness extend above and below zero

        # Do the radon transform and display the result
        sinogram = radon(I)
        # Find the RMS value of each row and find "busiest" rotation,
        # where the transform is lined up perfectly with the alternating dark
        # text and white lines
        r = array([self.rms_flat(line) for line in sinogram.transpose()])
        rotation = argmax(r)
        print('For file:', filename)

        # Plot the busy row
        row = sinogram[:, rotation]
        N = len(row)

        # Take spectrum of busy row and find line spacing
        window = blackman(N)
        spectrum = rfft(row * window)

        frequency = argmax(abs(spectrum))
        line_spacing = N / frequency  # pixels

    def mean(self, numbers):
        return sum(numbers) / len(numbers)

    def variance(self, numbers):
        mean_val = mean(numbers)
        return sum((x - mean_val) ** 2 for x in numbers) / len(numbers)

    def standard_deviation(self, numbers):
        return self.variance(numbers) ** 0.5


    def calculate_std_font_size(self, image_paths):
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
            


        # Calculate the overall average font size across all images
        if total_font_sizes:
            overall_std_font_size = self.standard_deviation(total_font_sizes)
            return overall_std_font_size
        else:
            print("No text detected in any image.")
            return None

    def std2_font_size(self, image_paths):
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
            std_dev = self.standard_deviation(font_sizes)
            return std_dev
    def runStd(self, image_paths):
        length = len(self.topImages)
        if length > 1:
            std_font_size = self.calculate_std_font_size(self.topImages)
            print("Standard deviation of more than one image", std_font_size)
        else:
            std_font_size = self.std2_font_size(image_paths)
            print("Standard deviation of one image", std_font_size)
    

    def angleOfText(self, filePath):
        img = cv2.imread(filePath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # threshold
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        # invert
        thresh = 255 - thresh

        # apply horizontal morphology close
        kernel = np.ones((5 ,191), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # get external contours
        contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        # draw contours
        coords = []
        coordsList = []
        result = img.copy()
        for cntr in contours:
            # get bounding boxes
            pad = 10
            x,y,w,h = cv2.boundingRect(cntr)
            cv2.rectangle(result, (x-pad, y-pad), (x+w+pad, y+h+pad), (0, 0, 255), 4)
            coords.append((x,y))
            coords.append((x+w,y))
            coords.append((x+w, y+h))
            coords.append((x,y+h))
            coordsList.append(coords)

    def angle_of_rectangle(self, rectangle):
        # Assuming rectangle is a list of points [(x1, y1), (x2, y2), ...]
        
        # Find the length of each side of the rectangle
        side_lengths = [((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) for p1, p2 in zip(rectangle, rectangle[1:] + rectangle[:1])]
        
        # Find the index of the longest side
        longest_side_index = side_lengths.index(max(side_lengths))
        
        # Get the two endpoints of the longest side
        point1 = rectangle[longest_side_index]
        point2 = rectangle[(longest_side_index + 1) % len(rectangle)]  # Wrap around to the start if needed
        
        # Calculate the vector representing the longer side
        vector_x = point2[0] - point1[0]
        vector_y = point2[1] - point1[1]
        
        # Calculate the angle between the vector and the x-axis
        angle_rad = math.atan2(vector_y, vector_x)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
    def preprocess_image(self,image_path):
        # Load the image
        image = cv2.imread(image_path)
        
        # Convert to grayscale

        
        # Apply Canny edge detection
        edges = cv2.Canny(image, 50, 150)
        
        return edges

    def find_text_boxes(self, image):
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

    def calculate_distances(self, text_boxes):
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
    def LineSpacing(self):
        # Path to input image containing text
        input_image_path = 'VVImages/BoxImages/9.png'
        
        # Preprocess the image (e.g., edge detection)
        edges = self.preprocess_image(input_image_path)
        
        # Find potential text bounding boxes
        text_boxes = self.find_text_boxes(edges)
        
        # Calculate vertical distances between text lines
        distances = self.calculate_distances(text_boxes)
        
        # Print individual line spacing
        pos_nos = [num for num in distances if num >= 0]
        if max(pos_nos)- min(pos_nos) < 2:
            print("The line spacing is uniform.")
        else:
            print("The line spacing is not uniform.")
            self.valid = False
    
def main():
    filename = 'Varpratap I797 (1).PDF'
    pdfChecker = PDFChecker(filename)
if __name__ == "__main__":
    main()