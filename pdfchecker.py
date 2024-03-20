import numpy as np
import cv2
import pdf2image
import os
import pytesseract
from PIL import Image
import re
from skimage.transform import radon
from PIL import Image
from numpy import asarray, mean, array, blackman
from numpy.fft import rfft
try:
    from parabolic import parabolic

    def argmax(x):
        return parabolic(x, np.argmax(x))[0]
except ImportError:
    from numpy import argmax

class PDFChecker:
    def __init__(self, pdfPath):
        self.pdfPath = pdfPath
        self.images = pdf2image.convert_from_path(self.pdfPath)
        self.rotatePdf(self.images)
        self.box_extraction("VVImages/rotated0.jpg", "./VVImages/BoxImages/")

        self.textList = ["Receipt Number", "Receipt", "Beneficiary", "Petitioner","Page", "Notice Date" "Case Type", "Date of Birth"]

        self.textChecker("./VVImages/BoxImages/", self.textList)
        self.calculateSpacing("./VVImages/BoxImages/9.png")
        for root, dirs, files in os.walk("./VVImages/BoxImages/"):
            for filename in files:
                image_path = os.path.join(root, filename)
                


    def rotatePdf(self, images):
        for x, img in enumerate(images):
            img.save('VVImages/output'+str(x)+'.jpg', 'JPEG')
            # load file, converting to grayscale
            img = cv2.imread('VVImages/output'+str(x)+'.jpg')
            print(img.shape)
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


        #Functon for extracting the box
    def box_extraction(self, img_for_box_extraction_path, cropped_dir_path):

        print("Reading image..")
        img = cv2.imread(img_for_box_extraction_path, 0)  # Read the image
        (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
        img_bin = 255-img_bin  # Invert the image

        print("Storing binary image to Images/Image_bin.jpg..")
        cv2.imwrite("Images/Image_bin.jpg",img_bin)

        print("Applying Morphological Operations..")
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

        print("Output stored in Output directiory!")

        idx = 0
        for c in contours:
            # Returns the location and width,height for every contour
            x, y, w, h = cv2.boundingRect(c)

            # If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
            if (w > 80 and h > 20) and w > 3*h:
                idx += 1
                new_img = img[y:y+h, x:x+w]
                cv2.imwrite(cropped_dir_path+str(idx) + '.png', new_img)

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
        print('Line spacing: {:.2f} pixels'.format(line_spacing))

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
        print(total_font_sizes)
            


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
        

def main():
    filename = 'Varpratap I797 (1).PDF'
    pdfChecker = PDFChecker(filename)
if __name__ == "__main__":
    main()