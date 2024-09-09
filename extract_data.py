import numpy as np
import os
import cv2
import glob
import shutil
import pytesseract
import re
import time
import argparse
from statistics import mode
from pdf2image import convert_from_path  # Added for PDF conversion

regex = r"P\d{17}"
found = {}
results = {}
queue = []
done = []
missing = []
pnr_area = [150, 450, 1600, 1150]  # [start_x, start_y, end_x, end_y]


def apply_threshold(img, argument):
    switcher = {
        1: cv2.threshold(cv2.GaussianBlur(img, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        2: cv2.threshold(cv2.GaussianBlur(img, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        3: cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        4: cv2.threshold(cv2.medianBlur(img, 5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        5: cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        6: cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 2),
        7: cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
    }
    return switcher.get(argument, "Invalid method")


def crop_image(img, start_x, start_y, end_x, end_y):
    cropped = img[start_y:end_y, start_x:end_x]
    return cropped


def get_string(img_path, method):
    # Read image using opencv
    img = cv2.imread(img_path)
    file_name = os.path.basename(img_path).split('.')[0]
    file_name = file_name.split()[0]

    output_path = os.path.join(output_dir, file_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Crop the areas where provision number is more likely present
    # img = crop_image(img, pnr_area[0], pnr_area[1], pnr_area[2], pnr_area[3])

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Apply threshold to get image with only black and white
    img = apply_threshold(img, method)
    save_path = os.path.join(output_path, file_name + "_filter_" + str(method) + ".jpg")
    cv2.imwrite(save_path, img)

    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(img, lang="eng")

    return result


def find_match(regex, text):
    matches = re.finditer(regex, text, re.MULTILINE)
    target = ""
    for matchNum, match in enumerate(matches):
        matchNum = matchNum + 1
        print(f"  Match {matchNum} was found at {match.start()}-{match.end()}: {match.group()}")
        target = match.group()

    return target


def pretty_print(result_dict):
    s = ''
    for key in result_dict:
        s += '# ' + key + ': ' + result_dict[key] + '\n'
    return s


def convert_pdf_to_images(pdf_path, output_dir):
    # Convert each page in the PDF to an image
    images = convert_from_path(pdf_path)
    file_name = os.path.basename(pdf_path).split('.')[0]
    img_paths = []

    for i, image in enumerate(images):
        img_path = os.path.join(output_dir, f"{file_name}_page_{i}.jpg")
        image.save(img_path, 'JPEG')
        img_paths.append(img_path)

    return img_paths


if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    input_dir = "./ocr/data/"
    output_dir = "./ocr/results/"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    im_names = glob.glob(os.path.join(input_dir, '*.png')) + \
               glob.glob(os.path.join(input_dir, '*.jpg')) + \
               glob.glob(os.path.join(input_dir, '*.jpeg')) + \
               glob.glob(os.path.join(input_dir, '*.pdf'))  # Added PDFs to the search

    overall_start_t = time.time()
    for im_name in sorted(im_names):
        queue.append(im_name)

    print(f"The following files will be processed and their provision numbers will be extracted: {queue}\n")

    for im_name in im_names:
        start_time = time.time()
        print(f"*** The documents that are in the queue *** \n{queue}\n")

        print('#=======================================================')
        print(f'# Regex is being applied on {im_name}')
        print('#=======================================================')
        queue.remove(im_name)
        file_name = im_name.split(".")[0].split("/")[-1]

        # If the file is a PDF, convert it to images
        if im_name.endswith('.pdf'):
            image_paths = convert_pdf_to_images(im_name, output_dir)
        else:
            image_paths = [im_name]

        for img_path in image_paths:
            i = 1
            while i < 8:
                print(f"> The filter method {i} is now being applied.")
                result = get_string(img_path, i)
                match = find_match(regex, result)
                if match:
                    if file_name in found:
                        found[file_name].append(match)
                    else:
                        found[file_name] = [match]

                f = open(os.path.join(output_dir, file_name, f"{file_name}_filter_{i}.txt"), 'w')
                f.write(result)
                f.close()
                i += 1

        pnr = ''
        if file_name in found:
            pnr = mode(found[file_name])
            results[file_name] = pnr
            done.append(file_name)
        else:
            missing.append(file_name)
        end_time = time.time()

        print(f'#=======================================================\n'
              f'# Results for: {file_name}\n'
              f'#=======================================================\n'
              f'# The provision number: {pnr}\n'
              f'# It took {end_time - start_time} seconds.\n'
              f'#=======================================================\n')

    overall_end_t = time.time()

    print(f'#=======================================================\n'
          f'# Summary \n'
          f'#=======================================================\n'
          f'# The documents that are successfully processed are: \n{pretty_print(results)}'
          f'#=======================================================\n'
          f'# The program failed to extract information from: \n'
          f'# {str(missing)}\n'
          f'#=======================================================\n'
          f'# It took {overall_end_t - overall_start_t} seconds.\n'
          f'#=======================================================\n')
