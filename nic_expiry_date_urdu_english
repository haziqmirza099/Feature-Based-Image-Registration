from flask import Flask, request, jsonify
import cv2
import easyocr
import numpy as np
import os
import re
from datetime import datetime
from scipy.spatial import distance

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], model_storage_directory='easyocr_models')

app = Flask(__name__)

# Output dictionary for detected regions
output_dict = {
    'Name': [[0.283, 0.271], [0.415, 0.271], [0.415, 0.325], [0.283, 0.325]],
    'Father Name': [[0.29, 0.456], [0.494, 0.456], [0.494, 0.514], [0.29, 0.514]],
    'Date of Birth': [[0.529, 0.751], [0.648, 0.751], [0.648, 0.803], [0.529, 0.803]],
    'Date of Issue': [[0.285, 0.857], [0.404, 0.857], [0.404, 0.908], [0.285, 0.908]],
    'Date of Expiry': [[0.531, 0.859], [0.65, 0.859], [0.65, 0.911], [0.531, 0.911]],
    'تاریخ تنسیخ': [[0.1, 0.5], [0.6, 0.5], [0.6, 0.7], [0.1, 0.7]]
}

# Ensure the folder exists for saving uploaded images
UPLOAD_FOLDER = 'uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def normalize(img, result):
    h, w = img.shape[:-1]
    normalize_bbx = []
    detected_labels = []
    for (bbox, text, prob) in result:
        (tl, tr, br, bl) = bbox
        tl[0], tl[1] = round(tl[0] / w, 3), round(tl[1] / h, 3)
        tr[0], tr[1] = round(tr[0] / w, 3), round(tr[1] / h, 3)
        br[0], br[1] = round(br[0] / w, 3), round(br[1] / h, 3)
        bl[0], bl[1] = round(bl[0] / w, 3), round(bl[1] / h, 3)
        normalize_bbx.append([tl, tr, br, bl])
        detected_labels.append(text)
    return normalize_bbx, detected_labels

def calculate_distance(key, bbx):
    euc_sum = 0
    for val1, val2 in zip(key, bbx):
        euc_sum += distance.euclidean(val1, val2)
    return euc_sum

def get_value(key, normalize_output):
    distances = {}
    for bbx, text in normalize_output:
        distances[text] = calculate_distance(key, bbx)
    return distances

def validate_expiry_date(date_text, date_pattern=r"\b(\d{2}[./-]\d{2}[./-]\d{4})\b"):
    expiry_date_candidates = re.findall(date_pattern, date_text)
    if expiry_date_candidates:
        try:
            expiry_date_obj = None
            for candidate in expiry_date_candidates:
                try:
                    if "/" in candidate:
                        parsed_date = datetime.strptime(candidate, "%d/%m/%Y")
                    else:
                        parsed_date = datetime.strptime(candidate, "%d.%m.%Y")
                except ValueError:
                    continue
                if not expiry_date_obj or parsed_date > expiry_date_obj:
                    expiry_date_obj = parsed_date

            if expiry_date_obj:
                today = datetime.now()
                if today > expiry_date_obj:
                    return f"The document is expired. Expiry Date: {expiry_date_obj.strftime('%d/%m/%Y')}", False
                else:
                    return f"The document is valid. Expiry Date: {expiry_date_obj.strftime('%d/%m/%Y')}", True
            else:
                return "Could not parse any valid expiry date.", False
        except ValueError:
            return "Could not parse the expiry date.", False
    else:
        return "Expiry date not found in the image.", False

@app.route('/process', methods=['GET'])
def process_documents():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({"error": "Both images are required."}), 400

    # Save the uploaded images to the UPLOAD_FOLDER
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    img1_path = os.path.join(UPLOAD_FOLDER, file1.filename)
    img2_path = os.path.join(UPLOAD_FOLDER, file2.filename)
    
    file1.save(img1_path)
    file2.save(img2_path)

    # Read the images using OpenCV
    image1 = cv2.imread(img1_path)
    image2 = cv2.imread(img2_path)

    # Process text extraction with EasyOCR
    result1 = reader.readtext(img1_path)
    result2 = reader.readtext(img2_path)

    # Normalize bounding boxes
    norm_boxes_image1, labels_image1 = normalize(image1, result1)
    norm_boxes_image2, labels_image2 = normalize(image2, result2)

    dict_data = {}

    # Extract relevant data from both images
    for key, value in output_dict.items():
        distances_image1 = get_value(value, zip(norm_boxes_image1, labels_image1))
        distances_image2 = get_value(value, zip(norm_boxes_image2, labels_image2))

        min_distance_image1 = min(distances_image1.items(), key=lambda x: x[1])
        min_distance_image2 = min(distances_image2.items(), key=lambda x: x[1])

        dict_data[key] = {
            'Image 1': min_distance_image1[0],
            'Image 2': min_distance_image2[0]
        }

    # Extract expiry date text from the results
    expiry_text_image1 = dict_data.get('Date of Expiry', {}).get('Image 1', '')
    expiry_text_image2 = dict_data.get('تاریخ تنسیخ', {}).get('Image 2', '')

    # Validate expiry date
    expiry_validation_image1, is_valid_image1 = validate_expiry_date(expiry_text_image1)
    expiry_validation_image2, is_valid_image2 = validate_expiry_date(expiry_text_image2)

    # Determine overall document status
    overall_status = "Valid" if is_valid_image1 or is_valid_image2 else "Expired"

    # Return the response with document data and expiry validation status
    return jsonify({
       # "Document Data": dict_data,
        "Expiry Validation Image 1": expiry_validation_image1,
        "Expiry Validation Image 2": expiry_validation_image2,
        "Overall Status": overall_status
    })

if __name__ == '__main__':
    app.run(debug=True)
