import cv2, os
import pytesseract
from edocr2.keras_ocr.recognition import Recognizer

###################### Tables Pipeline #################################
def ocr_table_cv2(image_cv2):
    """Recognize text in an OpenCV image using pytesseract and return both text and positions.
    
    Args:
        image_cv2: OpenCV image object.
        
    Returns:
        A list of dictionaries containing recognized text and their positions (left, top, width, height).
    """
    # Convert the OpenCV image to RGB format (pytesseract expects this)
    img_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    
    # Custom configuration to recognize a more complete set of characters
    custom_config = r'--psm 6'

    # Perform OCR and get bounding box details
    ocr_data = pytesseract.image_to_data(img_rgb, config=custom_config, output_type=pytesseract.Output.DICT)

    # Prepare result: text with their positions
    result = []
    for i in range(len(ocr_data['text'])):
        if ocr_data['text'][i].strip():  # If text is not empty
            text_info = {
                'text': ocr_data['text'][i],
                'left': ocr_data['left'][i],
                'top': ocr_data['top'][i],
                'width': ocr_data['width'][i],
                'height': ocr_data['height'][i]
            }
            result.append(text_info)

    return result

def ocr_tables(tables, process_img):
    results = []
    updated_tables = []
    for table in tables:
        for b in table:
            img = process_img[b.y - 2 : b.y + b.h + 4, b.x - 2 : b.x + b.w + 4][:]
            result = ocr_table_cv2(img)
            if result == []:
                continue
            else:
                results.append(result)
                updated_tables.append(table)
    return results, updated_tables

##################### GDT Pipeline #####################################

def is_not_empty(img, boxes, color_thres):
    for box in boxes:
            # Extract the region of interest (ROI) from the image
        roi = img[box.y + 2:box.y + box.h - 4, box.x + 2:box.x + box.w -4]
        
        # Convert the ROI to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Check if all pixels are near black or near white
        min_val, max_val, _, _ = cv2.minMaxLoc(gray_roi)
        
        # If the difference between min and max pixel values is greater than the threshold, the box contains color
        if (max_val - min_val) < color_thres:
            return False
        
    return True

def sort_gdt_boxes(boxes, y_thres = 3):
    """Sorts boxes in reading order: left-to-right, then top-to-bottom.
    
    Args:
        boxes: List of Rect objects or any object with x, y, w, h attributes.
        y_threshold: A threshold to group boxes that are on the same line (default is 10 pixels).
    
    Returns:
        A list of boxes sorted in reading order.
    """
    # Sort by the y-coordinate first (top-to-bottom)
    boxes.sort(key=lambda b: b.y)

    sorted_boxes = []
    current_line = []
    current_y = boxes[0].y

    for box in boxes:
        # If the box's y-coordinate is close to the current line's y-coordinate, add it to the same line
        if abs(box.y - current_y) <= y_thres:
            current_line.append(box)
        else:
            # Sort the current line by x-coordinate (left-to-right)
            current_line.sort(key=lambda b: b.x)
            sorted_boxes.extend(current_line)
            
            # Start a new line with the current box
            current_line = [box]
            current_y = box.y
    
    # Sort the last line and add it
    current_line.sort(key=lambda b: b.x)
    sorted_boxes.extend(current_line)
    
    return sorted_boxes

def recognize_gdt(img, block, recognizer):
    roi = img[block[0].y + 2:block[0].y + block[0].h - 4, block[0].x + 2:block[0].x + block[0].w - 4]
    pred = recognizer.recognize(image = roi)

    for i in range(1, len(block)):
        new_line = block[i].y - block[i - 1].y > 5
        roi = img[block[i].y:block[i].y + block[i].h, block[i].x:block[i].x + block[i].w]
        p = recognizer.recognize(image = roi)
        if new_line:
            pred += '\n' + p
        else:
            pred += '|' + p

    return pred

def ocr_gdt(img, gdt_boxes, alphabet = None, model_path = None):

    if alphabet and model_path: 
        recognizer =Recognizer(alphabet = alphabet)
        recognizer.model.load_weights(model_path)
    else:
        recognizer =Recognizer()

    updated_gdts = []
    results = []
    for block in gdt_boxes:
        for _, bl_list in block.items():
            if is_not_empty(img, bl_list, 50):
                sorted_block = sort_gdt_boxes(bl_list, 3)
                pred = recognize_gdt(img, sorted_block, recognizer)
                updated_gdts.append(block)
                results.append([pred, (sorted_block[0].x, sorted_block[0].y)])
    
    return results, updated_gdts

##################### Dimension Pipeline ###############################


