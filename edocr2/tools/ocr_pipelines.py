import cv2
import pytesseract


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


##################### Dimension Pipeline ###############################


