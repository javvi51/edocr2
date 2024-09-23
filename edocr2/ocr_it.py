import cv2, string, os
from edocr2 import tools
           
def ocr_drawing(file_path, gdt_tuple, dimension_tuple,
                frame = True, GDT_thres = 0.02, binary_thres= 127, 
                cluster_thres= 15, backg_save=False,
                output_path='.', save_mask=False, save_raw_output=False

                ):
    img = cv2.imread(file_path)
    #Layer Segmentation
    img_boxes, process_img, frame, gdt_boxes, tables  = tools.layer_segm.segment_img(img, frame = frame, GDT_thres = GDT_thres, binary_thres= binary_thres)
    
    #OCR Tables
    table_results, updated_tables = tools.ocr_pipelines.ocr_tables(tables, img)

    #GD&T OCR
    if gdt_boxes:
        gdt_results, updated_gdt_boxes = tools.ocr_pipelines.ocr_gdt(img, gdt_boxes, alphabet=gdt_tuple[0], model_path=gdt_tuple[1])
    else:
        gdt_results, updated_gdt_boxes = [], []

    #Dimension  OCR
    if frame:
        process_img = process_img[frame.y : frame.y + frame.h, frame.x : frame.x + frame.w]

    dimensions = tools.ocr_pipelines.ocr_dimensions(process_img, 
                                                    detector_path=dimension_tuple[0], 
                                                    alphabet=dimension_tuple[1], 
                                                    recognizer_path=dimension_tuple[2], cluster_thres=cluster_thres, backg_save=backg_save)
    #Saving
    if output_path:
        filename = os.path.splitext(os.path.basename(file_path)[0])
        os.makedirs(os.path.join(output_path, filename), exist_ok=True)

        if save_mask:
            mask_img = tools.output_tools.mask_img(img, updated_gdt_boxes, updated_tables, dimensions, frame)
            cv2.imwrite(os.path.join(output_path, filename, filename + '_mask.png'), mask_img)

        if save_raw_output:
            pass

if __name__ == 'main':
    file_path = 'tests/test_samples/4132864-.jpg'
    
    GDT_symbols = '⏤⏥○⌭⌒⌓⏊∠⫽⌯⌖◎↗⌰'
    FCF_symbols = 'ⒺⒻⓁⓂⓅⓈⓉⓊ'
    Extra = '(),.+-±:/°"⌀'

    alphabet_gdts = string.digits + ',.⌀ABCD' + GDT_symbols + FCF_symbols
    alphabet_dimensions = string.digits + 'AaBCDRGHhMmnx' + Extra

    kwargs = {
        'filepath': file_path, #MUST: Image path to OCR
        'gdt_tuple': (alphabet_gdts, 'edocr2/models/recognizer_gdts.keras'), #MUST: A Tuple with (gdt alphabet, model path)
        'dimension_tuple': (None, alphabet_dimensions, 'edocr2/models/recognizer_13_44.keras'), #MUST: A Tuple with (detector, dimension alphabet, model path)
        'frame' : True, #Do we want to spot a frame as the maximum rectangle?
        'GDT_thres': 0.02, #Maximum porcentage of the image area to consider a cluster of rectangles a GD&T box
        'binary_thres': 127, #Pixel value (0-255) to detect contourns, i.e, identify rectangles in the image
        'cluster_thres': 15, #Minimum distance in pixels between two text predictions to consider the same text box
        'backg_save': False, #Option to save the background once all text and boxes have been removed, for synth training purposes
        'output_path': '.', #Output path
        'save_mask': True, #Option to save the mask output
        'save_raw_output': False, #Option to save raw ouput, i.e, OCR text and box position,
        }
    
    ocr_drawing(**kwargs)