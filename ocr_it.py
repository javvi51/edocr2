import cv2, string, os, time
import numpy as np
from edocr2 import tools
from pdf2image import convert_from_path
           
def ocr_drawing(file_path, recognizer_gdt, dimension_tuple, #Must have
                frame = True, language = 'eng', binary_thres= 127, #general
                GDT_thres = 0.02, #GD&Ts
                cluster_thres= 15, patches = (5,3), max_char = 15, #Dimensions
                output_path='.', save_mask=False, save_raw_output=False, backg_save=False #Output
                ):
    #Read file
    if file_path.endswith('.pdf'):
        img = convert_from_path(file_path)
        img = np.array(img[0])
    else:
        img = cv2.imread(file_path)
    #Layer Segmentation
    times = []
    start_time = time.time()
    img_boxes, process_img, frame, gdt_boxes, tables  = tools.layer_segm.segment_img(img, frame = frame, GDT_thres = GDT_thres, binary_thres= binary_thres)
    end_time = time.time()
    times.append(end_time-start_time)
    print('Segmentation Done')
    #OCR Tables
    table_results, updated_tables = tools.ocr_pipelines.ocr_tables(tables, img, language=language)
    end_time = time.time()
    times.append(end_time-sum(times)-start_time)
    print('Prediction on Tables Done')
    #GD&T OCR
    gdt_results, updated_gdt_boxes = tools.ocr_pipelines.ocr_gdt(img, gdt_boxes,recognizer=recognizer_gdt)
    end_time = time.time()
    times.append(end_time-sum(times)-start_time)
    print('Prediction on GD&T Done')
    #Dimension  OCR
    if frame:
        process_img = process_img[frame.y : frame.y + frame.h, frame.x : frame.x + frame.w]

    dimensions, other_info, process_img = tools.ocr_pipelines.ocr_dimensions(process_img, dimension_tuple[0], dimension_tuple[1], 
                                                    patches=patches, max_char=max_char, cluster_thres=cluster_thres, backg_save=backg_save)
    end_time = time.time()
    times.append(end_time-sum(times)-start_time)
    print('Prediction on dimensions and extra information Done')
    #Saving
    if save_mask or save_raw_output:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_path, filename)
        os.makedirs(output_path, exist_ok=True)

    if save_mask:
        mask_img = tools.output_tools.mask_img(img, updated_gdt_boxes, updated_tables, dimensions, frame, other_info)
        cv2.imwrite(os.path.join(output_path, filename + '_mask.png'), mask_img)
        print('Mask saved')

    table_results, gdt_results, dimensions, other_info  = tools.output_tools.process_raw_output(output_path, table_results, gdt_results, dimensions, other_info, save=save_raw_output)
    end_time = time.time()
    times.append(end_time-sum(times)-start_time)
    print('Raw output saved')

    return {'tab': table_results, 'gdts': gdt_results, 'dim': dimensions, 'other': other_info}, times

def ocr_one_drawing():
    file_path = '/home/javvi51/edocr2/tests/test_samples/4132864.jpg'
    
    GDT_symbols = '⏤⏥○⌭⌒⌓⏊∠⫽⌯⌖◎↗⌰'
    FCF_symbols = 'ⒺⒻⓁⓂⓅⓈⓉⓊ'
    Extra = '(),.+-±:/°"⌀'

    alphabet_gdts = string.digits + ',.⌀ABCD' + GDT_symbols + FCF_symbols
    alphabet_dimensions = string.digits + 'AaBCDRGHhMmnx' + Extra

    #Session Loading
    start_time = time.time()
    from edocr2.keras_ocr.recognition import Recognizer
    from edocr2.keras_ocr.detection import Detector

    recognizer_gdt = Recognizer(alphabet=alphabet_gdts)
    recognizer_gdt.model.load_weights('edocr2/models/recognizer_gdts.keras')

    recognizer_dim = Recognizer(alphabet=alphabet_dimensions)
    recognizer_dim.model.load_weights('edocr2/models/recognizer_dimensions.keras')

    detector = Detector()
    #detector.model.load_weights('path/to/custom/detector')

    #Warming up models:
    dummy_image = np.zeros((1, 1, 3), dtype=np.float32)
    _ = recognizer_gdt.recognize(dummy_image)
    _ = recognizer_dim.recognize(dummy_image)
    dummy_image = np.zeros((32, 32, 3), dtype=np.float32)
    _ = detector.detect([dummy_image])
    end_time = time.time()

    kwargs = {
        #General
        'file_path': file_path, #MUST: Image or pdf path to OCR
        'binary_thres': 127, #Pixel value (0-255) to detect contourns, i.e, identify rectangles in the image
        'language': 'eng', #Language of the drawing, require installation of tesseract speficic language if not english
        'frame' : True, #Do we want to spot a frame as the maximum rectangle?
        #GD&T
        'recognizer_gdt': recognizer_gdt, #MUST: A Tuple with (gdt alphabet, model path)
        'GDT_thres': 0.02, #Maximum porcentage of the image area to consider a cluster of rectangles a GD&T box
        #Dimensions
        'dimension_tuple': (detector, recognizer_dim), #MUST: A Tuple with (detector, dimension alphabet, model path)
        'cluster_thres': 20, #Minimum distance in pixels between two text predictions to consider the same text box
        'patches': (5, 3), #Tuple with number of patches in X and Y direction. To ease text detection on large images
        'max_char': 15, #Max number of characters to consider a text prediction a dimension, otherwise -> other info
        #Output
        'backg_save': False, #Option to save the background once all text and boxes have been removed, for synth training purposes
        'output_path': '.', #Output path
        'save_mask': True, #Option to save the mask output
        'save_raw_output': True, #Option to save raw ouput, i.e, OCR text and box position,
        }
    
    results, times = ocr_drawing(**kwargs)
    final_time = time.time()
    print(
    "Session Timing Report:\n"
    "Loading session:           {:.3f} s\n"
    "----------------------\n"
    "Drawing segmentation:      {:.3f} s\n"
    "Table prediction:          {:.3f} s\n"
    "GD&T prediction:           {:.3f} s\n"
    "Dimension & other info:    {:.3f} s\n"
    "Saving & processing:       {:.3f} s\n"
    "----------------------\n"
    "OCR in {}: {:.3f} s\n"
    "----------------------\n"
    "Total time:                {:.3f} s\n"
    .format(end_time - start_time, times[0], times[1], times[2], times[3], times[4], os.path.basename(file_path), sum(times), final_time - start_time)
)

def ocr_folder():
    folder_path = '/home/javvi51/edocr2/tests/test_samples/Washers'
    
    GDT_symbols = '⏤⏥○⌭⌒⌓⏊∠⫽⌯⌖◎↗⌰'
    FCF_symbols = 'ⒺⒻⓁⓂⓅⓈⓉⓊ'
    Extra = '(),.+-±:/°"⌀'

    alphabet_gdts = string.digits + ',.⌀ABCD' + GDT_symbols + FCF_symbols
    alphabet_dimensions = string.digits + 'AaBCDRGHhMmnx' + Extra

    #Session Loading
    start_time = time.time()
    from edocr2.keras_ocr.recognition import Recognizer
    from edocr2.keras_ocr.detection import Detector

    recognizer_gdt = Recognizer(alphabet=alphabet_gdts)
    recognizer_gdt.model.load_weights('edocr2/models/recognizer_gdts.keras')

    recognizer_dim = Recognizer(alphabet=alphabet_dimensions)
    recognizer_dim.model.load_weights('edocr2/models/recognizer_dimensions.keras')

    detector = Detector()
    #detector.model.load_weights('path/to/custom/detector')

    #Warming up models:
    dummy_image = np.zeros((1, 1, 3), dtype=np.float32)
    _ = recognizer_gdt.recognize(dummy_image)
    _ = recognizer_dim.recognize(dummy_image)
    dummy_image = np.zeros((32, 32, 3), dtype=np.float32)
    _ = detector.detect([dummy_image])
    end_time = time.time()

    file_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename))]
    times =[]
    for file_path in file_paths:
        kwargs = {
            #General
            'file_path': file_path, #MUST: Image or pdf path to OCR
            'binary_thres': 127, #Pixel value (0-255) to detect contourns, i.e, identify rectangles in the image
            'language': 'eng', #Language of the drawing, require installation of tesseract speficic language if not english
            'frame' : True, #Do we want to spot a frame as the maximum rectangle?
            #GD&T
            'recognizer_gdt': recognizer_gdt, #MUST: A Tuple with (gdt alphabet, model path)
            'GDT_thres': 0.02, #Maximum porcentage of the image area to consider a cluster of rectangles a GD&T box
            #Dimensions
            'dimension_tuple': (detector, recognizer_dim), #MUST: A Tuple with (detector, dimension alphabet, model path)
            'cluster_thres': 20, #Minimum distance in pixels between two text predictions to consider the same text box
            'patches': (5, 3), #Tuple with number of patches in X and Y direction. To ease text detection on large images
            'max_char': 15, #Max number of characters to consider a text prediction a dimension, otherwise -> other info
            #Output
            'backg_save': False, #Option to save the background once all text and boxes have been removed, for synth training purposes
            'output_path': '.', #Output path
            'save_mask': False, #Option to save the mask output
            'save_raw_output': False, #Option to save raw ouput, i.e, OCR text and box position,
            }
        
        results_, times_ = ocr_drawing(**kwargs)
        times.append(sum(times_))
        print(
            "OCR in {}:\n"
            "Drawing segmentation:      {:.3f} s\n"
            "Table prediction:          {:.3f} s\n"
            "GD&T prediction:           {:.3f} s\n"
            "Dimension & other info:    {:.3f} s\n"
            "Saving & processing:       {:.3f} s\n"
            "----------------------\n"
            "Total time:                {:.3f} s\n"
            .format(os.path.basename(file_path),times_[0], times_[1], times_[2], times_[3], times_[4], sum(times_)))

    final_time = time.time()
    print(
    "Session Timing Report:\n"
    "Loading session:           {:.3f} s\n"
    "----------------------\n"
    .format(end_time - start_time))
    for i in range(len(file_paths)):
        print("OCR in {}: {:.3f} s".format(os.path.basename(file_paths[i]), times[i]))
    print(
        "----------------------\n"
        "Total time:           {:.3f} s\n"
        .format(final_time - start_time))

ocr_folder()