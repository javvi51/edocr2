import cv2, string, time
from edocr2 import tools
           
file_path = 'tests/test_samples/4132864-.jpg'
img = cv2.imread(file_path)
#img = convert_from_path(file_path)
#img = np.array(img[0])

#region ############# Alphabet definition #################
GDT_symbols = '⏤⏥○⌭⌒⌓⏊∠⫽⌯⌖◎↗⌰'
FCF_symbols = 'ⒺⒻⓁⓂⓅⓈⓉⓊ'
Extra = '(),.+-±:/°"⌀'

alphabet_gdts = string.digits + ',.⌀ABCD' + GDT_symbols + FCF_symbols
alphabet_dimensions = string.digits + 'AaBCDRGHhMmnx' + Extra
#alphabet_dimensions = string.digits + string.ascii_lowercase + string.ascii_uppercase + Extra
#endregion

#region ############ Segmentation Task ###################
start_time = time.time()

img_boxes, process_img, frame, gdt_boxes, tables  = tools.layer_segm.segment_img(img, frame = True, GDT_thres = 0.02)

end_time = time.time()
print(f"\033[1;33mSegmentation took {end_time - start_time:.6f} seconds to run.\033[0m")
#endregion

#region ############ OCR Tables ###########################
start_time = time.time()

table_results, updated_tables = tools.ocr_pipelines.ocr_tables(tables, img)

end_time = time.time()
print(f"\033[1;33mOCR in tables took {end_time - start_time:.6f} seconds to run.\033[0m")
#endregion

#region ############ OCR GD&T #############################
start_time = time.time()

gdt_model = 'edocr2/models/recognizer_10_41.keras'

if gdt_boxes:
    gdt_results, updated_gdt_boxes = tools.ocr_pipelines.ocr_gdt(img, gdt_boxes, alphabet_gdts, gdt_model)
else:
    gdt_results, updated_gdt_boxes = [], []

end_time = time.time()
print(f"\033[1;33mOCR in GD&T took {end_time - start_time:.6f} seconds to run.\033[0m")
#endregion

#region ############ OCR Dimensions #######################
start_time = time.time()

dimension_model = 'edocr2/models/recognizer_13_44.keras'
detector = None #'detector_15_37.keras'

if frame:
    process_img = process_img[frame.y : frame.y + frame.h, frame.x : frame.x + frame.w]

dimensions = tools.ocr_pipelines.ocr_dimensions(process_img, alphabet_dimensions, detector, dimension_model, cluster_thres=15, backg_save=False)

end_time = time.time()
print(f"\033[1;33mOCR in dimensions took {end_time - start_time:.6f} seconds to run.\033[0m")
#endregion

############ Output #######################
mask_img = tools.output_tools.mask_img(img, updated_gdt_boxes, updated_tables, dimensions, frame)

############ Communicating with LLM ###############

###################################################
cv2.imshow('boxes', mask_img)
cv2.waitKey(0)
cv2.destroyAllWindows()