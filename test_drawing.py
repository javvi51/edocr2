import cv2, string, time
from edocr2 import tools
           
file_path = 'tests/test_samples/halter.jpg'
img = cv2.imread(file_path)
#img = convert_from_path(file_path)
#img = np.array(img[0])

############# Alphabet definition #################
GDT_symbols = '⏤⏥○⌭⌒⌓⏊∠⫽⌯⌖◎↗⌰'
FCF_symbols = 'ⒺⒻⓁⓂⓅⓈⓉⓊ'
Extra = '(),.+-±:/°"⌀'

alphabet_gdts = string.digits + ',.⌀ABCD' + GDT_symbols #+ FCF_symbols
alphabet_dimensions = string.digits + 'AaBCDRGHhMmnx' + Extra

############# Segmentation Task ###################
start_time = time.time()

img_boxes, process_img, frame, gdt_boxes, tables  = tools.layer_segm.segment_img(img, frame = True, GDT_thres = 0.02)

end_time = time.time()
print(f"Segmentation took {end_time - start_time:.6f} seconds to run.")
           
############ OCR Tables ###########################
start_time = time.time()

table_results, updated_tables = tools.ocr_pipelines.ocr_tables(tables, img)

end_time = time.time()
print(f"OCR in tables took {end_time - start_time:.6f} seconds to run.")

############ OCR GD&T #############################
start_time = time.time()

gdt_model = 'recognizer_gdts.h5'
gdt_results, updated_gdt_boxes = tools.ocr_pipelines.ocr_gdt(img, gdt_boxes, alphabet_gdts, gdt_model)
gdt_boxes = updated_gdt_boxes

end_time = time.time()
print(f"OCR in GD&T took {end_time - start_time:.6f} seconds to run.")


############ OCR Dimensions #######################
start_time = time.time()

if frame:
    process_img = process_img[frame.y : frame.y + frame.h, frame.x : frame.x + frame.w]

dimensions = []

end_time = time.time()
print(f"OCR in dimensions took {end_time - start_time:.6f} seconds to run.")

############ Masking Images #######################
mask_img = tools.mask_results.mask_img(img, updated_gdt_boxes, updated_tables, dimensions, frame)

############ Communicating with LLM ###############

###################################################
cv2.imshow('boxes', mask_img)
cv2.waitKey(0)
cv2.destroyAllWindows()




