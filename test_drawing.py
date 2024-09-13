import time
import cv2
from edocr2 import tools

file_path='tests/test_samples/BM_part.jpg'
img=cv2.imread(file_path)
#img = convert_from_path(file_path)
#img = np.array(img[0])

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

updated_gdt_boxes = gdt_boxes

end_time = time.time()
print(f"OCR in tables took {end_time - start_time:.6f} seconds to run.")


############ OCR Dimensions #######################
start_time = time.time()

if frame:
    process_img = process_img[frame.y : frame.y + frame.h, frame.x : frame.x + frame.w]

dimensions = []

end_time = time.time()
print(f"OCR in tables took {end_time - start_time:.6f} seconds to run.")

############ Masking Images #######################
mask_img = tools.mask_results.mask_img(img, updated_gdt_boxes, updated_tables, dimensions, frame)

############ Communicating with LLM ###############

###################################################
cv2.imshow('boxes', mask_img)
cv2.waitKey(0)
cv2.destroyAllWindows()




