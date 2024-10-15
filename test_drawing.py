import cv2, string, time, os
import numpy as np
from edocr2 import tools
from pdf2image import convert_from_path
           
file_path = 'tests/test_samples/Adapterplatte.jpg'

if file_path.endswith('.pdf') or file_path.endswith(".PDF"):
    img = convert_from_path(file_path)
    img = np.array(img[0])
else:
    img = cv2.imread(file_path)

filename = os.path.splitext(os.path.basename(file_path))[0]
output_path = os.path.join('.', filename)

#region ############# Alphabet definition #################
GDT_symbols = '⏤⏥○⌭⌒⌓⏊∠⫽⌯⌖◎↗⌰'
FCF_symbols = 'ⒺⒻⓁⓂⓅⓈⓉⓊ'
Extra = '(),.+-±:/°"⌀'

alphabet_gdts = string.digits + ',.⌀ABCD' + GDT_symbols + FCF_symbols
alphabet_dimensions = string.digits + 'AaBCDRGHhMmnx' + Extra
language = 'swe'
#endregion

#region ############ Segmentation Task ####################
start_time = time.time()

img_boxes, frame, gdt_boxes, tables  = tools.layer_segm.segment_img(img, autoframe = True, frame_thres=0.7, GDT_thres = 0.02, binary_thres=127)

end_time = time.time()
print(f"\033[1;33mSegmentation took {end_time - start_time:.6f} seconds to run.\033[0m")
#endregion

#region ######## Set Session ##############################
start_time = time.time()
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import tensorflow as tf
from edocr2.keras_ocr.recognition import Recognizer
from edocr2.keras_ocr.detection import Detector


# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load models
gdt_model = 'edocr2/models/recognizer_gdts.keras'
dim_model = 'edocr2/models/recognizer_dimensions.keras'
detector_model = None #'edocr2/models/detector_12_46.keras'

recognizer_gdt = None
if gdt_boxes:
    recognizer_gdt = Recognizer(alphabet=alphabet_gdts)
    recognizer_gdt.model.load_weights(gdt_model)

recognizer_dim = Recognizer(alphabet=alphabet_dimensions)
recognizer_dim.model.load_weights(dim_model)
detector = Detector()

if detector_model:
    detector.model.load_weights(detector_model)

end_time = time.time()   
print(f"\033[1;33mLoading session took {end_time - start_time:.6f} seconds to run.\033[0m")
#endregion

#region ############ OCR Tables ###########################
start_time = time.time()

table_results, updated_tables, process_img= tools.ocr_pipelines.ocr_tables(tables, img, language)
end_time = time.time()
print(f"\033[1;33mOCR in tables took {end_time - start_time:.6f} seconds to run.\033[0m")
#endregion

#region ############ OCR GD&T #############################
start_time = time.time()
gdt_results, updated_gdt_boxes, process_img = tools.ocr_pipelines.ocr_gdt(process_img, gdt_boxes, recognizer_gdt)
end_time = time.time()
print(f"\033[1;33mOCR in GD&T took {end_time - start_time:.6f} seconds to run.\033[0m")
#endregion

#region ############ OCR Dimensions #######################
start_time = time.time()

if frame:
    process_img = process_img[frame.y : frame.y + frame.h, frame.x : frame.x + frame.w]

dimensions, other_info, process_img, dim_tess = tools.ocr_pipelines.ocr_dimensions(process_img, detector, recognizer_dim, alphabet_dimensions, cluster_thres=20, max_img_size=1024, language=language, backg_save=False)

end_time = time.time()
print(f"\033[1;33mOCR in dimensions took {end_time - start_time:.6f} seconds to run.\033[0m")
#endregion

#region ########### Output ################################
start_time = time.time()
mask_img = tools.output_tools.mask_img(img, updated_gdt_boxes, updated_tables, dimensions, frame, other_info)

table_results, gdt_results, dimensions, other_info = tools.output_tools.process_raw_output(output_path, table_results, gdt_results, dimensions, other_info, save=False)

end_time = time.time()
print(f"\033[1;33mRaw output generation took {end_time - start_time:.6f} seconds to run.\033[0m")
#endregion

###################################################
cv2.imshow('boxes', mask_img)
cv2.waitKey(0)
cv2.destroyAllWindows()