import cv2, time, os
import numpy as np
from edocr2 import tools
from pdf2image import convert_from_path

file_path = 'tests/test_samples/Candle_holder.jpg'
language = 'eng'

#Opening the file        
if file_path.endswith('.pdf') or file_path.endswith(".PDF"):
    img = convert_from_path(file_path)
    img = np.array(img[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    img = cv2.merge([img, img, img])
else:
    img = cv2.imread(file_path)

filename = os.path.splitext(os.path.basename(file_path))[0]
output_path = os.path.join('.', filename)


#region ############ Segmentation Task ####################

img_boxes, frame, gdt_boxes, tables, dim_boxes  = tools.layer_segm.segment_img(img, autoframe = True, frame_thres=0.7, GDT_thres = 0.02, binary_thres=127)

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
dim_model = 'edocr2/models/recognizer_dimensions_2.keras'
detector_model = None #'edocr2/models/detector_12_46.keras'

recognizer_gdt = None
if gdt_boxes:
    recognizer_gdt = Recognizer(alphabet=tools.ocr_pipelines.read_alphabet(gdt_model))
    recognizer_gdt.model.load_weights(gdt_model)
alphabet_dim = tools.ocr_pipelines.read_alphabet(dim_model)
recognizer_dim = Recognizer(alphabet=alphabet_dim)
recognizer_dim.model.load_weights(dim_model)
detector = Detector()

if detector_model:
    detector.model.load_weights(detector_model)

end_time = time.time()   
print(f"\033[1;33mLoading session took {end_time - start_time:.6f} seconds to run.\033[0m")
#endregion

#region ############ OCR Tables ###########################
process_img = img.copy()
table_results, updated_tables, process_img= tools.ocr_pipelines.ocr_tables(tables, process_img, language)

#endregion

#region ############ OCR GD&T #############################

gdt_results, updated_gdt_boxes, process_img = tools.ocr_pipelines.ocr_gdt(process_img, gdt_boxes, recognizer_gdt)

#endregion

#region ############ OCR Dimensions #######################
if frame:
    process_img = process_img[frame.y : frame.y + frame.h, frame.x : frame.x + frame.w]
process_img_ = process_img.copy()
dimensions, other_info, process_img, dim_tess = tools.ocr_pipelines.ocr_dimensions(process_img, detector, recognizer_dim, alphabet_dim, frame, dim_boxes, cluster_thres=20, max_img_size=1048, language=language, backg_save=False)

#endregion

#region ############# Qwen for tables #####################

qwen = False
if qwen:
    model, processor = tools.llm_tools.load_VL(model_name = "Qwen/Qwen2-VL-7B-Instruct")
    device = "cuda:1"
    query = ['Tolerance', 'material', 'Surface finish', 'weight']
    llm_tab_qwen = tools.llm_tools.llm_table(tables, llm = (model, processor), img = img, device = device, query=query)
    print(llm_tab_qwen)
#endregion

#region ########### Output ################################

mask_img = tools.output_tools.mask_img(img, updated_gdt_boxes, updated_tables, dimensions, frame, other_info)
table_results, gdt_results, dimensions, other_info = tools.output_tools.process_raw_output(output_path, table_results, gdt_results, dimensions, other_info, save=False)

#endregion

for b in tables[0]:
    infoblock_img = img[b.y : b.y + b.h, b.x : b.x + b.w][:]

infoblock_img = tools.llm_tools.convert_img(infoblock_img)
drw_img = tools.llm_tools.convert_img(process_img_)
manuf = False
quality = False

#region ########## Manufacturability ################
if manuf:
    messages = [
            {"role": "system",
                "content": [{"type": "text", "text": '''You are a specialized OCR system capable of reading mechanical drawings.'''},],
            },
            {"role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{infoblock_img}", "detail": "high"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{drw_img}", "detail": "high"}},
                            {"type": "text", "text": '''You are getting the inforamtion block of the drawing in the first image, and the views of the part in the second. 
                            I need you to tell me a PYTHON DICTIONARY with the manufacturing processes (keys) and short description (values)  that are best for this part.'''},],
            }]

    answer = tools.llm_tools.ask_gpt(messages)
    print('Manufacturing Answer: \n', answer)
#endregion

#region ######### Quality Control Check ##############
if quality:
    messages = [
            {"role": "system",
                "content": [{"type": "text", "text": '''You are a specialized OCR system capable of reading mechanical drawings.'''},],
            },
            {"role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{infoblock_img}", "detail": "high"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{drw_img}", "detail": "high"}},
                            {"type": "text", "text": '''You are getting the inforamtion block of the drawing in the first image, and the views of the part in the second. 
                            I need you to tell me IN A PYTHON LIST ONLY WHICH MEASUREMENTS -NUMERICAL VALUE AND TOLERANCE-  needs to be checked in the quality control process'''},],
            }]

    answer = tools.llm_tools.ask_gpt(messages)
    print('Quality Control Answer: \n', answer)
#endregion

###################################################
#cv2.imwrite('liu.png', mask_img)
#cv2.imshow('Mask Image', mask_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()