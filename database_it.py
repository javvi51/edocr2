from edocr2 import tools
import string, time, os, numpy as np, pandas as pd
import tensorflow as tf
import ocr_it
#The goal is to create a database for query search. It should look like this:
# ID | Drw Name | Annotation Value | Annotation Type | Pos x | Pos y | Recognizer  | Annotation Description |
# 1  | 'draw_1' | '⌀15'            | 'Dimension'     | 534   | 350   | 'recog_dim' | ''                     |
# 2  | 'draw_1' | 'Steel'          | 'Table'         | 1200  | 1450  | 'QuenVL'    | "Material"             |

###### We want to use all tensorflow utilities, then move to pytorch for the LLM tools. ######

#region Session Loading
start_time = time.time()

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from edocr2.keras_ocr.recognition import Recognizer
from edocr2.keras_ocr.detection import Detector

gdt_model = 'edocr2/models/recognizer_gdts.keras'
recognizer_gdt = Recognizer(alphabet=tools.ocr_pipelines.read_alphabet(gdt_model))
recognizer_gdt.model.load_weights(gdt_model)

dim_model = 'edocr2/models/recognizer_dimensions_2.keras'
alphabet_dim = tools.ocr_pipelines.read_alphabet(dim_model)
recognizer_dim = Recognizer(alphabet=alphabet_dim)
recognizer_dim.model.load_weights(dim_model)

detector = Detector()
#detector.model.load_weights('edocr2/models/detector_12_46.keras')

#Warming up models:
dummy_image = np.zeros((1, 1, 3), dtype=np.float32)
_ = recognizer_gdt.recognize(dummy_image)
_ = recognizer_dim.recognize(dummy_image)
dummy_image = np.zeros((32, 32, 3), dtype=np.float32)
_ = detector.detect([dummy_image])
end_time = time.time()

folder_path = 'tests/Washers'
file_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename))]
times =[]
results = {}
#endregion

###### #First we set the general kwargs for segmentation and ocr. #########
kwargs = {
    #General
    'binary_thres': 127, #Pixel value (0-255) to detect contourns, i.e, identify rectangles in the image
    'language': 'eng', #Language of the drawing, require installation of tesseract speficic language if not english
    'autoframe' : True, #Do we want to spot a frame as the maximum rectangle?
    'frame_thres': 0.75, #Frame boundary in % of img, if autoframe, this setting is overruled
    #GD&T
    'recognizer_gdt': recognizer_gdt, #MUST: A Tuple with (gdt alphabet, model path)
    'GDT_thres': 0.02, #Maximum porcentage of the image area to consider a cluster of rectangles a GD&T box
    #Dimensions
    'dimension_tuple': (detector, recognizer_dim, alphabet_dim), #MUST: A Tuple with (detector, dimension alphabet, model path)
    'cluster_thres': 20, #Minimum distance in pixels between two text predictions to consider the same text box
    'max_img_size': 1024, #Max size after applying scale for the img patch, bigger, better prediction, but more computationally expensive
    #Output
    'backg_save': False, #Option to save the background once all text and boxes have been removed, for synth training purposes
    'output_path': 'shit/', #Output path
    'save_mask': True, #Option to save the mask output
    'save_raw_output': False, #Option to save raw ouput, i.e, OCR text and box position,
    }

######### Now for every image we compute segmentation and ocr in dimensions ###########
ocr_results_df = pd.DataFrame(columns=[ "Drw Name", "Annotation Value", "Annotation Type", "Pos x", "Pos y", "Recognizer", "Annotation Description"])
drw_table_pair = {}
for file_path in file_paths:
    filename = os.path.splitext(os.path.basename(file_path))[0]
    kwargs['file_path'] = file_path
    results, times, tables, img = ocr_it.ocr_drawing(**kwargs)
    drw_table_pair[filename] = (tables, img)
    for dim in results['dim']:
        new_row = pd.DataFrame({
            "Drw Name": [filename],
            "Annotation Value": [dim[0]],
            "Annotation Type": ['Dimension'],
            "Pos x": [dim[1][0]],
            "Pos y": [dim[1][1]],
            "Recognizer": ['recog_dim'],
            "Annotation Description": ['']
        })
        # Append the information to the DataFrame
        ocr_results_df = pd.concat([ocr_results_df, new_row], ignore_index=True)
ocr_results_df.to_csv(os.path.join(kwargs['output_path'], 'dataframe.csv'), index=False)

###### Now we complete with table information #######################
query = ['name', 'material', 'part number', 'finishing']
model, processor = tools.llm_tools.load_llm(model_name = "Qwen/Qwen2-VL-7B-Instruct")
device = "cuda:1"
for filename, (tables, img) in drw_table_pair.items():
    llm_result = tools.ocr_pipelines.llm_table(tables, llm = (model, processor, device, query), img = img)
    for key, _ in llm_result.items():
        new_row = pd.DataFrame({
                "Drw Name": [filename],
                "Annotation Value": [llm_result[key]],
                "Annotation Type": ['Table'],
                "Pos x": [''],
                "Pos y": [''],
                "Recognizer": ['Qwen_VL'],
                "Annotation Description": [key]
            })
        ocr_results_df = pd.concat([ocr_results_df, new_row], ignore_index=True)

ocr_results_df.to_csv(os.path.join(kwargs['output_path'], 'dataframe.csv'), index=False)
