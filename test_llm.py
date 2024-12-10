from edocr2 import tools
import cv2, os

language = 'eng'
folder_path = 'tests/test_samples/'
output = 'llm_result.txt'
imgs = []
llm_dim_gpt, llm_dim_qwen = None, None
qwen, edocr_gpt, raw_gpt = False, True, False

#region ############ PreVL process ####################
for file in os.listdir(folder_path):
    if file.endswith(".jpg") or file.endswith(".png"):
        img = cv2.imread(os.path.join(folder_path, file))
        
        #Segmentation
        img_boxes, frame, gdt_boxes, tables, dim_boxes  = tools.layer_segm.segment_img(img, autoframe = True, frame_thres=0.7, GDT_thres = 0.02, binary_thres=127)
        process_img = img.copy()

        #Tables
        table_results, updated_tables, process_img= tools.ocr_pipelines.ocr_tables(tables, process_img, language)
        
        #GD&Ts
        recognizer_gdt = None
        if gdt_boxes:
            from edocr2.keras_ocr.recognition import Recognizer
            gdt_model = 'edocr2/models/recognizer_gdts.keras'
            recognizer_gdt = Recognizer(alphabet=tools.ocr_pipelines.read_alphabet(gdt_model))
            recognizer_gdt.model.load_weights(gdt_model)
        gdt_results, updated_gdt_boxes, process_img = tools.ocr_pipelines.ocr_gdt(process_img, gdt_boxes, recognizer_gdt)

        if frame:
            process_img = process_img[frame.y : frame.y + frame.h, frame.x : frame.x + frame.w]      
        imgs.append(process_img)
#endregion
       
for i in range(len(imgs)):
    file = os.listdir(folder_path)[i]
    filename = os.path.splitext(os.path.basename(file))[0]
    process_img = imgs[i]
    #region ########### Qwen ###################################
    if qwen:
        device = "cuda:1"
        model, processor = tools.llm_tools.load_VL(model_name = "Qwen/Qwen2-VL-7B-Instruct")
        llm_dim_qwen = tools.llm_tools.llm_dim((model, processor), process_img, device, scale = 0.33)
        with open(output, 'a') as file:
            file.write(f'{filename} Qwen dimensions:\n')
            file.write("  ".join(str(item) for item in llm_dim_qwen))
            file.write('\n')
    #endregion

    #region ############ GPT4 #################################
    if edocr_gpt:
        llm_dim_gpt = tools.llm_tools.gpt4_dim(process_img)
        with open(output, 'a') as file:
            file.write(f'{filename} GPT dimensions:\n')
            file.write("; ".join(str(item) for item in llm_dim_gpt))
            file.write('\n')

    #endregion

#region RawGPT ########################################
if raw_gpt:
    for file in os.listdir(folder_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            img = cv2.imread(os.path.join(folder_path, file))
            filename = os.path.splitext(os.path.basename(file))[0]
            llm_dim_gpt = tools.llm_tools.gpt4_dim(img)
            
            with open(output, 'a') as file:
                file.write(f'{filename} Raw GPT dimensions:\n')
                file.write("; ".join(str(item) for item in llm_dim_gpt))
                file.write('\n')
#endregion