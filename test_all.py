import json, os, glob, cv2, string, time, csv, math
import numpy as np
from edocr2 import tools
from pdf2image import convert_from_path


def process_json_labels(folder_path):
    def order_points_clockwise(points):
        # Find the point with the lowest x value (and lowest y if tied)
        start_point = min(points, key=lambda p: (p[0], p[1]))

        # Calculate the centroid of the polygon (for angle sorting)
        centroid_x = sum(p[0] for p in points) / len(points)
        centroid_y = sum(p[1] for p in points) / len(points)

        # Function to calculate the angle from the start point to the point
        def angle_from_start(point):
            # Calculate the angle using atan2 to get the angle relative to the start point
            return math.atan2(point[1] - start_point[1], point[0] - start_point[0])

        # Sort points based on the angle relative to the start point
        # Exclude the start point from the sorting
        sorted_points = sorted((p for p in points if p != start_point), key=angle_from_start)

        # Construct the final ordered list starting with the lowest x point
        ordered_points = [start_point] + sorted_points

        return ordered_points

    def convert_json_to_coordinates(json_data):
        converted_data = []

        for shape in json_data['shapes']:
            if shape['shape_type'] == 'rectangle':
                label = shape['label']
                # Extract the top-left and bottom-right points
                (x1, y1), (x2, y2) = shape['points']
                
                # Calculate the top-right and bottom-left points
                x3, y3 = x2, y1  # top-right
                x4, y4 = x1, y2  # bottom-left

                # Append the formatted output: x1, y1, x2, y2, x3, y3, x4, y4, label
                converted_data.append([x1, y1, x3, y3, x2, y2, x4, y4, label])
            elif shape['shape_type'] == 'polygon':
                label = shape['label']
                # Extract all points in the polygon
                points = shape['points']
                # Order them in clockwise direction
                ordered_points = order_points_clockwise(points)

                # Flatten the ordered points and append the label
                flattened_points = [coord for point in ordered_points for coord in point]
                converted_data.append(flattened_points + [label])

        return converted_data

    json_files = glob.glob(os.path.join(folder_path, '*.json'))

    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            converted_data = convert_json_to_coordinates(json_data)
        output_csv_file = os.path.splitext(file)[0] + '.csv'

        # Write the converted data to the respective text file
        with open(output_csv_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write each converted row into the CSV
            for item in converted_data:
                item_int = [int(x) if isinstance(x, float) else x for x in item]
                csvwriter.writerow(item_int)

def compute_metrics(filename, predictions, mask_img, iou_thres = 0.2):
    correct_detections = 0
    detection_iou_scores = []
    total_chars, correct_chars = 0, 0
    cum_gt, cum_pred = '', ''

    with open(filename, newline='') as csvfile:
        ground_truth = []
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            # Convert coordinates to int and append label
            ground_truth.append((row[8], np.array([[row[0], row[1]], [row[2], row[3]], [row[4], row[5]], [row[6], row[7]]])))

    for gt in ground_truth:
        best_pred = None
        best_iou = 0
        for pred in predictions:
            iou = tools.train_tools.calculate_iou(pred[1], gt[1])
            if iou > best_iou:
                best_iou = iou
                best_pred = pred
        
        if best_iou >= iou_thres:
            #Detection
            correct_detections += 1
            detection_iou_scores.append(best_iou)
            #Recognition
            if gt[0] != 'other_info' and best_pred[0] != 'other_info':
                label = gt[0]
                recog = best_pred[0]
                cum_gt += label
                cum_pred += recog
                correct_char = tools.train_tools.compare_characters(label, recog)
                total_chars += len(label)
                correct_chars += correct_char

    precision = correct_detections / len(predictions) if len(predictions) > 0 else 0
    recall = correct_detections / len(ground_truth) if len(ground_truth) > 0 else 0
    average_iou = np.mean(detection_iou_scores)
    dim_metrics = {'precision':precision, "recall": recall, 'IoU': average_iou}
    char_recall = (correct_chars / total_chars) * 100 if total_chars > 0 else 0
    cer = tools.train_tools.get_cer(cum_pred, cum_gt)
    recog_metrics = {'char_recall': char_recall, 'CER': cer}
    for gt in ground_truth:
        box = gt[1]
        pts=np.array([(box[0]),(box[1]),(box[2]),(box[3])]).astype(np.int64)
        mask_img = cv2.polylines(mask_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return dim_metrics, recog_metrics, mask_img

folder_path = 'tests/test_samples/'
#process_json_labels(folder_path)

#region Alphabet definition #################
GDT_symbols = '⏤⏥○⌭⌒⌓⏊∠⫽⌯⌖◎↗⌰'
FCF_symbols = 'ⒺⒻⓁⓂⓅⓈⓉⓊ'
Extra = '(),.+-±:/°"⌀'

alphabet_gdts = string.digits + ',.⌀ABCD' + GDT_symbols + FCF_symbols
alphabet_dimensions = string.digits + 'AaBCDRGHhMmnx' + Extra
language = 'eng'
#endregion

#region Set Session ##############################
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

for file in os.listdir(folder_path):
    if file.endswith(".jpg") or file.endswith(".pdf") or file.endswith(".PDF"):
        #Loading drawing
        if file.endswith('.pdf') or file.endswith(".PDF"):
            img = convert_from_path(os.path.join(folder_path, file))
            img = np.array(img[0])
        else:
            img = cv2.imread(os.path.join(folder_path, file))

        filename = os.path.splitext(os.path.basename(file))[0]
        
        #Segmentation
        img_boxes, frame, gdt_boxes, tables  = tools.layer_segm.segment_img(img, autoframe = True, frame_thres=0.7, GDT_thres = 0.02, binary_thres=127)
        
        #Tables
        table_results, updated_tables, process_img= tools.ocr_pipelines.ocr_tables(tables, img, language)
        
        #G&DTs
        gdt_results, updated_gdt_boxes, process_img = tools.ocr_pipelines.ocr_gdt(process_img, gdt_boxes, recognizer_gdt)
        
        #Dimensions
        process_img = process_img[frame.y : frame.y + frame.h, frame.x : frame.x + frame.w]
        dimensions, other_info, process_img, dim_pyt = tools.ocr_pipelines.ocr_dimensions(process_img, detector, recognizer_dim, alphabet_dimensions, cluster_thres=20, max_img_size=1024, language=language, backg_save=False)
        
        #Masking
        mask_img = tools.output_tools.mask_img(img, updated_gdt_boxes, tables, dimensions, frame, other_info)
        
        #Postprocessing for metric computation
        if frame:
            offset = (frame.x, frame.y)
        else:
            offset = (0, 0)
        update_dimensions = []
        for dim in dimensions:
            box = dim[1]
            pts=np.array([(box[0]+offset),(box[1]+offset),(box[2]+offset),(box[3]+offset)])
            update_dimensions.append([dim[0], pts])
        for dim in other_info:
            box = dim[1]
            pts=np.array([(box[0]+offset),(box[1]+offset),(box[2]+offset),(box[3]+offset)])
            update_dimensions.append(['other_info', pts])      
        
        #Metrics computation
        dim_metrics, recog_metrics, mask_img = compute_metrics(os.path.join(folder_path, filename + '.csv'), update_dimensions, mask_img)
        print(dim_metrics)
        print(recog_metrics)

        #Display
        cv2.imshow('boxes', mask_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
