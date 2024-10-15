import cv2, math, os
import numpy as np

###################### Tables and Others Pipeline #################################
def ocr_img_cv2(image_cv2, language = None, psm = 11):
    """Recognize text in an OpenCV image using pytesseract and return both text and positions.
    
    Args:
        image_cv2: OpenCV image object.
        
    Returns:
        A list of dictionaries containing recognized text and their positions (left, top, width, height).
    """
    import pytesseract
    # Convert the OpenCV image to RGB format (pytesseract expects this)
    img_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    
    # Custom configuration to recognize a more complete set of characters
    if language:
        custom_config = f'--psm {psm} -l {language}'
    else:
        custom_config = f'--psm {psm}'

    # Perform OCR and get bounding box details
    ocr_data = pytesseract.image_to_data(img_rgb, config=custom_config, output_type=pytesseract.Output.DICT)

    # Prepare result: text with their positions
    result = []
    all_text = ''
    for i in range(len(ocr_data['text'])):
        if ocr_data['text'][i].strip():  # If text is not empty
            text_info = {
                'text': ocr_data['text'][i],
                'left': ocr_data['left'][i],
                'top': ocr_data['top'][i],
                'width': ocr_data['width'][i],
                'height': ocr_data['height'][i]
            }
            all_text += ocr_data['text'][i]
            result.append(text_info)
    
    return result, all_text

def ocr_tables(tables, process_img, language = None):
    results = []
    updated_tables = []

    tables = sorted(tables, key=lambda cluster_dict: next(iter(cluster_dict)).y * 10000 + next(iter(cluster_dict)).x, reverse=True)

    for table in tables:
        for b in table:
            img = process_img[b.y : b.y + b.h, b.x : b.x + b.w][:]
            result, all_text = ocr_img_cv2(img, language)
            if result == [] or len(all_text) < 5:
                continue
            else:
                for r in result:
                    r['left'] += b.x
                    r['top'] += b.y
                results.append(result)
                updated_tables.append(table)
    for table in updated_tables:
        for b in table:
            process_img[b.y : b.y + b.h, b.x : b.x + b.w][:] = 255
    
    return results, updated_tables, process_img

def llm_table(tables, llm, img):
    from edocr2.tools.llm_tools import call_vision_infoblock
    for b in tables[0]:
        tab_img = img[b.y : b.y + b.h, b.x : b.x + b.w][:]
    llm_dict = call_vision_infoblock(tab_img, model=llm[0], processor=llm[1], device = llm[2], query= llm[3])
    return llm_dict

##################### GDT Pipeline #####################################

def img_not_empty(roi, color_thres = 100):
    # Convert the ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Check if all pixels are near black or near white
    min_val, max_val, _, _ = cv2.minMaxLoc(gray_roi)
    
    # If the difference between min and max pixel values is greater than the threshold, the box contains color
    if (max_val - min_val) < color_thres:
        return False
        
    return True

def is_not_empty(img, boxes, color_thres):
    for box in boxes:
            # Extract the region of interest (ROI) from the image
        roi = img[box.y + 2:box.y + box.h - 4, box.x + 2:box.x + box.w -4]
        
        if img_not_empty(roi, color_thres) == False:
            return False
             
    return True

def sort_gdt_boxes(boxes, y_thres = 3):
    """Sorts boxes in reading order: left-to-right, then top-to-bottom.
    
    Args:
        boxes: List of Rect objects or any object with x, y, w, h attributes.
        y_threshold: A threshold to group boxes that are on the same line (default is 10 pixels).
    
    Returns:
        A list of boxes sorted in reading order.
    """
    # Sort by the y-coordinate first (top-to-bottom)
    boxes.sort(key=lambda b: b.y)

    sorted_boxes = []
    current_line = []
    current_y = boxes[0].y

    for box in boxes:
        # If the box's y-coordinate is close to the current line's y-coordinate, add it to the same line
        if abs(box.y - current_y) <= y_thres:
            current_line.append(box)
        else:
            # Sort the current line by x-coordinate (left-to-right)
            current_line.sort(key=lambda b: b.x)
            sorted_boxes.extend(current_line)
            
            # Start a new line with the current box
            current_line = [box]
            current_y = box.y
    
    # Sort the last line and add it
    current_line.sort(key=lambda b: b.x)
    sorted_boxes.extend(current_line)
    
    return sorted_boxes

def recognize_gdt(img, block, recognizer):
    roi = img[block[0].y + 2:block[0].y + block[0].h - 4, block[0].x + 2:block[0].x + block[0].w - 4]
    pred = recognizer.recognize(image = roi)
    #cv2.imwrite(f"{0}.png", roi)

    for i in range(1, len(block)):
        new_line = block[i].y - block[i - 1].y > 5
        roi = img[block[i].y:block[i].y + block[i].h, block[i].x:block[i].x + block[i].w]
        p = recognizer.recognize(image = roi)
        #cv2.imwrite(f"{i}.png", roi)
        if new_line:
            pred += '\n' + p
        else:
            pred += '|' + p

    return pred

def ocr_gdt(img, gdt_boxes, recognizer):

    updated_gdts = []
    results = []
    if gdt_boxes:
        for block in gdt_boxes:
            for _, bl_list in block.items():
                if is_not_empty(img, bl_list, 50):
                    sorted_block = sort_gdt_boxes(bl_list, 3)
                    pred = recognize_gdt(img, sorted_block, recognizer)
                    updated_gdts.append(block)
                    results.append([pred, (sorted_block[0].x, sorted_block[0].y)])
    for gdt in updated_gdts:
        for g in gdt.values():
            for b in g:
                img[b.y - 5 : b.y + b.h + 10, b.x - 5 : b.x + b.w + 10][:] = 255
    return results, updated_gdts, img

##################### Dimension Pipeline ###############################

class Pipeline:
    """A wrapper for a combination of detector and recognizer.
    Args:
        detector: The detector to use
        recognizer: The recognizer to use
        scale: The scale factor to apply to input images
        max_size: The maximum single-side dimension of images for
            inference.
    """
    def __init__(self, detector, recognizer, alphabet_dimensions, scale = 2, max_size = 1024, language = 'eng'):
        self.scale = scale
        self.detector = detector
        self.recognizer = recognizer
        self.max_size = max_size
        self.language = language
        self.alphabet_dimensions = alphabet_dimensions

    def detect(self, img, detection_kwargs = None):
        """Run the pipeline on one or multiples images.
        Args:
            images: The images to parse (numpy array)
            detection_kwargs: Arguments to pass to the detector call
            recognition_kwargs: Arguments to pass to the recognizer call
        Returns:
            A list of lists of (text, box) tuples.
        """ 
        from edocr2.keras_ocr.tools import adjust_boxes

        if np.max((img.shape[0], img.shape[1])) < self.max_size / self.scale:
            scale = self.scale
        else:
            scale = self.max_size / np.max((img.shape[0], img.shape[1]))

        if detection_kwargs is None:
            detection_kwargs = {}
        
        new_size = (int(img.shape[1]* scale), int(img.shape[0]* scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

        box_groups = self.detector.detect(images=[img], **detection_kwargs)
        box_groups = [
            adjust_boxes(boxes=boxes, boxes_format="boxes", scale=1 / scale)
            if scale != 1
            else boxes
            for boxes, scale in zip(box_groups, [scale])
        ]
        return box_groups
    
    def ocr_the_rest(self, img):

        def sort_boxes_by_centers(boxes, y_threshold=20):
            # Sort primarily by the y_center (top-to-bottom), and secondarily by x_center (left-to-right)
            sorted_boxes = sorted(boxes, key=lambda box: (box['top'], box['left']))  # Sort by (y_center, x_center)
            final_sorted_text = ""

            current_line = []
            current_y = sorted_boxes[0]['top']  # y_center of the first box

            for box in sorted_boxes:
                if abs(box['top'] - current_y) <= y_threshold:  # If y_center is within threshold, same line
                    current_line.append(box)
                else:
                    # Sort the current line by x_center (left-to-right)
                    current_line = sorted(current_line, key=lambda b: b['left'])  # Sort by x_center
                    line_text = ' '.join([b['text'] for b in current_line])  # Join text in current line
                    final_sorted_text += line_text + '\n'  # Add the text for the line and a newline
                    
                    current_line = [box]  # Start a new line
                    current_y = box['top']

            # Sort the last line and add to final result
            current_line = sorted(current_line, key=lambda b: b['left'])
            line_text = ' '.join([b['text'] for b in current_line])
            final_sorted_text += line_text  # No newline for the last line

            return final_sorted_text
    
        results, _ = ocr_img_cv2(img, self.language)
        if results:
            text = sort_boxes_by_centers(results)
            return text
        return ''

    def dimension_criteria(self, pred):
        allowed_exceptions = set('''?—!@~;¢«#_%\&€$»[]§¥©‘"'£<*“”I|ZNOXi \n''')
        return all(char in set(self.alphabet_dimensions) or char in allowed_exceptions for char in pred)
                    
    def recognize_dimensions(self, box_groups, img):
        predictions=[]
        predictions_pyt=[]
        other_info=[]
        for box in box_groups:
            rect = cv2.minAreaRect(box)
            angle = get_box_angle(box)
            angle = adjust_angle(angle)
            w=int(max(rect[1]) + 15)
            h=int(min(rect[1]) + 5)
            img_croped = subimage(img, rect[0], angle, w, h)  
            img_croped,thresh=clean_h_lines(img_croped)
            gray = cv2.cvtColor(img_croped, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            cnts = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] #Get contourns
            
            if len(cnts)==1:
                #pred=self.recognizer.recognize(image=cv2.rotate(img_croped,cv2.ROTATE_90_COUNTERCLOCKWISE))
                img_croped=cv2.rotate(img_croped,cv2.ROTATE_90_COUNTERCLOCKWISE)
                pred = self.recognizer.recognize(image=img_croped)
                if pred.isdigit():
                    predictions.append((pred, box))
            else:
                pred_pyt = self.ocr_the_rest(img_croped)
                if self.dimension_criteria(pred_pyt) or pred_pyt == '':
                    arr=check_tolerances(img_croped)
                    pred=''
                    for img_ in arr:
                        pred_ = self.recognizer.recognize(image=img_) + ' '
                        pred += pred_
                    if len(pred) - len(arr) < 3: #Error handling for tolerance detection
                        pred = self.recognizer.recognize(image=img_croped) + ' '
                    if any (char.isdigit() for char in pred):
                        predictions.append((pred[:-1], box))
                    else:
                        other_info.append((pred[:-1], box))
                    predictions_pyt.append((pred_pyt, box))
                else:
                    other_info.append((pred_pyt, box))
        return predictions, other_info, predictions_pyt

    def ocr_img_patches(self, img, ol = 0.05, cluster_t = 20):

        '''
        This functions split the original images into patches and send it to the text detector. 
        Groupes the predictions and recognize the text.
        Input: img
        patches : number of patches in both axis
        ol: overlap between patches
        cluster_t: threshold for grouping
        '''
        patches = (int(img.shape[1] / self.max_size + 2), int(img.shape[0] / self.max_size + 2))

        a_x = int((1 - ol) / (patches[0]) * img.shape[1]) # % of img covered in a patch (horizontal stride)
        b_x = a_x + int(ol* img.shape[1]) # Size of horizontal patch in % of img
        a_y = int((1 - ol) / (patches[1]) * img.shape[0]) # % of img covered in a patch (vertical stride)
        b_y = a_y + int(ol * img.shape[0]) # Size of horizontal patch in % of img
        box_groups = []

        for i in range(0, patches[0]):
            for j in range(0, patches[1]):
                offset = (a_x * i, a_y * j)
                patch_boundary = (i * a_x + b_x, j * a_y + b_y)
                img_patch = img[offset[1] : patch_boundary[1], 
                                offset[0] : patch_boundary[0]]
                if img_not_empty(img_patch, 100):
                    box_group=self.detect(img_patch)
                    for b in box_group:
                        for xy in b:
                            xy = xy + offset
                            box_groups.append(xy)
                
        box_groups = group_polygons_by_proximity(box_groups, eps = cluster_t)
        new_group = [box for box in box_groups]
        print('Detection finished. Performing recognition...')
        dimensions, other_info, dimensions_pyt = self.recognize_dimensions(np.int32(new_group), np.array(img))
        return dimensions, other_info, dimensions_pyt

def group_polygons_by_proximity(polygons, eps=50):
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union

        def polygon_intersects_or_close(p1, p2, eps):
            
            """
            Check if two polygons either intersect or are within the distance threshold `eps`.
            """
            # Create Polygon objects from the arrays
            poly1 = Polygon(p1)
            poly2 = Polygon(p2)
            
            # Check if the polygons intersect
            if poly1.intersects(poly2):
                return True
            
            # If not, check the minimum distance between their boundaries
            return poly1.distance(poly2) <= eps

        n = len(polygons)
        parent = list(range(n))  # Union-find structure to track connected components
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                parent[rootX] = rootY
        
        # Compare all polygon pairs
        for i in range(n):
            for j in range(i + 1, n):
                if polygon_intersects_or_close(polygons[i], polygons[j], eps):
                    union(i, j)
        
        # Group polygons by connected components and merge them
        grouped_polygons = {}
        for i in range(n):
            root = find(i)
            if root not in grouped_polygons:
                grouped_polygons[root] = []
            grouped_polygons[root].append(polygons[i])
        
        # Now merge the polygons in each group
        merged_polygons = []
        for group in grouped_polygons.values():
            # Collect all points from the polygons in this group
            all_points = []
            for polygon in group:
                all_points.extend(polygon)
            
            # Use Shapely to create a merged polygon
            merged_polygon = unary_union([Polygon(p) for p in group])
            
            # Convert to coordinates for OpenCV to find the min-area bounding box
            if isinstance(merged_polygon, MultiPolygon):
                merged_polygon = unary_union(merged_polygon)
            if merged_polygon.is_empty:
                continue

            # Find the minimum rotated bounding box for the merged polygon
            min_rotated_box = merged_polygon.minimum_rotated_rectangle.exterior.coords[0:4]
            
            # Add the resulting rotated box to the list
            merged_polygons.append(min_rotated_box)
        
        return merged_polygons

def check_tolerances(img):
    img_arr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Convert img to grayscale
    flag=False
    tole = False
    ## Find top and bottom line
    for i in range(0, img_arr.shape[0] - 1): # find top line
        for j in range(0,img_arr.shape[1] - 1):
            if img_arr[i, j] < 200:
                top_line = i
                flag = True
                break
        if flag == True:
            flag = False
            break
    for i in range(img_arr.shape[0] - 1, top_line, -1): # find bottom line
        for j in range(0, img_arr.shape[1] - 1):
            if img_arr[i, j] < 200:
                bot_line = i
                flag = True
                break
        if flag == True:
            break        
    ##Measure distance from right end backwards until it finds a black pixel from top line to bottom line
    stop_at = []
    for i in range(top_line, bot_line):
        for j in range(img_arr.shape[1] -1, 0, -1):
            if img_arr[i,j] < 200:
                stop_at.append(img_arr.shape[1] - j)
                break
        else:
            stop_at.append(img_arr.shape[1])
    ##Is there a normalized distance (l) relatively big with respect the others?
    for d in stop_at[int(0.3 * len(stop_at)): int(0.7 * len(stop_at))]:
        if d > img_arr.shape[0] * 0.8:
            tole = True
            tole_h_cut = stop_at.index(d) + top_line + 1
            break
        else:
            tole = False

    #If yes -> Find last character from the measurement (no tolerance)
    if tole == True:
        if d < img_arr.shape[1]: #handle error
            tole_v_cut = None
            for j in range(img_arr.shape[1] - d, img_arr.shape[1]):
                    if np.all(img_arr[int(0.3 * img_arr.shape[0]): int(0.7 * img_arr.shape[0]), j] > 200):
                        tole_v_cut=j+2
                        break
            #-> crop images
            if tole_v_cut: #handle error
                try:
                    measu_box = img_arr[:, :tole_v_cut]
                    up_tole_box = img_arr[:tole_h_cut, tole_v_cut:]
                    bot_tole_box = img_arr[tole_h_cut:, tole_v_cut:]
                    return [cv2.cvtColor(measu_box, cv2.COLOR_GRAY2BGR), cv2.cvtColor(up_tole_box, cv2.COLOR_GRAY2BGR), cv2.cvtColor(bot_tole_box, cv2.COLOR_GRAY2BGR)]
                except:
                    return [img]  
        else:
            up_text=img_arr[:tole_h_cut, :]
            bot_text=img_arr[tole_h_cut:, :]
            return [cv2.cvtColor(up_text, cv2.COLOR_GRAY2BGR), cv2.cvtColor(bot_text, cv2.COLOR_GRAY2BGR)] 
    return [img]

def get_box_angle(box):
    exp_box = np.vstack((box[3], box, box[0]))
    i = np.argmax(box[:, 1])
    B = box[i]
    A = exp_box[i]
    C = exp_box[i + 2]
    AB_ = math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
    BC_ = math.sqrt((C[0] - B[0]) ** 2+(C[1] - B[1])** 2)
    m = np.array([(A, AB_), (C, BC_)], dtype = object)
    j = np.argmax(m[:, 1])
    O = m[j, 0]
    if B[0] == O[0]:
        alfa = math.pi / 2
    else:
        alfa = math.atan((O[1] - B[1]) / (O[0] - B[0]))
    if alfa == 0:
        return alfa / math.pi * 180
    elif B[0] < O[0]:
        return - alfa / math.pi * 180
    else:
        return (math.pi - alfa) / math.pi * 180
    
def adjust_angle(alfa, i = 5):
    if -i < alfa < 90 - i:
        return - round(alfa / i)*i
    elif 90 - i < alfa < 90 + i:
        return round(alfa / i) * i - 180
    elif 90 + i < alfa < 180 + i:
        return 180 - round(alfa / i) * i
    else:
        return alfa

def subimage(image, center, theta, width, height):
    ''' 
    Rotates OpenCV image around center with angle theta (in deg)
    then crops the image according to width and height.
    '''
    shape = ( image.shape[1], image.shape[0] ) # cv2.warpAffine expects shape in (length, height)
    matrix = cv2.getRotationMatrix2D( center=center, angle=theta, scale=1 )
    image = cv2.warpAffine( src=image, M=matrix, dsize=shape )
    x, y = (int( center[0] - width/2 ),int( center[1] - height/2 ))
    x2, y2 = x + width, y + height

    if x < 0: x=0
    if x2 > shape[0]: x2 = shape[0]
    if y < 0: y=0
    if y2 > shape[1]: y2 = shape[1]

    image = image[ y:y2, x:x2 ]
    
    return image

def clean_h_lines(img_croped):
    gray = cv2.cvtColor(img_croped, cv2.COLOR_BGR2GRAY) #Convert img to grayscale
    _,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV) #Threshold to binary image
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(img_croped.shape[1]*0.9)-15,1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        img_croped = cv2.drawContours(img_croped, [c], -1, (255,255,255), 2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,int(img_croped.shape[1]*0.9)-5))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        img_croped = cv2.drawContours(img_croped, [c], -1, (255,255,255), 3)
    return img_croped, thresh

def ocr_dimensions(img, detector, recognizer, alphabet_dim, cluster_thres = 20, language = 'eng', max_img_size = 2048, backg_save = False):
    
    pipeline = Pipeline(recognizer=recognizer, detector=detector, alphabet_dimensions=alphabet_dim, max_size= max_img_size, language=language)
    dimensions, other_info, dim_pyt = pipeline.ocr_img_patches(img, 0.05, cluster_thres)

    process_img = img.copy()
    for dim in dimensions:
        box = dim[1]
        pts=np.array([(box[0]),(box[1]),(box[2]),(box[3])])
        cv2.fillPoly(process_img, [pts], (255, 255, 255))
    
    for dim in other_info:
        box = dim[1]
        pts=np.array([(box[0]),(box[1]),(box[2]),(box[3])])
        cv2.fillPoly(process_img, [pts], (255, 255, 255))
    
    # patches background generation for synthetic data training
    # Save the image
    if backg_save:
        
        backg_path = os.path.join(os.getcwd(), 'edocr2/tools/backgrounds')
        os.makedirs(backg_path, exist_ok=True)
        i = 0
        for root_dir, cur_dir, files in os.walk(backg_path):
            i += len(files)
        image_filename = os.path.join(backg_path , f'backg_{i + 1}.png')
        cv2.imwrite(image_filename, process_img)
        
    return dimensions, other_info, process_img, dim_pyt
