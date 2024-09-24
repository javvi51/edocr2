import cv2, math, os
import numpy as np

###################### Tables Pipeline #################################
def ocr_table_cv2(image_cv2, languague = None):
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
    if languague:
        custom_config = f'--psm 6 -l {languague}'
    else:
        custom_config = r'--psm 6'

    # Perform OCR and get bounding box details
    ocr_data = pytesseract.image_to_data(img_rgb, config=custom_config, output_type=pytesseract.Output.DICT)

    # Prepare result: text with their positions
    result = []
    for i in range(len(ocr_data['text'])):
        if ocr_data['text'][i].strip():  # If text is not empty
            text_info = {
                'text': ocr_data['text'][i],
                'left': ocr_data['left'][i],
                'top': ocr_data['top'][i],
                'width': ocr_data['width'][i],
                'height': ocr_data['height'][i]
            }
            result.append(text_info)

    return result

def ocr_tables(tables, process_img, languague = None):
    results = []
    updated_tables = []
    for table in tables:
        for b in table:
            img = process_img[b.y : b.y + b.h, b.x : b.x + b.w][:]
            result = ocr_table_cv2(img, languague)
            if result == []:
                continue
            else:
                results.append(result)
                updated_tables.append(table)
    return results, updated_tables

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
    
    return results, updated_gdts

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
    def __init__(self, detector, recognizer, scale = 2, max_size = 2048):
        self.scale = scale
        self.detector = detector
        self.recognizer = recognizer
        self.max_size = max_size

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

    def recognize_dimensions(self, box_groups, img):
        predictions=[]
        recognition_kwargs={}
        for box in box_groups:
            rect = cv2.minAreaRect(box)
            angle = get_box_angle(box)
            angle = adjust_angle(angle)
            w=int(max(rect[1]) + 5)
            h=int(min(rect[1]) + 2)
            img_croped = subimage(img, rect[0], angle, w, h)  
            img_croped,thresh=clean_h_lines(img_croped)
            cnts = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] #Get contourns
            
            if len(cnts)==1:
                #pred=self.recognizer.recognize(image=cv2.rotate(img_croped,cv2.ROTATE_90_COUNTERCLOCKWISE))
                img_croped=cv2.rotate(img_croped,cv2.ROTATE_90_COUNTERCLOCKWISE)
                box_groups=[np.array([[[0,0],[h,0],[h,w],[0,w]]])]
                pred=self.recognizer.recognize_from_boxes(images=[img_croped],box_groups=box_groups,**recognition_kwargs)[0][0]
                predictions.append((pred, box))
            elif 1<len(cnts)<20:
                arr=check_tolerances(img_croped)
                pred=''
                for img_ in arr:
                    h,w,_=img_.shape
                    box_groups=[np.array([[[0,0],[w,0],[w,h],[0,h]]])]
                    pred_ = self.recognizer.recognize_from_boxes(images=[img_],box_groups=box_groups,**recognition_kwargs)[0][0]
                    if pred_=='':
                        pred=self.recognizer.recognize(image=img_croped)+' '
                        break
                    else:
                        pred+=pred_+' '
                pred=pred[:-1]
                predictions.append((pred, box))
        return predictions

    def ocr_img_patches(self, img, patches, ol = 0.05, cluster_t = 20):

        '''
        This functions split the original images into patches and send it to the text detector. 
        Groupes the predictions and recognize the text.
        Input: img
        patches : number of patches in both axis
        ol: overlap between patches
        cluster_t: threshold for grouping
        '''

        a_x = (1 - ol) / (patches[0]) # % of img covered in a patch (horizontal stride)
        b_x = a_x + ol # Size of horizontal patch in % of img
        a_y = (1 - ol) / (patches[1]) # % of img covered in a patch (vertical stride)
        b_y = a_y + ol # Size of horizontal patch in % of img
        box_groups = []

        for i in range(0, patches[0]):
            for j in range(0, patches[1]):
                offset = (int(a_x * i * img.shape[1]), int(a_y * j * img.shape[0]))
                patch_boundary = (int((i * a_x + b_x) * img.shape[1]), int((j * a_y + b_y) * img.shape[0]))
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
        snippets = self.recognize_dimensions(np.int32(new_group), np.array(img))
        return snippets

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
    x, y = (int( center[0] - width/2  ),int( center[1] - height/2 ))
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
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(img_croped.shape[1]),1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(img_croped, [c], -1, (255,255,255), 2)
    return img_croped, thresh

def ocr_dimensions(img, detector, recognizer, cluster_thres = 20, patches = (5, 4), backg_save = False):
    
    pipeline = Pipeline(recognizer=recognizer, detector=detector)
    results = pipeline.ocr_img_patches(img, patches, 0.05, cluster_thres)

    #For patches background generation in synthetic data training
    if backg_save:
        backg_path = os.path.join(os.getcwd(), 'edocr2/tools/backgrounds')
        os.makedirs(backg_path, exist_ok=True)
        i = 0
        for root_dir, cur_dir, files in os.walk(backg_path):
            i += len(files)

        process_img = img.copy()
        for dim in results:
            box = dim[1]
            pts=np.array([(box[0]),(box[1]),(box[2]),(box[3])])
            cv2.fillPoly(process_img, [pts], (255, 255, 255))
        # Save the image
        image_filename = os.path.join(backg_path , f'backg_{i + 1}.png')
        cv2.imwrite(image_filename, process_img)
        
    return results
