import cv2
import numpy as np

class Rect():
    def __init__(self, name, x, y ,w, h, state = 'green', parent = None, children = None):
        self.name = name
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.state = state
        self.parent = parent
        self.children = children if children is not None else []
    
    def __repr__(self):
        return f"{self.name} ({self.x}, {self.y}, {self.w}, {self.h})"

def angle(pt0,pt1,pt2):
    '''get the angle of the lines conformed by pt0-pt1 and pt0-pt2
    Args: 
        pt0: np array with x,y coordinates
        pt1: np array with x,y coordinates
        pt2: np array with x,y coordinates
    Returns:
        angle: angle conformed by the lines, in degrees
        '''
    # Vector differences
    d1 = pt1[0] - pt0[0]
    d2 = pt2[0] - pt0[0]

    # Dot product and magnitudes of the vectors
    dot_product = np.dot(d1, d2)
    norm_d1 = np.linalg.norm(d1)
    norm_d2 = np.linalg.norm(d2)

    # Calculate the angle in radians and convert to degrees
    angle_rad = np.arccos(dot_product / (norm_d1 * norm_d2))
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def is_contained(inner_rect, outer_rect):
    """Check if inner_rect is fully contained inside outer_rect."""
    return (outer_rect.x <= inner_rect.x <= outer_rect.x + outer_rect.w) and \
           (outer_rect.y <= inner_rect.y <= outer_rect.y + outer_rect.h) and \
           (outer_rect.x <= inner_rect.x + inner_rect.w <= outer_rect.x + outer_rect.w) and \
           (outer_rect.y <= inner_rect.y + inner_rect.h <= outer_rect.y + outer_rect.h)

def build_hierarchy(rectangles):
    """Organize Rect objects into a nested hierarchy based on containment."""
    # Sort rectangles by size (area) in descending order
    rectangles.sort(key=lambda r: r.w * r.h, reverse=True)

    hierarchy = []

    def add_to_hierarchy(parent, child):
        """Recursively add a child rect to the appropriate parent rect."""
        for rect in parent.children:
            if is_contained(child, rect):
                add_to_hierarchy(rect, child)
                return
        parent.children.append(child)
        child.parent = parent

    # Add the largest rectangle to the hierarchy as root
    for rect in rectangles:
        if not hierarchy:
            hierarchy.append(rect)  # First rectangle becomes the root
        else:
            # Try to insert this rect in one of the existing parents
            for root in hierarchy:
                if is_contained(rect, root):
                    add_to_hierarchy(root, rect)
                    break
            else:
                hierarchy.append(rect)  # If no parent found, it's at root level

    return hierarchy

def print_hierarchy(rectangles, level=0):
    """Recursively print the rectangle hierarchy."""
    for rect in rectangles:
        print("  " * level + f"{rect.name}")
        print_hierarchy(rect.children, level + 1)

def is_box(approx):
    if len(approx) == 4 and 88<angle(approx[1],approx[0],approx[2])<92 and 88<angle(approx[3],approx[0],approx[2])<92: #if cnt can be approx with only 4 points, it is a 4 side polygon
        dx1 = abs(approx[0][0][0] - approx[1][0][0])  # Difference in x for the first side
        dy1 = abs(approx[0][0][1] - approx[1][0][1])  # Difference in y for the first side
        dx2 = abs(approx[1][0][0] - approx[2][0][0])  # Difference in x for the second side
        dy2 = abs(approx[1][0][1] - approx[2][0][1])  # Difference in y for the second side
        
        # Set a threshold to ensure the sides are aligned to the image axes
        threshold = 10 # Allowable deviation from horizontal or vertical alignment

        if (dx1 < threshold or dy1 < threshold) and (dx2 < threshold or dy2 < threshold):
            return True
        else:
            return False
    else:
        return False

def find_rectangles(img, binary_thres = 127):
    '''Returns a list with rectangles from an image
    Args: img: the image (mechanical engineering drawing)
    returns: 
        rect_list: list of rects
        img_boxes: image with the contourns of the boxes'''
    # Convert img to grayscale and threshold to binary image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, binary_thres, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_boxes = img.copy()
    r = 0
    rect_list=[]
    for cnt in contours:
        x1,y1 = cnt[0][0] #Top-left corner
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True) #Approach cnt to polygons with multiple points
        if is_box(approx): #if cnt can be approx with only 4 points, it is a 4 side polygon
            x, y, w, h = cv2.boundingRect(cnt) #get rectangle information
            if w*h>1000: #Clean very small rectangles
                #cv2.putText(img_boxes, 'rect_'+str(r), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 127, 83), 2) #Add rectangle tag
                img_boxes = cv2.drawContours(img_boxes, [cnt], -1, (0,127,255),4) #Plot rectangle contourn
                rect_list.append(Rect('rect_'+str(r),x,y,w,h)) #Get a list of rectangles
                r = r + 1

    #print('Number of rectangles:', len(rect_list))
    hierarchy = build_hierarchy(rect_list)
    

    return img_boxes, rect_list, hierarchy

def find_frame(img, frame_thres):
    from scipy.signal import find_peaks
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 11, 61, 39)
    edges = cv2.Canny(blurred, 0, 255)
    kernel = np.ones((5, 5), np.uint8) 
    edges = cv2.dilate(edges, kernel, iterations=1 )

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,20))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))

    v_morphed = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel, iterations=2)
    v_morphed = cv2.dilate(v_morphed, None)
    h_morphed = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel, iterations=2)
    h_morphed = cv2.dilate(h_morphed, None)

    v_acc = cv2.reduce(v_morphed, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    h_acc = cv2.reduce(h_morphed, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    s_v_acc = smooth(v_acc[0,:],9) 
    s_h_acc = smooth(h_acc[:,0],9) 

    v_peaks, v_props = find_peaks(s_v_acc, 0.8*np.max(np.max(s_v_acc)))
    h_peaks, h_props = find_peaks(s_h_acc, 0.8*np.max(np.max(s_h_acc)))
    '''tmp = img.copy()
    for peak_index in v_peaks:
        cv2.line(tmp, (peak_index, 0), (peak_index, img.shape[0]), (0, 127, 255),4)
    for peak_index in h_peaks:
        cv2.line(tmp, (0, peak_index), (img.shape[1], peak_index), (255,127,0),4)
    
    cv2.imwrite('frame.png', tmp)'''
    # Ensure we have enough peaks to define a box
    if len(v_peaks) < 2 or len(h_peaks) < 2:
        return None  # Not enough peaks to define a frame

    min_frame_tup = (img.shape[0] * frame_thres, img.shape[1] * frame_thres) #(y,x)
    
    def find_best_peaks(peaks, v_h): #v_h is a int= 0 or 1 to identify if we are talking vertical or horizontal peaks
        min_peaks = np.array([peak for peak in peaks if peak < img.shape[v_h] / 2])
        max_peaks = np.array([peak for peak in peaks if peak > img.shape[v_h] / 2])
        min_frame = min_frame_tup[v_h]
        diff_matrix = np.subtract.outer(max_peaks, min_peaks)
        valid_diff_mask = (diff_matrix > min_frame)
        if np.any(valid_diff_mask):
            min_diff = np.min(diff_matrix[valid_diff_mask])
            # Find the index of the minimum difference
            min_diff_index = np.where(diff_matrix == min_diff)
            max_idx, min_idx = min_diff_index[0][0], min_diff_index[1][0]
            max_p = max_peaks[max_idx]
            min_p = min_peaks[min_idx]
            return max_p, min_p
    
    bottom, top = find_best_peaks(h_peaks, 0)
    right, left = find_best_peaks(v_peaks, 1)
    best_frame = Rect('frame', left, top, right - left, bottom - top)

    # Return the best (innermost) frame found
    return best_frame

def touching_box(cl, cl_fire, thres=1.1):
    '''returns true if cl is adjacent to cl_fire, a threshold ratio is applied to scale up cl
    Args:
        cl: rect to analyze
        cl_fire: list of rect
        thres: scale up ratio for cl
    return: boolean'''
     # Calculate scaled-up boundaries for cl
    scaled_w, scaled_h  = cl.w * thres, cl.h * thres
    cl_center_x, cl_center_y  = cl.x + cl.w / 2, cl.y + cl.h / 2

    # Get new x, y based on the center of the rectangle after scaling
    scaled_x, scaled_y = cl_center_x - scaled_w / 2, cl_center_y - scaled_h / 2

    # Check if scaled box overlaps with any in cl_fire
    for f in cl_fire:
        # Check if cl_box_ overlaps with fire_box
        if not (scaled_x + scaled_w < f.x or scaled_x > f.x + f.w or
                scaled_y + scaled_h < f.y or scaled_y > f.y + f.h):
            return True  # There is an overlap
    return False

def fire_propagation(class_list, cl_fire):
    '''Returns all boxes that are either touching cl_fire or each other, as a fire that propagates
    Args: 
        class_list: list of rect
        cl_fire: the origin of the fire
    return: 
        burnt: a list of rect in contact with cl_fire or its propagations'''
    cl_fire.state = 'fire'
    on_fire = [cl_fire]
    green = class_list
    burnt = []
    l = 0
    while len(on_fire):
        l=l+1
        for g in green:
            if touching_box(g, on_fire, 1.1) is True:
                g.state = 'fire'
        for f in on_fire:
            f.state = 'burnt'
        on_fire=[cl for cl in class_list if cl.state == 'fire']
        green=[cl for cl in class_list if cl.state == 'green']
        burnt=[cl for cl in class_list if cl.state == 'burnt']
    return burnt

def find_clusters(rect_list):
    new_list =[]
    for rect in rect_list:
        if len(rect.children) == 0:
            new_list.append(rect)

    clusters = []
    singles = []
    while len(new_list):
        rect = new_list[0]
        if rect.state == 'green':
            cluster = fire_propagation(new_list, rect)
            if len(cluster) > 1:
                min_x = min(rect.x for rect in cluster)
                min_y = min(rect.y for rect in cluster)
                max_x = max(rect.x + rect.w for rect in cluster)
                max_y = max(rect.y + rect.h for rect in cluster)

                # Calculate width and height of the covering rectangle
                width = max_x - min_x
                height = max_y - min_y

                # Create a new Rect object that covers all rectangles
                covering_rect = Rect(name='cluster_'+ str(len(clusters)), x=min_x, y=min_y, w=width, h=height)
                clusters.append({covering_rect:cluster})
            else:
                singles.append(cluster[0])
            new_list = [b for b in new_list if b not in cluster]
    return clusters, singles

def cluster_criteria(clusters, singles,  GDT_thres):
    gdt, tab, dim = [], [], []
    for cl in clusters:
        for k_cl in cl:
            if k_cl.w * k_cl.h > GDT_thres:
                tab.append(cl)
            else:
                gdt.append(cl)
    for sl in singles:
        is_contained = False
        for cl in clusters:
            # Calculate the bounding box of the entire cluster
            cluster_min_x = min(k_cl.x for k_cl in cl)
            cluster_max_x = max(k_cl.x + k_cl.w for k_cl in cl)
            cluster_min_y = min(k_cl.y for k_cl in cl)
            cluster_max_y = max(k_cl.y + k_cl.h for k_cl in cl)

            # Check if the single rect (sl) is fully contained within the cluster's bounding box
            if (cluster_min_x <= sl.x <= cluster_max_x and
                cluster_min_y <= sl.y <= cluster_max_y and
                cluster_min_x <= sl.x + sl.w <= cluster_max_x and
                cluster_min_y <= sl.y + sl.h <= cluster_max_y):
                is_contained = True
                break  # No need to check further if it’s already contained

        # If not contained in any cluster, append to dim
        if not is_contained:
            if sl.w * sl.h < GDT_thres / 4:
                dim.append(sl)
    return gdt, tab, dim

def segment_img(img, frame_thres = 0.85, autoframe = True, GDT_thres = 0.02, binary_thres = 127):
    
    #Find rectangles
    img_boxes, rect_list, _  = find_rectangles(img, binary_thres= binary_thres)
    #print_hierarchy(hierarchy)
    frame = False
    if autoframe:
        frame = find_frame(img, frame_thres)
    clusters, singles = find_clusters(rect_list)

    gdt_boxes, tables, dim_boxes = cluster_criteria(clusters, singles, GDT_thres * img.shape[0] * img.shape[1])

    if not frame:
        frame = Rect('frame', int(img.shape[1] * (1-frame_thres)/2), int(img.shape[0] * (1-frame_thres)/2), int(img.shape[1] * frame_thres), int(img.shape[0] * frame_thres))

    filtered_gdt_boxes = [
        {cluster: rects} for g in gdt_boxes
        for cluster, rects in g.items()
        if is_contained(cluster, frame)
    ]
    
    return img_boxes, frame, filtered_gdt_boxes, tables, dim_boxes