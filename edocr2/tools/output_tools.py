import numpy as np
import cv2, csv, os

def mask_box(mask_img, points, color):
    mask = np.ones_like(mask_img, dtype=np.uint8) * 255
    cv2.fillPoly(mask, [points], color)

    # Apply the mask to the image
    img_with_overlay = cv2.bitwise_and(mask_img, mask)

    return img_with_overlay

def mask_frame(mask_img, cl, tables, color):
    # Blend the original image with the specified color
    blend = 0.5
    img_color = cv2.addWeighted(mask_img, blend, np.full_like(mask_img, color), 1 - blend, 0)

    # Create a mask with a white rectangle in the specified region
    mask = np.zeros_like(mask_img, dtype=np.uint8)
    x1, y1, x2, y2 = cl.x, cl.y, cl.x + cl.w, cl.y + cl.h
    cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
    for table in tables:
        for tab in table:
            pts = np.array([(tab.x, tab.y), (tab.x+tab.w, tab.y), (tab.x+tab.w, tab.y+tab.h),(tab.x,tab.y+tab.h)], np.int32)
            cv2.fillPoly(mask, [pts], (255, 255, 255))

    # Apply the mask to choose between the blended image and the original
    result = np.where(mask == 255, mask_img, img_color)

    return result

def mask_img(img, gdt_boxes, tables, dimensions, frame, other_info):
    mask_img=img.copy()
    for table in tables:
        for tab in table:
            pts = np.array([(tab.x, tab.y), (tab.x+tab.w, tab.y), (tab.x+tab.w, tab.y+tab.h),(tab.x,tab.y+tab.h)], np.int32)
            mask_img = mask_box(mask_img, pts, (212,242,247))
    
    for gdt in gdt_boxes:
        for g in gdt.values():
            for tab in g:
                pts = np.array([(tab.x, tab.y), (tab.x+tab.w, tab.y), (tab.x+tab.w, tab.y+tab.h),(tab.x,tab.y+tab.h)], np.int32)
                mask_img = mask_box(mask_img, pts, (68,136,179))

    if frame:
        mask_img = mask_frame(mask_img, frame, tables, (100*2, 78*2, 73*2))
        offset = (frame.x, frame.y)
    else:
        offset = (0, 0)

    for dim in dimensions:
        box = dim[1]
        pts=np.array([(box[0]+offset),(box[1]+offset),(box[2]+offset),(box[3]+offset)])
        mask_img = mask_box(mask_img, pts, (110,185,187))
    
    for info in other_info:
        box = info[1]
        pts=np.array([(box[0]+offset),(box[1]+offset),(box[2]+offset),(box[3]+offset)])
        mask_img = mask_box(mask_img, pts, (128,102,90))

   
    return mask_img

def process_raw_output(output_path, table_results = None, gdt_results = None, dimension_results = None, other_info = None, save = False):
    #Write Table Results
    if table_results and save:
        csv_file = os.path.join(output_path, 'table_results.csv')
        new_table_results =[]
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header    
            for t in range(len(table_results)):
                writer.writerow([f'Table_{t}'])
                writer.writerow(["Text", "X Coordinate", "Y Coordinate"])
                tab_results = []
                for item in table_results[t]:
                    line = [item['text'], (item['left'], item['top'])]
                    tab_results.append(line)
                    writer.writerow([item['text'], item['left'], item['top']])
                new_table_results.append(tab_results)
            table_results = new_table_results

    #Write GD&T Results
    if gdt_results and save:
        csv_file = os.path.join(output_path, 'gdt_results.csv')
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(["Text", "X Coordinate", "Y Coordinate"])
            # Write the data
            for item in gdt_results:
                text, coords = item
                writer.writerow([text, coords[0], coords[1]])
                
    #Write Dimension Results
    if dimension_results:
        new_dim_results = []
        for item in dimension_results:
            text, coords = item
            center = np.mean(coords, axis=0).astype(int).tolist()
            new_dim_results.append([text, center])
        dimension_results = new_dim_results
        if save:
            csv_file = os.path.join(output_path, 'dimension_results.csv')
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write the header
                writer.writerow(["Text", "X Coordinate", "Y Coordinate"])
                # Write the data
                for i in new_dim_results:
                    writer.writerow([i[0], i[1][0], i[1][1]])
    
    if other_info:
        new_info_results = []
        for item in other_info:
            text, coords = item
            center = np.mean(coords, axis=0).astype(int).tolist()
            new_info_results.append([text, center])
        other_info = new_info_results
        if save:
            csv_file = os.path.join(output_path, 'other_info.csv')
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write the header
                writer.writerow(["Text", "X Coordinate", "Y Coordinate"])
                # Write the data
                for i in new_info_results:
                    writer.writerow([i[0], i[1][0], i[1][1]])

    return table_results, gdt_results, dimension_results, other_info


