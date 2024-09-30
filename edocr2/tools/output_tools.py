import numpy as np
import cv2, csv, os

def mask_box(mask_img, points, color):
    mask = np.ones_like(mask_img) * 255
    cv2.fillPoly(mask, [points], color)
    img_with_overlay = np.int64(mask_img) * mask # <- use int64 terms
    max_px_val = np.amax(img_with_overlay) # <-- Max pixel alue
    img_with_overlay = np.uint8((img_with_overlay/max_px_val) * 255) # <- normalize and convert back to uint8
    return img_with_overlay

def mask_frame(mask_img, cl, color):
    color_full = np.full_like(mask_img,color)
    blend = 0.6
    img_color = cv2.addWeighted(mask_img, blend, color_full, 1-blend, 0)
    mask = np.zeros_like(mask_img)
    x1,y1,x2,y2 = cl.x,cl.y,cl.x+cl.w,cl.y+cl.h
    mask = cv2.rectangle(mask, (x1, y1), (x2, y2), (255,255,255), -1)
    result = np.where(mask==0, img_color, mask_img)
    return result

def mask_img(img, gdt_boxes, tables, dimensions, frame):
    mask_img=img.copy()
    for table in tables:
        for tab in table:
            pts = np.array([(tab.x, tab.y), (tab.x+tab.w, tab.y), (tab.x+tab.w, tab.y+tab.h),(tab.x,tab.y+tab.h)], np.int32)
            mask_img = mask_box(mask_img, pts, (180, 220, 250))
    
    for gdt in gdt_boxes:
        for g in gdt.values():
            for tab in g:
                pts = np.array([(tab.x, tab.y), (tab.x+tab.w, tab.y), (tab.x+tab.w, tab.y+tab.h),(tab.x,tab.y+tab.h)], np.int32)
                mask_img = mask_box(mask_img, pts, (94, 204, 243))

    if frame:
        mask_img = mask_frame(mask_img, frame, (167, 234, 82))
        offset = (frame.x, frame.y)
    else:
        offset = (0, 0)

    for dim in dimensions:
        box = dim[1]
        pts=np.array([(box[0]+offset),(box[1]+offset),(box[2]+offset),(box[3]+offset)])
        mask_img = mask_box(mask_img, pts, (93, 206, 175))
   
    return mask_img

def process_raw_output(output_path, table_results = None, gdt_results = None, dimension_results = None, save = False):
    if save:
        os.makedirs(output_path, exist_ok=True)
    #Write Table Results
    if table_results:

        def group_lines_by_top(ocr_data, tolerance=10):
            """Groups words into lines by their top coordinate, using a tolerance."""
            lines = []
            ocr_data = sorted(ocr_data, key=lambda x: x['top'])  # Sort by top coordinate

            current_line = []
            current_top = ocr_data[0]['top'] if ocr_data else None

            for entry in ocr_data:
                # If the text is within tolerance of the current line, add to current line
                if abs(entry['top'] - current_top) <= tolerance:
                    current_line.append(entry)
                else:
                    # Sort current line by left value (left-to-right) and append to lines
                    lines.append(sorted(current_line, key=lambda x: x['left']))
                    current_line = [entry]  # Start new line
                    current_top = entry['top']
            
            if current_line:
                lines.append(sorted(current_line, key=lambda x: x['left']))  # Add last line

            return lines

            # Group the OCR data into lines
        
        if save:
            csv_file = os.path.join(output_path, 'table_results.csv')

            for t in range(len(table_results)):
                grouped_lines = group_lines_by_top(table_results[t])
                # Save to CSV
                with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([f'TABLE_{t}'])
                    
                    for line in grouped_lines:
                        # Write each text instance in a separate cell
                        writer.writerow([entry['text'] for entry in line])
                    writer.writerow([''])

        new_table_results =[]
        for table in table_results:
            tab_results = []
            for item in table:
                tab_results.append([item['text'], (item['left'], item['top'])])
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
                writer.writerow(["Text", "Center X Coordinate", "Center Y Coordinate"])
                # Write the data
                for i in new_dim_results:
                    writer.writerow([i[0], i[1][0], i[1][1]])

    return table_results, gdt_results, dimension_results


