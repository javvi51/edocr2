import numpy as np
import cv2

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