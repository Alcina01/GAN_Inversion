
import os
import random
import cv2
import numpy as np
import json
from pycocotools import mask as coco_mask
 
def load_images_and_masks(root_folder):
    data = {}
    for cls_folder in os.listdir(root_folder):
        cls_path = os.path.join(root_folder, cls_folder)
        if not os.path.isdir(cls_path):
            continue
        segments_path = os.path.join(cls_path, 'primary', 'segments')
        masks_path = os.path.join(cls_path, 'primary', 'masks')
        if os.path.exists(segments_path) and os.path.exists(masks_path):
            images = sorted(os.listdir(segments_path))
            masks = sorted(os.listdir(masks_path))
            for img, mask in zip(images, masks):
                img_path = os.path.join(segments_path, img)
                mask_path = os.path.join(masks_path, mask)
                data.setdefault(cls_folder, []).append((img_path, mask_path))
    return data
 
def random_augment(image, mask):
    # Define augmentations
    augmentations = ["flip", "rotate", "colorize"]
    
    # Randomly select a number of augmentations to apply
    applied_augs = random.sample(augmentations, random.randint(0, len(augmentations)))

    if "flip" in applied_augs:
        flip_code = random.choice([-1, 0, 1])  # -1 is both axes, 0 is vertical, 1 is horizontal
        image = cv2.flip(image, flipCode=flip_code)
        mask = cv2.flip(mask, flipCode=flip_code)

    if "rotate" in applied_augs:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        angle = random.uniform(0, 360)  # Random angle between 0 and 360 degrees
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        new_width = int(height * abs_sin + width * abs_cos)
        new_height = int(height * abs_cos + width * abs_sin)
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
        mask = cv2.warpAffine(mask, rotation_matrix, (new_width, new_height))

    if "colorize" in applied_augs:
        factor = random.uniform(0.5, 1.5)  # Random factor between 0.5 and 1.5 for color adjustment
        image = (image * factor).clip(0, 255).astype(np.uint8)  # Brightness adjustment for the image

    return image, mask

 
def calculate_pixel_iou(occluder_box, occludee_box, mask_occluder, mask_occludee, w=640, h=640):
    mask1 = np.zeros((w,h))
    mask1[occludee_box[1]:occludee_box[3], occludee_box[0]:occludee_box[2]] = mask_occludee
    mask2 = np.zeros((w,h))
    mask2[occluder_box[1]:occluder_box[3], occluder_box[0]:occluder_box[2]] = mask_occluder
    intersection_mask = np.logical_and(mask1, mask2)
    occluder_mask = mask2
    intersection = intersection_mask.sum()
    occludee_mask = np.logical_xor(mask1,intersection_mask)
    occluder_mask_sum = occluder_mask.sum()
    mask1sum = mask1.sum()
    # print(f"intersection sum {intersection},mask1 sum {mask1sum}, mask 2 sum {occluder_mask_sum}")
    occlusion_inter_pair =(intersection / mask1sum) * 100
    if mask1sum != 0: 
        occlusion = int((intersection / mask1sum) * 100) 
        # cv2.imshow("Intersection Mask", intersection_mask.astype(np.uint8) * 255) 
        # cv2.imshow("occludee mask", occludee_mask.astype(np.uint8) * 255)                       #occludee_mask
        # cv2.imshow("occluder mask", occluder_mask.astype(np.uint8) * 255)                                #occluder_mask
        # cv2.imshow("Original occludee mask", mask1.astype(np.uint8) * 255)
        # cv2.imshow("Original occluder mask", mask2.astype(np.uint8) * 255)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        occlusion = 0.0  
    if (intersection >=occluder_mask_sum):
        occlusion =100
    if occlusion_inter_pair < 0.99 and occlusion_inter_pair > 0.0:
        occlusion =100
        print("occlusion is there but not visible")
 
    return occlusion ,occludee_mask,occluder_mask ,mask1 , mask2



def paste_with_occlusion(canvas, canvas_mask, canvas_mask_bb, img1, img2, mask1, mask2, occ, occlusion_0_count, occupied_coords, pair_no):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas_h, canvas_w = canvas.shape[:2]
    canvas_region = (0, 0, canvas_w, canvas_h)
    intermediate_canvas = np.ones((canvas_w, canvas_h, 3), dtype=np.uint8) * 127
    intermediate_canvas_2 = np.ones((canvas_w, canvas_h, 3), dtype=np.uint8) * 127


    for _ in range(350):
        y1 = random.randint(0, canvas_h - h1)
        x1 = random.randint(0, canvas_w - w1)
        region1 = (x1, y1, x1 + w1, y1 + h1)
 
        mask1_resized = cv2.resize(mask1, (w1, h1), interpolation=cv2.INTER_LINEAR)[..., np.newaxis] / 255.0
        resd ,occludee_mask,occluder_mask,gt_mask_occludee,gt_mask_occluder = calculate_pixel_iou(region1, canvas_region, mask1_resized.squeeze(axis=-1), canvas_mask_bb)
        if resd == 0.0:
            for _ in range(250):
                y2 = random.randint(0, canvas_h - h2)
                x2 = random.randint(0, canvas_w - w2)
                occluder_box = (x2, y2, x2 + w2, y2 + h2)
                mask2_resized = cv2.resize(mask2, (w2, h2), interpolation=cv2.INTER_LINEAR)[..., np.newaxis] / 255.0
                res3x ,occludee_mask,occluder_mask,gt_mask_occludee,gt_mask_occluder = calculate_pixel_iou(occluder_box, canvas_region, mask2_resized.squeeze(axis=-1), canvas_mask_bb)
                if res3x == 0.0:
                    occlusion_percent ,occludee_mask,occluder_mask ,gt_mask_occludee,gt_mask_occluder= calculate_pixel_iou(occluder_box, region1,
                                                            mask2_resized.squeeze(axis=-1), mask1_resized.squeeze(axis=-1))
                    print("Occlusion Percentage", occlusion_percent)
                    if occlusion_percent == 0.0 and (pair_no == 0 or pair_no == 1):
                        continue
                        
                    if occlusion_percent <= 30:
                        intermediate_canvas[y1:y1 + h1, x1:x1 + w1] = (
                                intermediate_canvas[y1:y1 + h1, x1:x1 + w1] * (1 - mask1_resized) + img1 * mask1_resized).astype(np.uint8)   #occludee_gt
                        intermediate_canvas_2[y2:y2 + h2, x2:x2 + w2] = (
                                intermediate_canvas_2[y2:y2 + h2, x2:x2 + w2] * (1 - mask2_resized) + img2 * mask2_resized).astype(np.uint8)   #occluder_gt

                        canvas_mask[y1:y1 + h1, x1:x1 + w1] += (mask1_resized.squeeze() > 0).astype(np.uint8)
                        canvas[y1:y1 + h1, x1:x1 + w1] = (
                                canvas[y1:y1 + h1, x1:x1 + w1] * (1 - mask1_resized) + img1 * mask1_resized).astype(np.uint8)
                        # cv2.imshow("First Canvas Mask", canvas_mask.astype(np.uint8) * 255) 
                        # cv2.imshow("First Canvas Mask with BB", canvas_mask_bb.astype(np.uint8) * 255) 
                        # cv2.imshow("First Canvas",canvas)

                        canvas[y2:y2 + h2, x2:x2 + w2] = (
                                canvas[y2:y2 + h2, x2:x2 + w2] * (1 - mask2_resized) + img2 * mask2_resized).astype(np.uint8)
                        canvas_mask[y2:y2 + h2, x2:x2 + w2] += (mask2_resized.squeeze() > 0).astype(np.uint8)
                        occ.append(occlusion_percent) 

                        # Save the occupied coordinates
                        occupied_coords.append({'occluder': (x2, y2, x2 + w2, y2 + h2), 'occludee': (x1, y1, x1 + w1, y1 + h1)})  # BB for occlusion
                        # Debugging steps ----------------------
                        # cv2.imshow("Second Canvas Mask", canvas_mask.astype(np.uint8) * 255) 
                        # cv2.imshow("Second Canvas Mask with BB", canvas_mask_bb.astype(np.uint8) * 255) 
                        # cv2.imshow("Second Canvas",canvas)
                        # cv2.imshow("intermediate Canvas Mask", intermediate_canvas) 
                        # cv2.imshow("second intermediate Canvas Mask", intermediate_canvas_2) 
                        # cv2.imshow("occludee mask", occludee_mask.astype(np.uint8) * 255)                       #occludee_mask
                        # cv2.imshow("occluder mask", occluder_mask.astype(np.uint8) * 255)                                #occluder_mask
                        # cv2.waitKey()  # Wait for a key press
                        # cv2.destroyAllWindows()
                        # Debugging steps ----------------------

                        # Update canvas_mask_bb with bounding box areas
                        canvas_mask_bb[y1:y1 + h1, x1:x1 + w1] = 1  # Set the occludee area as occupied in canvas_mask_bb
                        canvas_mask_bb[y2:y2 + h2, x2:x2 + w2] = 1  # Set the occluder area as occupied in canvas_mask_bb

                        if occlusion_percent == 0.0:
                            occlusion_0_count += 1
                
                        return True,intermediate_canvas,intermediate_canvas_2 ,occludee_mask,occluder_mask,gt_mask_occluder,gt_mask_occludee
                else:
                    return False,intermediate_canvas,intermediate_canvas_2 ,occludee_mask,occluder_mask,gt_mask_occluder,gt_mask_occludee
             
            return False ,intermediate_canvas,intermediate_canvas_2 ,occludee_mask,occluder_mask,gt_mask_occluder,gt_mask_occludee
    else:
        return False,intermediate_canvas,intermediate_canvas_2 ,occludee_mask,occluder_mask,gt_mask_occluder,gt_mask_occludee


def create_dd(root_folder, canvas_size=(640, 640), iterations=1):
    data = load_images_and_masks(root_folder)
    class_list = list(data.keys())
    base_path ="/home/pinto/miniconda3/envs/DGPrior/DGP/newDS"
    coco_data = {
    "images": [],
    "categories": [
        {"id": 1, "name": "occluder", "supercategory": "stone"},
        {"id": 2, "name": "occludee", "supercategory": "stone"}
    ],"annotations": []}
    annotation_id = 1  # Unique annotation ID
 
    for iter_idx in range(iterations):
        occ = []
        neigh_class = []
        occupied_coords = []  # List to store the occupied coordinates
        canvas = np.ones((canvas_size[0], canvas_size[1], 3), dtype=np.uint8) * 127
        canvas_mask = np.zeros((canvas_size[0], canvas_size[1]), dtype=np.uint8)
        canvas_mask_bb = np.zeros_like(canvas_mask)  # Create the new mask for bounding boxes
        occlusion_0_count = 0  # Counter to track 0% occlusions
        # cvs= np.ones((canvas_size[0], canvas_size[1], 3), dtype=np.uint8) * 127
        # cvs2= np.ones((canvas_size[0], canvas_size[1], 3), dtype=np.uint8) * 127

 
        for pair_idx in range(3):
            placed = False
            retry_count = 0
            while not placed and retry_count < 250:  # Retry up to 100 times per pair
                retry_count += 1
 
                # STEP 1: Randomly pick the first instance
                class1 = random.choice(class_list)
                instance1 = random.choice(data[class1])
 
                # STEP 2: Randomly pick the second instance (neighboring class)
                class_num = int(class1.split('_')[1])
                neighbors = [cls for cls in [f'cls_{class_num - 1:02d}', f'cls_{class_num + 1:02d}'] if cls in data]
                class2 = random.choice(neighbors)
                instance2 = random.choice(data[class2])
                # Load images and masks
                img1 = cv2.imread(instance1[0])
                mask1 = cv2.imread(instance1[1], cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(instance2[0])
                mask2 = cv2.imread(instance2[1], cv2.IMREAD_GRAYSCALE)

                # Apply augmentations
                img1, mask1 = random_augment(img1, mask1)
                img2, mask2 = random_augment(img2, mask2)
 
                # Randomly choose one of the images as occluder and the other as occludee
                if random.choice([True, False]):
                    occluder, occludee, mask_occluder, mask_occludee = (img1, img2, mask1, mask2)
                else:
                    occluder, occludee, mask_occluder, mask_occludee = (img2, img1, mask2, mask1)

                # Try placing the pair on the canvas with occlusion tracking
                res,intermediate_canvas,intermediate_canvas_2 ,occludee_mask,occluder_mask,gt_mask_occluder,gt_mask_occludee= paste_with_occlusion(
                    canvas, canvas_mask, canvas_mask_bb, occludee, occluder, mask_occludee, mask_occluder, occ, occlusion_0_count, occupied_coords,
                    pair_idx)
                
                if res:
                    placed = True
                    gt_mask_occludee_uint8 = np.uint8(gt_mask_occludee * 255)
                    gt_mask_occluder_uint8 = np.uint8(gt_mask_occluder * 255)
                    occludee_mask_uint8 = np.uint8(occludee_mask * 255)
                    occluder_mask_uint8 = np.uint8(occluder_mask * 255)                                         
                    masks_path=os.path.join(base_path, f'mask/Image_{iter_idx:04d}')
                    gt_path=os.path.join(base_path, f'gt/Image_{iter_idx:04d}')
                    gt_mask_path=os.path.join(base_path, f'gt_mask/Image_{iter_idx:04d}')
                    if not os.path.exists(masks_path):
                        os.makedirs(masks_path)
                    if not os.path.exists(gt_path):
                        os.makedirs(gt_path)
                    if not os.path.exists(gt_mask_path):
                        os.makedirs(gt_mask_path)
                    output_occluder_gt_mask_path = os.path.join(base_path, f'gt_mask/Image_{iter_idx:04d}/{iter_idx:04d}_0{pair_idx + 1}_occluder_{occ[-1]}.png')
                    output_occludee_gt_mask_path = os.path.join(base_path, f'gt_mask/Image_{iter_idx:04d}/{iter_idx:04d}_0{pair_idx + 1}_occludee_{occ[-1]}.png')
                    cv2.imwrite(output_occluder_gt_mask_path, gt_mask_occluder_uint8)
                    cv2.imwrite(output_occludee_gt_mask_path, gt_mask_occludee_uint8)
                    output_occluder_mask_path = os.path.join(base_path, f'mask/Image_{iter_idx:04d}/{iter_idx:04d}_0{pair_idx + 1}_occluder_{occ[-1]}.png')
                    output_occludee_mask_path = os.path.join(base_path, f'mask/Image_{iter_idx:04d}/{iter_idx:04d}_0{pair_idx + 1}_occludee_{occ[-1]}.png')
                    cv2.imwrite(output_occluder_mask_path, occluder_mask_uint8)
                    cv2.imwrite(output_occludee_mask_path, occludee_mask_uint8)
                    output_occluder_gt_path = os.path.join(base_path, f'gt/Image_{iter_idx:04d}/{iter_idx:04d}_0{pair_idx + 1}_occluder_{occ[-1]}.png')
                    output_occludee_gt_path = os.path.join(base_path, f'gt/Image_{iter_idx:04d}/{iter_idx:04d}_0{pair_idx + 1}_occludee_{occ[-1]}.png')
                    cv2.imwrite(output_occluder_gt_path, intermediate_canvas_2)
                    cv2.imwrite(output_occludee_gt_path, intermediate_canvas)
                    bbox_occludee = cv2.boundingRect(occludee_mask_uint8)
                    bbox_occluder = cv2.boundingRect(occluder_mask_uint8)
                    # Polygon segmentation (contour extraction)
                    contours_occludee, _ = cv2.findContours(occludee_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours_occluder, _ = cv2.findContours(occluder_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    segmentation_occludee = sum([contour.flatten().tolist() for contour in contours_occludee], [])
                    segmentation_occluder = sum([contour.flatten().tolist() for contour in contours_occluder], [])

                    # cv2.rectangle(cvs, (bbox_occludee[0], bbox_occludee[1]),
                    #             (bbox_occludee[0] + bbox_occludee[2], bbox_occludee[1] + bbox_occludee[3]),
                    #             (0, 255, 0), 2)
                    # cv2.rectangle(cvs, (bbox_occluder[0], bbox_occluder[1]),
                    #             (bbox_occluder[0] + bbox_occluder[2], bbox_occluder[1] + bbox_occluder[3]),
                    #             (0, 0, 255), 2)
                    # cv2.imshow('Bounding Boxes', cvs)
                    # segmentation_occludee = np.array(segmentation_occludee, dtype=np.int32)
                    # segmentation_occluder = np.array(segmentation_occluder, dtype=np.int32)
                    # segmentation_occludee = segmentation_occludee.reshape((-1, 1, 2))
                    # segmentation_occluder = segmentation_occluder.reshape((-1, 1, 2))
                    # cv2.polylines(cvs2, [segmentation_occludee], isClosed=True, color=(0, 255, 0), thickness=2)
                    # cv2.polylines(cvs2, [segmentation_occluder], isClosed=True, color=(0, 0, 255), thickness=2)
                    # cv2.imshow('Segmentation', cvs2)
                    # cv2.waitKey()  # Wait for a key press
                    # cv2.destroyAllWindows()

                    #annotation for occluder
                    occlusion_percent = occ[-1]
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": iter_idx + 1,
                        "category_id": 1,  # Occluder
                        "segmentation": segmentation_occluder,
                        "area": cv2.contourArea(contours_occluder[0]),
                        "bbox": bbox_occluder,
                        "occlusion_percent": occlusion_percent,
                        "iscrowd": 1
                    })
                    annotation_id += 1

                    #annotation for occludee
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": iter_idx + 1,
                        "category_id": 2,  # Occludee
                        "segmentation": segmentation_occludee,
                        "area": cv2.contourArea(contours_occludee[0]),
                        "bbox": bbox_occludee,
                        "occlusion_percent": occlusion_percent,
                        "iscrowd": 1
                    })
                    annotation_id += 1

                else:
                    print(f"Retrying pair {pair_idx + 1}...")

            if not placed:
                print(f"Pair {pair_idx + 1} skipped after 100 retries.")

        # Save the results
        output_image_path = os.path.join(base_path, f'dataset/{iter_idx:04d}_['+','.join([str(elem) for elem in occ])+'].png')
        cv2.imwrite(output_image_path, canvas)
        image_filename = f'dataset/{iter_idx:04d}_{occ}.png'
        coco_data["images"].append({
            "id": iter_idx + 1,
            "width": canvas_size[0],
            "height": canvas_size[1],
            "file_name": image_filename
        })
        if iter_idx >=2199:
            for annotation in coco_data["annotations"]:
                annotation["segmentation"] = ",".join(map(str, annotation["segmentation"]))
            with open(os.path.join(base_path, f"annotations/annotations_{iter_idx:04d}.json"), 'w') as json_file:
                json.dump(coco_data, json_file, indent=4)


create_dd("single_stones_for_SD copy")