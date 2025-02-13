import numpy as np
import cv2  # Assuming OpenCV is available for image processing

import numpy as np
import cv2

def save_images(preprocessed_images, ground_truths, part_option='Part 1'):
    # Loop through the preprocessed images and save them with appropriate filenames
    for idx, (preprocessed_image, gt_image) in enumerate(zip(preprocessed_images, ground_truths)):
        # Generate filenames for each processed image
        filename = f"/home/pinto/miniconda3/envs/DGPrior/DGP/jk/processed_image_{part_option}_{idx}.png"
        gt_filename = f"/home/pinto/miniconda3/envs/DGPrior/DGP/jk/gt_image_{idx}.png"
        
        # Save the processed images and ground truth images
        cv2.imwrite(filename, preprocessed_image)
        cv2.imwrite(gt_filename, gt_image)
        print(f"Saved {filename} and {gt_filename}")



def create_image(source_image, gt_images,masks, part_option='Part 1'):
    preprocessed_images = []
    ground_truths = []


    # Ensure the masks and the source image have the same size and number of channels
    source_height, source_width = source_image.shape[:2]
    
    for i in range(len(masks)):
        print("i: ",i)
        replacement_color = (127, 127, 127) 
        processed_image=source_image
        # Resize the mask to match the source image dimensions if necessary
        mask = masks[i]
        if mask.shape[:2] != (source_height, source_width):
            mask = cv2.resize(mask, (source_width, source_height))

        # If the mask is single-channel and the source image is multi-channel, expand mask channels
        if len(mask.shape) == 2 and len(source_image.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Step 1: Part 1 - Generate pre-processed images with one block mask (others hidden)
        if part_option == 'Part 1':
            for j, other_mask in enumerate(masks):
                print("j: ",j)
                ground_truths.append(gt_images[i])

                # Resize the other_mask to match the source image size
                if other_mask.shape[:2] != (source_height, source_width):
                    other_mask = cv2.resize(other_mask, (source_width, source_height))
                if len(other_mask.shape) == 2 and len(source_image.shape) == 3:
                    other_mask = cv2.cvtColor(other_mask, cv2.COLOR_GRAY2BGR)
                # Apply all masks except the current one (mask[i]) to the image
                if j != i:
                    processed_image = cv2.bitwise_and(processed_image, cv2.bitwise_not(other_mask))
                    # cv2.imshow("mask",mask)
                    # cv2.imshow("pp image",processed_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
            # inpainted_mask = np.all(processed_image == [0, 0, 0], axis=-1)
            # processed_image[inpainted_mask] = replacement_color
            black_pixels = np.all(processed_image == [0, 0, 0], axis=-1)
            processed_image[black_pixels] = replacement_color
            preprocessed_images.append(processed_image)

            print("pp image appended")

    return preprocessed_images, ground_truths


# Example usage
source_image = cv2.imread("/home/pinto/miniconda3/envs/DGPrior/DGP/newDS/dataset/0000_[28, 11, 0].png")  # Read the input image
mask1 = cv2.imread("/home/pinto/miniconda3/envs/DGPrior/DGP/newDS/mask/Image_0000/0000_01_occludee_28.png", cv2.IMREAD_GRAYSCALE)  # Read mask 1 (binary image)
mask2 = cv2.imread("/home/pinto/miniconda3/envs/DGPrior/DGP/newDS/mask/Image_0000/0000_01_occluder_28.png", cv2.IMREAD_GRAYSCALE)  # Read mask 2 (binary image)
mask3 = cv2.imread("/home/pinto/miniconda3/envs/DGPrior/DGP/newDS/mask/Image_0000/0000_02_occludee_11.png", cv2.IMREAD_GRAYSCALE)  # Read mask 3 (binary image)
mask4 = cv2.imread("/home/pinto/miniconda3/envs/DGPrior/DGP/newDS/mask/Image_0000/0000_02_occluder_11.png", cv2.IMREAD_GRAYSCALE)  # Read mask 4 (binary image)
mask5 = cv2.imread("/home/pinto/miniconda3/envs/DGPrior/DGP/newDS/mask/Image_0000/0000_03_occludee_0.png", cv2.IMREAD_GRAYSCALE)  # Read mask 3 (binary image)
mask6 = cv2.imread("/home/pinto/miniconda3/envs/DGPrior/DGP/newDS/mask/Image_0000/0000_03_occluder_0.png", cv2.IMREAD_GRAYSCALE)  # Read mask 4 (binary image)

masks = [mask1, mask2, mask3, mask4,mask5,mask6]  # List of binary masks

gt1=cv2.imread("/home/pinto/miniconda3/envs/DGPrior/DGP/newDS/gt/Image_0000/0000_01_occludee_28.png")
gt2=cv2.imread("/home/pinto/miniconda3/envs/DGPrior/DGP/newDS/gt/Image_0000/0000_01_occluder_28.png")
gt3=cv2.imread("/home/pinto/miniconda3/envs/DGPrior/DGP/newDS/gt/Image_0000/0000_02_occludee_11.png")
gt4=cv2.imread("/home/pinto/miniconda3/envs/DGPrior/DGP/newDS/gt/Image_0000/0000_02_occluder_11.png")
gt5=cv2.imread("/home/pinto/miniconda3/envs/DGPrior/DGP/newDS/gt/Image_0000/0000_03_occludee_0.png")
gt6=cv2.imread("/home/pinto/miniconda3/envs/DGPrior/DGP/newDS/gt/Image_0000/0000_03_occluder_0.png")


gt_images= [gt1,gt2,gt3,gt4,gt5,gt6]

preprocessed_images, ground_truths = create_image(source_image, gt_images, masks, part_option='Part 1')
save_images(preprocessed_images, ground_truths, part_option='Part 1')


        # # Step 2: Part 2 - Generate pre-processed images with one block mask (others visible)
        # elif part_option == 'Part 2':
        #     # Apply only the current mask (i)
        #     processed_image = cv2.bitwise_and(source_image, mask)

        #     # Combine this mask with the next mask(s) and apply them
        #     for j in range(len(masks)):
        #         if j != i:  # Skip the current mask
        #             other_mask = masks[j]
        #             if other_mask.shape[:2] != (source_height, source_width):
        #                 other_mask = cv2.resize(other_mask, (source_width, source_height))
                    
        #             if len(other_mask.shape) == 2 and len(source_image.shape) == 3:
        #                 other_mask = cv2.cvtColor(other_mask, cv2.COLOR_GRAY2BGR)

        #             processed_image = cv2.bitwise_or(processed_image, other_mask)

        #     # Append the processed image to the list
        #     preprocessed_images.append(processed_image)
        #     # Assign the corresponding GT image
        #     ground_truths.append(gt_images[i])