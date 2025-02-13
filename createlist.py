import os

gt_txt = "masklist.txt"

# # Generate list.txt
# image_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".png")])
# with open(list_txt_path, "w") as list_file:
#     for image_file in image_files:
#         # Repeat each file path 6 times
#         file_path = os.path.join("trialDS/dataset", image_file)
#         for _ in range(6):
#             list_file.write(file_path + "\n")

# # Generate masklist.txt
# with open(masklist_txt_path, "w") as masklist_file:
#     subfolders = sorted(os.listdir(mask_dir))
#     for subfolder in subfolders:
#         subfolder_path = os.path.join(mask_dir, subfolder)
#         if os.path.isdir(subfolder_path):  # Check if it's a directory
#             mask_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith(".png")])
#             for mask_file in mask_files:
#                 # Create relative path for the mask file
#                 relative_path = os.path.join("trialDS/gt", subfolder, mask_file)
#                 masklist_file.write(relative_path + "\n")

# print(f"Generated '{list_txt_path}' and '{masklist_txt_path}'.")

# with open(gt_txt, 'r') as f:
#     # import ipdb;ipdb.set_trace()
#     lines = f.readlines()

# odd_lines = lines[::2]
# even_lines = lines[1::2]

# lines_combined = list(zip(even_lines, odd_lines))
# print(lines_combined[:2])

# with open('new_gt.txt', 'w') as f:
#     for line in lines_combined:
#         f.write(line[0])
#         f.write(line[1])


import torch

# Load the checkpoint
z_checkpoint = torch.load("00exp/z_0000_[24,15,0]_5.pth", map_location="cpu")

# Print keys if it's a dictionary
if isinstance(z_checkpoint, dict):
    print("Keys in z.pth:", z_checkpoint.keys())
else:
    print("Type of z.pth:", type(z_checkpoint))
z_tensor = z_checkpoint["z"]  # Replace "z" with the actual key if different
z_numpy = z_tensor.cpu().numpy()  # Convert to NumPy
