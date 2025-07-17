import cv2
import math
import os
import matplotlib.pyplot as plt
from text_recognition import dfs_algorithm
from text_recognition import bounding_box
from text_recognition import extract_letters, extract_letters_with_padding
from text_recognition import sort_boxes
from text_recognition_2 import connected_components

plt.ioff()
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

img = cv2.imread('testing_images/1_SCoWMbVi-AxLOuFWJThJOA.png', cv2.IMREAD_GRAYSCALE)
blurred = cv2.GaussianBlur(img, (3, 3), 0)
binary_img = cv2.adaptiveThreshold(
    img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
)

# # Αφαίρεση θορύβου (morphological opening)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
# binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
#
# # Προαιρετικά dilation για ένωση κομματιών
# binary_img = cv2.dilate(binary_img, kernel, iterations=1)
#
# img = cv2.imread('testing_images/1_SCoWMbVi-AxLOuFWJThJOA.png', cv2.IMREAD_GRAYSCALE)
# blurred = cv2.GaussianBlur(img, (5,5), 0)
#
# binary_img = cv2.adaptiveThreshold(
#     blurred, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 8
# )
#
# # Αφαίρεση θορύβου (morphological opening)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
# binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
#
# # Προαιρετικά dilation για ένωση κομματιών
# binary_img = cv2.dilate(binary_img, kernel, iterations=1)


# --- DFS Algorithm ---
dfs_labels, dfs_count = dfs_algorithm(binary_img)
print("DFS - Number of regions:", dfs_count)
dfs_boxes = bounding_box(dfs_labels)
print("DFS - Bounding boxes:")
for box in dfs_boxes:
    print(f"Label {box[0]}: min_y={box[1]}, min_x={box[2]}, max_y={box[3]}, max_x={box[4]}")
dfs_sorted_boxes = sort_boxes(dfs_boxes)
dfs_letters = extract_letters_with_padding(binary_img, dfs_sorted_boxes)

# --- Connected Components ---
cc_labels, cc_count = connected_components(binary_img)
print("Connected Components - Number of regions:", cc_count)
cc_boxes = bounding_box(cc_labels)
print("Connected Components - Bounding boxes:")
for box in cc_boxes:
    print(f"Label {box[0]}: min_y={box[1]}, min_x={box[2]}, max_y={box[3]}, max_x={box[4]}")
cc_sorted_boxes = sort_boxes(cc_boxes)
cc_letters = extract_letters_with_padding(binary_img, cc_sorted_boxes)

# --- Visualization ---
cols = 10
dfs_rows = math.ceil(len(dfs_letters) / cols)
cc_rows = math.ceil(len(cc_letters) / cols)

plt.figure(figsize=(cols*2, (dfs_rows + cc_rows)*2))
for i, letter in enumerate(dfs_letters):
    plt.subplot(dfs_rows + cc_rows, cols, i+1)
    plt.imshow(letter, cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.title('DFS Letters')
for i, letter in enumerate(cc_letters):
    plt.subplot(dfs_rows + cc_rows, cols, dfs_rows*cols + i+1)
    plt.imshow(letter, cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.title('CC Letters')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"text_recognition_results3.png"))
