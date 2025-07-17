import cv2
import os
from letter_recognition import (
    load_templates,
    save_templates,
    interactive_training,
    letters_to_text_interactive
)
from text_recognition_2 import connected_components
from text_recognition import (
    bounding_box,
    sort_boxes,
    extract_letters_with_padding
)

output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

# === Load and preprocess image ===
img = cv2.imread('testing_images/', cv2.IMREAD_GRAYSCALE)
blurred = cv2.GaussianBlur(img, (5, 5), 0)

binary_img = cv2.adaptiveThreshold(
    blurred, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 8
)

# Morphological filtering για καθαρά γράμματα
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
binary_img = cv2.dilate(binary_img, kernel, iterations=1)

# === Connected components ===
labels, count = connected_components(binary_img)
print(f"Number of connected components: {count}")

# === Sort and extract letters ===
boxes = bounding_box(labels)
sorted_boxes = sort_boxes(boxes)

# ✨ Χρησιμοποιούμε padding όπως στο demo
letters = extract_letters_with_padding(binary_img, sorted_boxes, pad=3)
print(f"Extracted {len(letters)} letters")

# === Load or train templates ===
template_path = os.path.join(output_dir, "templates.pkl")
templates = load_templates(template_path)

if not templates:
    print("Δεν βρέθηκαν templates. Ξεκινάμε εκπαίδευση...")
    new_templates = interactive_training(letters)
    templates.update(new_templates)
    save_templates(templates, template_path)

# === Recognition with interactive labeling για νέα/άγνωστα γράμματα ===
recognized_text = letters_to_text_interactive(
    sorted_boxes, letters, templates, save_path=template_path
)

print("\n===== RECOGNIZED TEXT =====\n")
print(recognized_text)
