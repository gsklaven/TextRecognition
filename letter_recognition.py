import numpy as np
import cv2
import matplotlib.pyplot as plt


def create_templates(chars="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    templates = {}
    for c in chars:
        img = np.zeros((30, 30), dtype=np.uint8)
        cv2.putText(img, c, (2, 26), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
        _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        templates[c] = img_bin
    return templates


def preprocess_letter(letter_img):
    # Pad to square
    h, w = letter_img.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = letter_img

    # Resize and threshold
    resized = cv2.resize(square, (30, 30), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    return binary


def recognize_letter(letter_img, templates):
    letter_img = preprocess_letter(letter_img)
    best_char = '?'
    best_score = float('inf')
    for char, tmpl in templates.items():
        score = cv2.matchTemplate(letter_img, tmpl, cv2.TM_SQDIFF)[0][0]
        if score < best_score:
            best_score = score
            best_char = char
    return best_char
