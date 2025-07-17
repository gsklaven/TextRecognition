import numpy as np


def dfs_algorithm(img) -> tuple:
    """
    This function implements the Depth-First Search (DFS) algorithm.
    """
    h, w = img.shape
    labels = np.zeros((h, w), dtype=int)
    label = 1

    for y in range(h):
        for x in range(w):
            if img[y, x] == 1 and labels[y, x] == 0:

                stack = [(y, x)]
                labels[y, x] = label

                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= nx < w and 0 <= ny < h and img[ny, nx] == 1 and labels[ny, nx] == 0:
                            labels[ny, nx] = label
                            stack.append((ny, nx))
                label += 1

    return labels, label - 1


def bounding_box(labels: np.ndarray) -> np.ndarray:
    """
    This function calculates the bounding box for each labeled region.
    """
    boxes = []

    for label in range(1, np.max(labels) + 1):
        ys, xs = np.where(labels == label)
        if ys.size == 0 or xs.size == 0:
            continue

        min_y, max_y = ys.min(), ys.max()
        min_x, max_x = xs.min(), xs.max()

        boxes.append([label, min_y, min_x, max_y, max_x])

    return np.array(boxes)


def sort_boxes(boxes, line_spacing=10):
    """
    This function sorts the bounding boxes into lines based on their vertical position.
    """
    boxes = sorted(boxes, key=lambda b: b[1])
    lines = []
    current_line = []

    for box in boxes:
        label, min_y, min_x, max_y, max_x = box
        if not current_line:
            current_line.append(box)
        else:
            prev_y = current_line[-1][1]
            if abs(min_y - prev_y) <= line_spacing:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]

    if current_line:
        lines.append(current_line)

    for i in range(len(lines)):
        lines[i] = sorted(lines[i], key=lambda b: b[2])  # sort by min_x

    sorted_boxes = [box for line in lines for box in line]
    return np.array(sorted_boxes)


def extract_letters(img: np.ndarray, boxes: np.ndarray, pad=3) -> list:
    """
    This function extracts letters from the image based on the bounding boxes.
    """
    letters = []
    h, w = img.shape
    for box in boxes:
        label, min_y, min_x, max_y, max_x = box
        min_y_p = max(min_y - pad, 0)
        min_x_p = max(min_x - pad, 0)
        max_y_p = min(max_y + pad, h - 1)
        max_x_p = min(max_x + pad, w - 1)
        letter_img = img[min_y_p:max_y_p + 1, min_x_p:max_x_p + 1]
        letters.append(letter_img)
    return letters