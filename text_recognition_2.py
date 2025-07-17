import numpy as np


def connected_components(img) -> tuple:
    """
    This function implements the connected components algorithm to label connected regions in a binary image.
    """

    h, w = img.shape
    labels = np.zeros((h, w), dtype=int)
    parent = {}
    next_label = 1

    for y in range(h):
        for x in range(w):
            if img[y, x] == 0:
                continue
            else:
                neighbors = []
                if y > 0 and labels[y - 1, x] > 0:
                    neighbors.append(labels[y - 1, x])
                if x > 0 and labels[y, x - 1] > 0:
                    neighbors.append(labels[y, x - 1])
                if not neighbors:
                    labels[y, x] = next_label
                    parent[next_label] = next_label
                    next_label += 1
                else:
                    min_label = min(neighbors)
                    labels[y, x] = min_label
                    for n in neighbors:
                        if n != min_label:
                            parent[n] = min_label
                            union(min_label, n, parent)
    for y in range(h):
        for x in range(w):
            if labels[y, x] > 0:
                labels[y, x] = find(labels[y, x], parent)

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    return labels, len(unique_labels)


def find(label, parent):
    if label != parent[label]:
        parent[label] = find(parent[label], parent)
    return parent[label]


def union(label1, label2, parent):
    root1 = find(label1, parent)
    root2 = find(label2, parent)
    if root1 != root2:
        parent[root2] = root1