import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt

def preprocess_letter(letter_img, size=(28, 28)):
    h, w = letter_img.shape
    scale = min(size[0] / h, size[1] / w)
    resized = cv2.resize(letter_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    new_img = np.zeros(size, dtype=np.uint8)
    y_off = (size[0] - resized.shape[0]) // 2
    x_off = (size[1] - resized.shape[1]) // 2
    new_img[y_off:y_off + resized.shape[0], x_off:x_off + resized.shape[1]] = resized
    _, binarized = cv2.threshold(new_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return 255 - binarized  # letters white on black

def mse(img1, img2):
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

def recognize_letter(letter_img, templates):
    best_score = float("inf")
    best_char = "?"
    for char, template_list in templates.items():
        for template in template_list:
            score = mse(letter_img, template)
            if score < best_score:
                best_score = score
                best_char = char
    return best_char, best_score

def interactive_training(letters):
    trained_templates = {}
    print("===== Εκπαίδευση: Πληκτρολογήστε το γράμμα και πατήστε Enter (ή '?' για παράλειψη) =====")

    for i, letter_img in enumerate(letters):
        plt.ion()
        fig, ax = plt.subplots()
        ax.imshow(letter_img, cmap='gray')
        ax.set_title(f"Letter {i + 1}")
        ax.axis('off')
        plt.show()
        plt.pause(0.001)

        char = input(f"Letter {i + 1}: Πληκτρολόγησε γράμμα ή '?' για παράλειψη: ").strip()

        plt.close(fig)

        if char == '?':
            print(f"Letter {i + 1}: Παράλειψη")
            continue
        elif len(char) == 1 and char.isalpha():
            char = char.upper()
            print(f"Letter {i + 1}: Αποθηκεύτηκε ως '{char}'")
            norm = preprocess_letter(letter_img)
            if char not in trained_templates:
                trained_templates[char] = []
            trained_templates[char].append(norm)
        else:
            print(f"Letter {i + 1}: Μη έγκυρη είσοδος. Παράλειψη.")
            continue

    return trained_templates

def save_templates(templates, path):
    with open(path, 'wb') as f:
        pickle.dump(templates, f)

def load_templates(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        return {}

def letters_to_text_interactive(sorted_boxes, letters, templates, save_path=None, mse_threshold=3000):
    """
    Αναγνωρίζει τα γράμματα, αλλά αν η διαφορά (MSE) με τα υπάρχοντα templates είναι μεγάλη,
    ζητάει να επιβεβαιώσει ή να προσθέσει νέο γράμμα (interactive).
    """
    text_lines = []
    current_line = []
    current_y = sorted_boxes[0][1]
    threshold = 10
    updated = False

    for i, (box, letter_img) in enumerate(zip(sorted_boxes, letters)):
        y = box[1]
        if abs(y - current_y) > threshold:
            text_lines.append(''.join(current_line))
            current_line = []
            current_y = y

        norm = preprocess_letter(letter_img)
        best_char, best_score = recognize_letter(norm, templates)

        # Αν η απόσταση είναι πάνω από το όριο, ζητάμε νέα ετικέτα
        if best_score > mse_threshold:
            plt.ion()
            fig, ax = plt.subplots()
            ax.imshow(letter_img, cmap='gray')
            ax.set_title(f"Letter {i + 1} (MSE={best_score:.1f})")
            ax.axis('off')
            plt.show()
            plt.pause(0.001)

            char = input(f"Letter {i + 1} (MSE={best_score:.1f}): Πληκτρολόγησε γράμμα ή '?' για παράλειψη: ").strip()
            plt.close(fig)

            if char == '?':
                print(f"Letter {i + 1}: Παράλειψη")
                char = '?'
            elif len(char) == 1 and char.isalpha():
                char = char.upper()
                print(f"Letter {i + 1}: Αποθηκεύτηκε ως '{char}'")
                if char not in templates:
                    templates[char] = []
                templates[char].append(norm)
                updated = True
            else:
                print(f"Letter {i + 1}: Μη έγκυρη είσοδος. Παράλειψη.")
                char = '?'
            best_char = char

        current_line.append(best_char)

    if current_line:
        text_lines.append(''.join(current_line))

    if updated and save_path:
        save_templates(templates, save_path)
        print(f"Νέα templates αποθηκεύτηκαν στο {save_path}")

    return '\n'.join(text_lines)
