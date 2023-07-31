import os
import cv2
import numpy as np
import glob

def resize_image(image, target_size=(640, 480)):
    return cv2.resize(image, target_size)

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def enhance_contrast(image):
    return cv2.equalizeHist(image)

def normalize_image(image):
    return image / 255.0

def preprocess_mask(mask, num_classes):
    one_hot_mask = np.zeros((mask.shape[0], mask.shape[1], num_classes), dtype=np.uint8)
    for class_idx in range(num_classes):
        one_hot_mask[:, :, class_idx] = (mask == class_idx).astype(np.uint8)
    return one_hot_mask

def preprocess_folder(input_folder, output_folder):
    num_classes = 3
    target_size = (640, 480)

    image_files = glob.glob(os.path.join(input_folder, '*.jpg'))
    mask_files = glob.glob(os.path.join(input_folder, '*.png'))

    for image_path in image_files:
        image = cv2.imread(image_path)
        mask_path = os.path.join(input_folder, os.path.splitext(os.path.basename(image_path))[0] + '.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = resize_image(image, target_size)
        mask = resize_image(mask, target_size)

        grayscale_image = convert_to_grayscale(image)
        enhanced_image = enhance_contrast(grayscale_image)
        normalized_image = normalize_image(enhanced_image)

        one_hot_mask = preprocess_mask(mask, num_classes)
        output_image = np.concatenate((image, np.expand_dims(grayscale_image, axis=-1), np.expand_dims(enhanced_image, axis=-1)), axis=1)
        output_image = np.concatenate((output_image, np.expand_dims(normalized_image, axis=-1)), axis=1)
        for i in range(num_classes):
            output_image = np.concatenate((output_image, np.expand_dims(one_hot_mask[:, :, i], axis=-1)), axis=1)

        # Save the combined output image
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, output_image)

if __name__ == "__main__":
    input_folder = 'input_folder'
    output_folder = 'model_input' # <-- A little confusing, input for the model not this file :) 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    preprocess_folder(input_folder, output_folder)



