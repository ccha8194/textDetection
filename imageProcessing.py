import cv2
import numpy as np

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

def main():
    input_image_path = 'input_image.jpg'
    input_mask_path = 'input_mask.png'
    image = cv2.imread(input_image_path)
    mask = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)

    target_size = (640, 480)
    image = resize_image(image, target_size)
    mask = resize_image(mask, target_size)

    grayscale_image = convert_to_grayscale(image)

    enhanced_image = enhance_contrast(grayscale_image)

    normalized_image = normalize_image(enhanced_image)

    num_classes = 3
    one_hot_mask = preprocess_mask(mask, num_classes)
    

if __name__ == "__main__":
    main()

