from PIL import Image
import numpy as np
import cv2

def resize_image(image, target_size=(640, 480)):
    image.thumbnail(target_size, Image.ANTIALIAS)
    return image

def convert_to_grayscale(image):
    return image.convert('L')

def main():   
    input_image_path = '/content/pre_image_folder' 
    image = Image.open(input_image_path)
    resized_image = resize_image(image, target_size=(640, 480))
    grayscale_image = convert_to_grayscale(resized_image)
    preprocessed_image_path = '/content/processed_image_folder'
    grayscale_image.save(preprocessed_image_path)

if __name__ == "__main__":
    main()
