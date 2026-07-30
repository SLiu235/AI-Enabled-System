import cv2
import numpy as np

class Augmentation:
    def __init__(self):
        self.image = None 

    def horizontal_flip(self, image):
        flipped_image = cv2.flip(image, 1)
        return flipped_image

    def gaussian_blur(self, image, kernel_size=(7, 7)):
        blurred_image = cv2.GaussianBlur(image, kernel_size, 0.5)
        return blurred_image

    def resize(self, image, scale_percent=50):
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized_image

    def rotate(self, image, angle=45):
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated_image

    def adjust_brightness(self, image, value=30):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        final_hsv = cv2.merge((h, s, v))
        bright_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return bright_image

    def add_noise(self, image, mean=0, var=0.1):
        row, col, ch = image.shape
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy_image = image + gauss
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image

    def augment(self, images):
        augmented_images = []
        
        for image in images:
            augmentations = [
                self.horizontal_flip(image),
                self.gaussian_blur(image),
                self.resize(image),
                self.rotate(image),
                self.adjust_brightness(image),
                self.add_noise(image)
            ]
            augmented_images.extend(augmentations)  
        
        return augmented_images


if __name__ == "__main__":
    image = cv2.imread('../logistics/-_10_jpg.rf.7216b8fd23d002354ad6cdd0d7691cd5.jpg')
    
    augmenter = Augmentation()

    # Apply augmentations
    # flipped_image = augmenter.horizontal_flip(image)
    # blurred_image = augmenter.gaussian_blur(image)
    # resized_image = augmenter.resize(image, scale_percent=75)
    # rotated_image = augmenter.rotate(image, angle=90)
    # bright_image = augmenter.adjust_brightness(image, value=50)
    # noisy_image = augmenter.add_noise(image)
