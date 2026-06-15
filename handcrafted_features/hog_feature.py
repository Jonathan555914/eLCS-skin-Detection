
import numpy as np
import cv2
from skimage.feature import hog
from skimage import data, exposure
import matplotlib.pyplot as plt
from pathlib import Path

def extract_hog_features(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9, visualize=True):
    """
   Extract HOG (Histogram of Oriented Gradients) features from an image

    
    Parameters:
        image: Input image
        pixels_per_cell: Size of each cell in pixels
        cells_per_block: Number of cells in each block
        orientations: Number of gradient orientation bins
        visualize: Whether to return HOG visualization
    
    Returns:
        hog_features: HOG feature vector
        hog_image: HOG visualization image (if visualize=True)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    if visualize:
        hog_features, hog_image = hog(gray, orientations=orientations,
                                      pixels_per_cell=pixels_per_cell,
                                      cells_per_block=cells_per_block,
                                      visualize=True,
                                      feature_vector=True)
        return hog_features, hog_image
    else:
        hog_features = hog(gray, orientations=orientations,
                           pixels_per_cell=pixels_per_cell,
                           cells_per_block=cells_per_block,
                           visualize=False,
                           feature_vector=True)
        return hog_features

def manual_gradient_and_orientation(image):
    """
    Manual calculation of gradients and orientations (for educational purposes)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    gray = gray.astype(np.float32)
    
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
    
    return gx, gy, magnitude, orientation

def demo_hog():
    """
    English: Demo function to show HOG feature extraction

    """
    image = np.zeros((128, 64), dtype=np.uint8)
    
    for i in range(0, 128, 16):
        image[i:i+8, :] = 255
    
    noise = np.random.randint(0, 30, (128, 64), dtype=np.uint8)
    image = cv2.add(image, noise)
    
    hog_features, hog_image = extract_hog_features(image)
    
    gx, gy, magnitude, orientation = manual_gradient_and_orientation(image)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gx, cmap='gray')
    axes[0, 1].set_title('Gradient X (gx)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(gy, cmap='gray')
    axes[0, 2].set_title('Gradient Y (gy)')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(magnitude, cmap='gray')
    axes[1, 0].set_title('Gradient Magnitude')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(orientation, cmap='hsv')
    axes[1, 1].set_title('Gradient Orientation')
    axes[1, 1].axis('off')
    
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    axes[1, 2].imshow(hog_image_rescaled, cmap='gray')
    axes[1, 2].set_title('HOG Visualization')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_path = Path(__file__).resolve().parent / 'hog_demo.png'
    plt.savefig(output_path)
    
    print('HOG demo completed! Image saved as hog_demo.png')
    print(f' HOG feature vector length: {len(hog_features)}')
if __name__ == '__main__':
    demo_hog()
