
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from pathlib import Path

def extract_lbp_features(image, radius=1, n_points=8, method='uniform'):
    """
    Extract LBP (Local Binary Pattern) features from an image
    
    
    Parameters:
        image: Input grayscale image
        radius: Radius of the LBP circle
        n_points: Number of sampling points
        method: LBP method ('uniform', 'default', 'ror', 'var')
    
    Returns:
        lbp_image: LBP transformed image
        lbp_hist: Normalized LBP histogram
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    lbp = local_binary_pattern(gray, n_points, radius, method=method)
    
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins))
    
    return lbp, lbp_hist

def manual_lbp(image):
    """
     Manual implementation of basic LBP for educational purposes
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    height, width = gray.shape
    lbp_image = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            center = gray[y, x]
            binary = 0
            
            positions = [(-1, -1), (-1, 0), (-1, 1),
                        (0, 1), (1, 1), (1, 0),
                        (1, -1), (0, -1)]
            
            for i, (dy, dx) in enumerate(positions):
                if gray[y + dy, x + dx] >= center:
                    binary |= (1 << i)
            
            lbp_image[y, x] = binary
    
    return lbp_image

def demo_lbp():
    """
    Demo function to show LBP feature extraction

    """
    image = np.zeros((100, 100), dtype=np.uint8)
    image[20:40, 20:40] = 255
    image[60:80, 60:80] = 255
    noise = np.random.randint(0, 50, (100, 100), dtype=np.uint8)
    image = cv2.add(image, noise)
    
    lbp, hist = extract_lbp_features(image)
    manual_lbp_img = manual_lbp(image)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(manual_lbp_img, cmap='gray')
    plt.title('Manual LBP')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(lbp, cmap='gray')
    plt.title('skimage LBP (Uniform)')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.bar(range(len(hist)), hist, width=0.8)
    plt.title('LBP Histogram')
    plt.xlabel('Bin')
    plt.ylabel('Normalized Frequency')
    
    plt.tight_layout()
    output_path = Path(__file__).resolve().parent / 'lbp_demo.png'
    plt.savefig(output_path)
    print('LBP demo completed! Image saved as lbp_demo.png')
    

if __name__ == '__main__':
    demo_lbp()
