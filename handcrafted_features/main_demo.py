
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lbp_feature import extract_lbp_features
from hog_feature import extract_hog_features
from dwt_feature import extract_dwt_features
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def create_sample_image():
    """
    Create a sample image for demonstration
   
    """
    image = np.zeros((128, 128), dtype=np.uint8)
    
    cv2.rectangle(image, (20, 20), (40, 60), 200, -1)
    cv2.circle(image, (80, 40), 25, 255, -1)
    cv2.line(image, (30, 80), (100, 100), 180, 5)
    
    noise = np.random.randint(0, 40, (128, 128), dtype=np.uint8)
    image = cv2.add(image, noise)
    
    return image

def main():
    """
    Main function to demonstrate all three feature extraction methods
   
    """
    print('=' * 60)
    print('Handcrafted Feature Extraction Demo')
    print('=' * 60)
    
    image = create_sample_image()
    
    print('\n[1] Extracting LBP features...')
    lbp_img, lbp_hist = extract_lbp_features(image)
    print(f'    LBP histogram length: {len(lbp_hist)}')
   
    
    print('\n[2] Extracting HOG features...')
    hog_feat, hog_img = extract_hog_features(image)
    print(f'     HOG feature vector length: {len(hog_feat)}')
    
    print('\n[3] Extracting DWT features...')
    dwt_feat, dwt_coeffs = extract_dwt_features(image, wavelet='haar', level=2)
    print(f'    DWT statistical feature length: {len(dwt_feat)}')
   
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image\n')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(lbp_img, cmap='gray')
    axes[0, 1].set_title('LBP Image\n')
    axes[0, 1].axis('off')
    
    axes[0, 2].bar(range(len(lbp_hist)), lbp_hist, width=0.8)
    axes[0, 2].set_title('LBP Histogram\n')
    axes[0, 2].set_xlabel('Bin')
    
    axes[0, 3].imshow(hog_img, cmap='gray')
    axes[0, 3].set_title('HOG Visualization\n')
    axes[0, 3].axis('off')
    
    cA = dwt_coeffs[0]
    axes[1, 0].imshow(cA, cmap='gray')
    axes[1, 0].set_title('DWT Approximation (cA)\n')
    axes[1, 0].axis('off')
    
    cH, cV, cD = dwt_coeffs[1].values()
    axes[1, 1].imshow(cH, cmap='gray')
    axes[1, 1].set_title('DWT Horizontal (cH)\n')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cV, cmap='gray')
    axes[1, 2].set_title('DWT Vertical (cV)\n')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(cD, cmap='gray')
    axes[1, 3].set_title('DWT Diagonal (cD)\n')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    output_path = Path(__file__).resolve().parent / 'all_features_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    print(f'\nCombined demo image saved to: {output_path}')
    print('\n' + '=' * 60)
    print('All features extracted successfully!')
    print('=' * 60)

if __name__ == '__main__':
    main()
