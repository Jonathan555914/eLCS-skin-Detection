
import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt
from pathlib import Path

def extract_dwt_features(image, wavelet='haar', level=2, feature_type='stats'):
    """
    Extract DWT (Discrete Wavelet Transform) features from an image

    
    Parameters:
        image: Input image
        wavelet: Wavelet type ('haar', 'db1', 'db2', 'sym2', etc.)
        level: Decomposition level
        feature_type: 'stats' for statistical features, 'full' for full coefficients
    
    Returns:
        features: Extracted DWT features
        coeffs: Wavelet coefficients (for visualization)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    coeffs = pywt.wavedec2(gray, wavelet=wavelet, level=level)
    
    if feature_type == 'stats':
        features = []
        cA = coeffs[0]
        features.extend([np.mean(cA), np.std(cA), np.max(cA), np.min(cA)])
        
        for i in range(1, len(coeffs)):
            cH, cV, cD = coeffs[i]
            for coeff in [cH, cV, cD]:
                features.extend([np.mean(coeff), np.std(coeff), np.max(coeff), np.min(coeff)])
        
        return np.array(features), coeffs
    else:
        flat_coeffs = []
        flat_coeffs.append(coeffs[0].flatten())
        for i in range(1, len(coeffs)):
            cH, cV, cD = coeffs[i]
            flat_coeffs.extend([cH.flatten(), cV.flatten(), cD.flatten()])
        
        return np.concatenate(flat_coeffs), coeffs

def dwt_decomposition_visualization(image, wavelet='haar', level=2):
    """
    Create a visualization of DWT decomposition
   
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    coeffs = pywt.wavedec2(gray, wavelet=wavelet, level=level)
    
    return coeffs

def plot_dwt_coefficients(coeffs, level):
    """
    Plot DWT coefficients
    
    """
    fig, axes = plt.subplots(level + 1, 3, figsize=(15, 5 * (level + 1)))
    
    if level == 1:
        axes = axes.reshape(1, -1)
    
    cA = coeffs[0]
    axes[0, 0].imshow(cA, cmap='gray')
    axes[0, 0].set_title('Approximation (cA)')
    axes[0, 0].axis('off')
    
    for i in range(1, level + 1):
        cH, cV, cD = coeffs[i]
        
        axes[i, 0].imshow(cH, cmap='gray')
        axes[i, 0].set_title(f'Horizontal Detail (cH) - Level {i}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(cV, cmap='gray')
        axes[i, 1].set_title(f'Vertical Detail (cV) - Level {i}')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(cD, cmap='gray')
        axes[i, 2].set_title(f'Diagonal Detail (cD) - Level {i}')
        axes[i, 2].axis('off')
    
    for j in range(1, 3):
        axes[0, j].axis('off')
    
    plt.tight_layout()
    return fig

def demo_dwt():
    """
    Demo function to show DWT feature extraction

    """
    image = np.zeros((128, 128), dtype=np.uint8)
    
    x = np.linspace(0, 4*np.pi, 128)
    y = np.linspace(0, 4*np.pi, 128)
    X, Y = np.meshgrid(x, y)
    image = (np.sin(X) * np.cos(Y) * 127 + 128).astype(np.uint8)
    
    circle = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(circle, (64, 64), 30, 255, -1)
    image = cv2.addWeighted(image, 0.5, circle, 0.5, 0)
    
    wavelet = 'haar'
    level = 2
    
    features, coeffs = extract_dwt_features(image, wavelet=wavelet, level=level)
    
    fig = plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    coeffs2 = pywt.wavedec2(image, wavelet=wavelet, level=level)
    img_array, _ = pywt.coeffs_to_array(coeffs2)
    
    plt.subplot(1, 3, 2)
    plt.imshow(img_array, cmap='gray')
    plt.title(f'DWT Decomposition ({wavelet}, level {level})')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    feature_names = ['cA_mean', 'cA_std', 'cA_max', 'cA_min']
    for i in range(1, level + 1):
        feature_names.extend([f'cH{i}_mean', f'cH{i}_std', f'cH{i}_max', f'cH{i}_min'])
        feature_names.extend([f'cV{i}_mean', f'cV{i}_std', f'cV{i}_max', f'cV{i}_min'])
        feature_names.extend([f'cD{i}_mean', f'cD{i}_std', f'cD{i}_max', f'cD{i}_min'])
    
    plt.bar(range(len(features)), features, color='steelblue')
    plt.title('Statistical Features')
    plt.xlabel('Feature Index')
    plt.ylabel('Value')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = Path(__file__).resolve().parent / 'dwt_demo.png'
    plt.savefig(output_path)
    
    print('DWT demo completed! Image saved as dwt_demo.png')
    print(f'Number of statistical features extracted: {len(features)}')


if __name__ == '__main__':
    demo_dwt()
