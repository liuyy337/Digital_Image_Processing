import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import convolve, Gaussian2DKernel, CustomKernel
from astropy.io import fits
from skimage.exposure import match_histograms

def my_imshow(data, ax, title=""):
    mean = np.mean(data)
    sigma = np.std(data)
    vmin = mean - 3 * sigma
    vmax = mean + 3 * sigma
    ax.imshow(data, origin='lower', cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_title(title)

def process_filters(data):
    kernel = Gaussian2DKernel(x_stddev=1) # 高斯核
    smoothed = convolve(data, kernel) # 平滑
    sharpened = data + (data - smoothed) * 1.5 # 锐化 
    # 高频增强滤波
    laplacian_matrix = np.array([
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)
    lap_kernel = CustomKernel(laplacian_matrix)
    edges = convolve(data, lap_kernel, normalize_kernel=False)
    high_freq = data + 0.5 * edges
    return smoothed, sharpened, high_freq

def main1(): # 空域平滑，锐化，高频增强滤波
    with fits.open('hw3/sdo_image.fits') as hdul:
        data = hdul[0].data.astype(np.float32)
    smoothed, sharpened, high_freq = process_filters(data)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axs = axes.flatten()
    my_imshow(data, axs[0], title="Original")
    my_imshow(smoothed, axs[1], title="Smoothed")
    my_imshow(sharpened, axs[2], title="Sharpened")
    my_imshow(high_freq, axs[3], title="High-Frequency Enhanced")
    fig.suptitle("Spatial Filtering Results", fontsize=20, fontweight='bold', y=0.96)
    # plt.savefig('hw3/output1.png', dpi=100)
    plt.show()

def main2(): # 直方图规定化
    with fits.open('hw3/sdo_image.fits') as hdul:
        data = hdul[0].data.astype(np.float32)
    smoothed, sharpened, high_freq = process_filters(data) # 空域平滑，锐化，高频增强滤波
    reference_image = data
    matched_smooth = match_histograms(smoothed, reference_image) # 将平滑图像的直方图规定化为原图的直方图
    matched_sharpen = match_histograms(sharpened, reference_image)
    matched_high_freq = match_histograms(high_freq, reference_image)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axs = axes.flatten()
    my_imshow(data, axs[0], title="Original")
    my_imshow(matched_smooth, axs[1], title="Smoothed")
    my_imshow(matched_sharpen, axs[2], title="Sharpened")
    my_imshow(matched_high_freq, axs[3], title="High-Frequency Enhanced")
    fig.suptitle("Histogram Matched Filtering Results", fontsize=20, fontweight='bold', y=0.96)
    # plt.savefig('hw3/output2.png', dpi=100)
    plt.show()

if __name__ == "__main__":
    main1()
    # main2()