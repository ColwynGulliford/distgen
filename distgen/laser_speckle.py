import numpy as np
from numpy.fft import fft2, fftshift, ifft2

def generate_laser_speckle_fft(image_size, speckle_size, random_seed=None):
    """
    Generate a laser speckle pattern using FFT.

    Parameters:
    - image_size: Tuple of the form (height, width) for the output image size.
    - speckle_size: Controls the frequency content of the speckle pattern,
                        affecting the speckle size. Larger values result in larger speckles.

    Returns:
    - speckle_pattern: Generated laser speckle pattern as a numpy array.
    """

    # Generate a random phase pattern in Fourier domain

    if(random_seed is not None):
        pass
    
    random_phase = np.exp(2j * np.pi * np.random.rand(image_size[0], image_size[1]))

    # Apply a low-pass filter to control the speckle size
    Y, X = np.ogrid[:image_size[0], :image_size[1]]
    center = (image_size[0] // 2, image_size[1] // 2)
    distance = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    
    frequency_cutoff = speckle_size

    #print(speckle_size, frequency_cutoff)
    
    low_pass_filter = distance <= frequency_cutoff

    filtered_phase = random_phase * low_pass_filter

    # Perform an Inverse FFT to get the speckle pattern in spatial domain
    speckle_field = ifft2(filtered_phase)
    speckle_pattern = np.abs(speckle_field)

    # Normalize the speckle pattern for visualization
    speckle_pattern = (speckle_pattern - speckle_pattern.min()) / (speckle_pattern.max() - speckle_pattern.min()) * 255

    return speckle_pattern

def generate_speckle_pattern_with_filter(image_size, sigma, random_seed=None):
    """
    Generate a laser speckle pattern using FFT with a Gaussian filter to control speckle size.

    Parameters:
    - image_size: Tuple indicating the size of the output image (height, width).
    - sigma: Standard deviation of the Gaussian filter, controlling the speckle size.
             Smaller sigma leads to larger speckles.

    Returns:
    - speckle_pattern: The generated speckle pattern as a 2D numpy array.
    """
    # Generate a random phase distribution
    if(random_seed):
        np.random.seed(random_seed)
    
    random_phase = np.exp(2j * np.pi * np.random.rand(image_size[0], image_size[1]))
    
    # Perform FFT of the random phase distribution
    fft_image = fft2(random_phase)
    
    # Generate a Gaussian filter
    x = np.linspace(-image_size[1]//2, image_size[1]//2, image_size[1])
    y = np.linspace(-image_size[0]//2, image_size[0]//2, image_size[0])
    X, Y = np.meshgrid(x, y)
    #print('sigma', sigma)
    gaussian_filter = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    
    # Apply the Gaussian filter in the Fourier domain
    fft_image_filtered = fftshift(fft_image) * gaussian_filter
    
    # Inverse FFT to get the speckle pattern
    speckle_field = ifft2(fft_image_filtered)
    speckle_pattern = np.abs(speckle_field)
    
    # Normalize the pattern for visualization
    speckle_pattern_normalized = (speckle_pattern - np.min(speckle_pattern)) / (np.max(speckle_pattern) - np.min(speckle_pattern))
    
    return speckle_pattern_normalized