"""
Automatically detect rotation and line spacing of an image of text using
Radon transform
If image is rotated by the inverse of the output, the lines will be
horizontal (though they may be upside-down depending on the original image)
It doesn't work with black borders
"""

from skimage.transform import radon
from PIL import Image
from numpy import asarray, mean, array, blackman
import numpy as np
from numpy.fft import rfft

try:
    # More accurate peak finding from
    # https://gist.github.com/endolith/255291#file-parabolic-py
    from parabolic import parabolic

    def argmax(x):
        return parabolic(x, np.argmax(x))[0]
except ImportError:
    from numpy import argmax


def rms_flat(a):
    """
    Return the root mean square of all the elements of *a*, flattened out.
    """
    return np.sqrt(np.mean(np.abs(a) ** 2))


filename = 'VVImages/9.png'

# Load file, converting to grayscale
I = asarray(Image.open(filename))
I = I - mean(I)  # Demean; make the brightness extend above and below zero

# Do the radon transform and display the result
sinogram = radon(I)
# Find the RMS value of each row and find "busiest" rotation,
# where the transform is lined up perfectly with the alternating dark
# text and white lines
r = array([rms_flat(line) for line in sinogram.transpose()])
rotation = argmax(r)
print('Rotation: {:.2f} degrees'.format(90 - rotation))

# Plot the busy row
row = sinogram[:, rotation]
N = len(row)

# Take spectrum of busy row and find line spacing
window = blackman(N)
spectrum = rfft(row * window)

frequency = argmax(abs(spectrum))
line_spacing = N / frequency  # pixels
print('Line spacing: {:.2f} pixels'.format(line_spacing))
