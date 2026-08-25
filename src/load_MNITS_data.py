import numpy as np
import matplotlib.pyplot as plt
import struct

# ---------------------------
#  function reading images
# ---------------------------
def read_images(filename):
    with open(filename, 'rb') as f:
        _, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)
    return data

# ---------------------------
#  function reading labels
# ---------------------------
def read_labels(filename):
    with open(filename, 'rb') as f:
        _, num_items = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# ---------------------------
#  Read Data
# ---------------------------
# train_images = read_images("./Data/t10k-images-idx3-ubyte/train-images.idx3-ubyte")
# train_labels = read_labels("./Data/t10k-images-idx3-ubyte/train-labels.idx1-ubyte")
# test_images  = read_images("./Data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte")
# test_labels  = read_labels("./Data/t10k-images-idx3-ubyte/t10k-labels.idx1-ubyte")