import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def merge_images(image_batch, size = [5, 5]):
    h, w, c = image_batch.shape[1:]
    img = np.zeros((int(h*size[0]), w*size[1], c))
    for idx, im in enumerate(image_batch):
        if idx == np.prod(size):
            break
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w,:] = im#/255.0 #notice we divide by 255 to get 0-1 float range
    return img

def save_reconstructed_images(batch, batch_reconstructed, filename):
    n_grid = 5
    batch_images = merge_images(batch, size = [n_grid, n_grid])

    batch_images_recon = merge_images(batch_reconstructed, size = [n_grid, n_grid])

    fig, ax = plt.subplots(ncols = 2, figsize = (20,n_grid))

    # ax[0].set_title('Original')
    ax[0].axis('off')
    _ = ax[0].imshow(batch_images)

    # ax[1].set_title('Reconstructed')
    ax[1].axis('off')
    _ = ax[1].imshow(batch_images_recon)

    fig.savefig(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dir', help='Directory with batches npy files', required=True)
    parser.add_argument(
        '--out', help='Out directory for images', required=True)
    args = parser.parse_args()
    dir_in_name = args.dir
    dir_out_name = args.out
    list_of_files = os.listdir(dir_in_name)
    for f in list_of_files:
        batch, batch_reconstructed = np.load(dir_in_name + '/' + f)
        filename = dir_out_name + '/' + f.split('.')[0] + '.png'
        save_reconstructed_images(batch, batch_reconstructed, filename)