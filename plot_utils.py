import os
import glob
import time
from PIL import Image
from IPython.display import HTML

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams['animation.embed_limit'] = 2**128

ROOT_DIR = 'generated_images/'

def create_plot_and_save(generator: nn.Module,
                         fix_noise: torch.Tensor,
                         real_points: torch.Tensor,
                         folder_name: str):
    plt.figure(figsize=(7, 7))
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)

    plt.plot(real_points[:, 0].tolist(),
             real_points[:, 1].tolist(),
             'o', color='magenta', label="REAL")

    with torch.no_grad():
        fake_points = generator(fix_noise).detach()

    plt.plot(fake_points[:, 0].tolist(),
             fake_points[:, 1].tolist(),
             'o', color='green', label="GENERATED")

    plt.legend(loc='upper right', framealpha=1.0)
    
    IMAGE_PATH = ROOT_DIR + folder_name
    if not os.path.exists(IMAGE_PATH):
        os.makedirs(IMAGE_PATH)

    plt.savefig(IMAGE_PATH + '/%.4f.png'%time.time())
    plt.close()


def delete_all_images(folder_name: str):
    for f in glob.glob(ROOT_DIR + folder_name + '/*'):
        os.remove(f)


def load_plot_and_show(folder_name: str):
    fig = plt.figure(figsize=(7,7))
    plt.axis("off")

    images = []
    for f in sorted(glob.glob(ROOT_DIR + folder_name + '/*')):
        temp = np.asarray(Image.open(f))
        temp = [plt.imshow(temp, animated=True)]
        images.append(temp)

    ani = animation.ArtistAnimation(fig,
                                    images,
                                    interval=50, repeat_delay=100, blit=False)
    plt.close()
    return HTML(ani.to_jshtml())

