

from torch.utils.data import Dataset
import os
from typing import List
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def loss_and_acc(acc, loss, title, epochs):
    fig = plt.figure(figsize=(10, 8))
    plt.rc('axes', titlesize=20)
    plt.rc('axes', labelsize=20)
    plt.rc('legend', fontsize=17)
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)

    lns1 = plt.plot(range(len(loss)), loss, label='loss', color='darkblue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(linestyle='--', color='lavender')

    ax_loss = plt.twinx()
    lns2 = ax_loss.plot(range(len(acc)), acc, label='acc', color='darkred')
    plt.ylabel('Accuracy')

    plt.xticks(np.arange(0, epochs, step=5), ["{:<2d}".format(x) for x in np.arange(0, epochs, step=5)],
               fontsize=10,
               rotation=45
               )
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc=0)
    plt.title(title, fontsize=30)

    plt.show()


def list_files(root: str, suffix: str, prefix: bool = False) -> List[str]:
    root = os.path.expanduser(root)
    files = [int(p[:-4]) for p in os.listdir(root) if os.path.isfile(os.path.join(root, p)) and p.endswith(suffix)]
    if prefix is True:
        files = [os.path.join(root, d) for d in files]
    return files


class TestDataSet(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.images = list_files(root, suffix='.jpg')
        self.images = sorted(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        img_loc = os.path.join(self.root, str(image_name))
        img_loc = img_loc+'.jpg'
        image = Image.open(img_loc).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_name