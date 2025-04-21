# -*- codeing = utf-8 -*-
"""
@Time : 2024/11/3
@Author : AC
@File : clustering.py
@Software : PyCharm
"""
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
import torch.utils.data as data
import pandas as pd
from sklearn import preprocessing


class My_Kmeans(object):
    def __init__(self, k):
        self.k = k
        self.images_lists = None

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data)

        # cluster the data
        I, loss = run_kmeans(xb, self.k, verbose)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss


def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans using scikit-learn.

    Args:
        x: data (numpy array)
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
        float: final k-means loss
    """
    kmeans = KMeans(n_clusters=nmb_clusters, n_init=20, max_iter=300, random_state=np.random.randint(1234))
    kmeans.fit(x)

    # Get cluster labels
    labels = kmeans.labels_.tolist()

    # Get the final loss (inertia)
    final_loss = kmeans.inertia_

    if verbose:
        print('k-means loss evolution: Not available in sklearn, final loss: {0}'.format(final_loss))

    return labels, final_loss


def preprocess_features(npdata, pca=256):
    """Preprocess an array of features.

    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened, and L2-normalized
    """
    # Convert Tensor to NumPy array if necessary
    if isinstance(npdata, torch.Tensor):
        npdata = npdata.detach().cpu().numpy()  # Convert to NumPy array

    # Ensure the data is in float32
    npdata = npdata.astype('float32')

    # Apply PCA with whitening
    pca_model = PCA(n_components=pca, whiten=True)
    npdata = pca_model.fit_transform(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1, keepdims=True)
    npdata = npdata / row_sums

    # Handle any potential division by zero
    npdata[np.isnan(npdata)] = 0  # Set NaNs (from division by zero) to 0

    return npdata


class ReassignedDataset(data.Dataset):
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, image_indexes, pseudolabels, dataset, dataset_data, transform=None):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
        self.transform = transform
        self.dataset_data = dataset_data

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        for j, idx in enumerate(image_indexes):
            path = dataset[idx]
            # path = dataset[idx][0]
            pseudolabel = label_to_idx[pseudolabels[j]]
            images.append((path, pseudolabel))
        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        path, pseudolabel = self.imgs[index]
        signal_data = self.dataset_data[path]
        if self.transform is not None:
            signal_data = self.transform(signal_data)
        return signal_data, pseudolabel

    def __len__(self):
        return len(self.imgs)


def cluster_assign(images_lists, dataset, dataset_data):
    """Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster
        dataset (list): initial dataset :list of tuples with paths to images
        dataset_data: real signal data from dataloader
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels
    """
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    return ReassignedDataset(image_indexes, pseudolabels, dataset, dataset_data, None)
