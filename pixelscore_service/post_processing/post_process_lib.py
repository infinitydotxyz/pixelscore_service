"""Converts folder of images to numpy array for a single NFT colelction.

1) Converts images to numpy arrays
2) Creates labels for them based on ground_truth rarityScore
3) Saves results as np arrays

example run:
python3 pixelscore_service/within_collection_score/img_to_numpy.py
  --collection_id='0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d'
  --base_dir=/mnt/disks/ssd/data
  --checkpoints_dir=/mnt/disks/ssd/data/checkpoints

"""

import numpy
import pandas as pd
import sklearn
import scipy
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import gc
import sys
import numpy as np
from PIL import Image
from absl import app
from absl import flags

from sklearn.preprocessing import KBinsDiscretizer

from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from keras import backend as K
from numpy import savez_compressed

# Functions for loading model and scoring one collection of NFTs.
FLAGS = flags.FLAGS
N_CLASSES = 10
N_CLASSES_STANDARD_MODEL = 1000
GROUND_TRUTH_N_CLASSES = 10
# Used for debug.
# Read only MAX_EXAMPLES from collection, to always read full collection
# set to 100K
MAX_EXAMPLES = 100000
EFFICIENTNET_IMAGE_SIZE = 224
# Number of bins for pixel rarity score, must be less than collection size.
PIXEL_SCORE_BINS = 100
# Flag names are globally defined!  So in general, we need to be
# careful to pick names that are unlikely to be used by other libraries.
# If there is a conflict, we'll get an error at import time.
flags.DEFINE_string(
    'collections_whitelist',
    '/mnt/disks/ssd/pixelscore_service/whitelists_blacklists/global_hist_ready_10_Apr_2022.csv',
    'Collections whitelist.')
flags.DEFINE_string(
    'collection_id',
    '0x9a534628b4062e123ce7ee2222ec20b86e16ca8f',
    'Collection id.')
flags.DEFINE_string(
    'base_dir',
    '/mnt/disks/ssd/data',
    'Local base directory containing images, resized, metadata, etc.')
flags.DEFINE_string(
    'pre_reveal_logs_dir',
    '/mnt/disks/additional-disk/raw_logs/tmp_preprocess/is_pre_reveal',
    'Logs to save pre reveal ids.')
flags.DEFINE_string(
    'is_empty_logs_dir',
    '/mnt/disks/additional-disk/raw_logs/tmp_preprocess/is_empty',
    'Logs to save is empty ids.')
flags.DEFINE_string(
    'results_file',
    '/mnt/disks/ssd/pixelscore_service/within_collection_score/scoring_results/results.csv',
    'File .csv containing stats for current scoring round.')
flags.DEFINE_string(
    'checkpoints_dir',
    '/mnt/disks/ssd/checkpoints',
    'Local dire where model checkpoints for each collection are stored.')
flags.DEFINE_boolean(
    'use_checkpoint',
    False,
    'Whether to use model checkpoint transfer learned for the given collection. If False, base EfficientNet with imagenet weights is used.')
flags.DEFINE_string(
    'mode',
    'is_pre_reveal',
    'Which mode to run main script: is_pre_reveal, .')
flags.DEFINE_string(
    'scored_collections_whitelist',
    '/mnt/disks/ssd/pixelscore_service/whitelists_blacklists/scored_ids_11_Apr_2022.csv',
    'Scored collections.')
flags.DEFINE_string(
    'merged_scores_file',
    '/mnt/disks/ssd/pixelscore_service/post_processing/merged_sorted_global_raw_pixelscore.csv',
    'File with all scores for all collections merged.')
flags.DEFINE_string(
    'post_processing_dir',
    '/mnt/disks/additional-disk/post_processing',
    'Dir to store post-processing results, analysis, charts.')
flags.DEFINE_string(
    'global_hist_path',
    '/mnt/disks/additional-disk/global_hist/global_hist.npz',
    'Path to global hist.')

# Standard image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224

def get_global_hist(global_hist, i, j):
    """Quick get global hist."""
    index = i + IMG_HEIGHT + j
    hist = global_hist[index, :, :, :]
    return hist

def apply_qcut(df, n_classes):
    """Applies careful qcut to rarityScore.

    if number of possible quantiles less than n_classes, then n_classes is
    reduced to max num of quantiles.

    Reduction typically happens if collection size < 2K

    Returns df qith rarity_bin column
    """
    groups = pd.qcut(
        df['rarityScore'],
        n_classes,
        duplicates='drop')
    n_labels = groups.describe()['unique']
    if n_labels == n_classes:
        df['rarity_bin'] = pd.qcut(
            df['rarityScore'],
            n_classes,
            duplicates='drop',
            labels=np.arange(n_classes))
        return df
    else:
        print('Max possible number of bins (labels) for rarityScore is {}, which is less than proposed n_classes {}'.format(
            n_labels, n_classes))
        df['rarity_bin'] = pd.qcut(
            df['rarityScore'],
            n_labels,
            duplicates='drop',
            labels=np.arange(n_labels-1))
    return df


def load_labels(base_dir, collection_id, ids):
    """Loads labels based on ground-truth rarity.score for a specific nft collection.

    Args:
      base_dir: Base data directory on the current vm e.g. /mnt/disks/ssd/data
      collection_id: collection address e.g. '0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d'
    Returns:
      y_train: np array with labels for  entire collection e.g. [collection_length]
    """
    # Load metadata with ground truth rarity scores.
    path = base_dir + '/{}'.format(collection_id) + '/metadata'
    filename = path + '/metadata.csv'
    df = pd.read_csv(filename, header=None, low_memory=False)
    df.columns = ['id', 'rarityScore', 'rarityRank', 'url']
    # TODO(dstorcheus): check if dropping values preserves consistent training (X,y)  ids.
    df.drop(df[df.rarityScore == 'undefined'].index, inplace=True)
    df = df.astype({"rarityScore": float})
    df = apply_qcut(df, GROUND_TRUTH_N_CLASSES)
    print(df.head())
    y_train = []
    # Match labels by ids.
    # TODO(dstorcheus): double check that ids are correctly matching.
    # Name of the image corresponds to id row in metadata base.
    print(ids)
    for this_id in ids:
        df_select = df.loc[df['id'] == int(this_id)]
        print(df_select)
        if (len(df_select.index) > 1):
            print('Error: non-unique matching index for labelling.')
        else:
            this_label = df_select['rarity_bin'].values[0]
            print('This label. {}'.format(this_label))
        y_train.append(this_label)
    return np.array(y_train)


def save_pixels_numpy(base_dir, collection_id, X_train, ids):
    """Saves nft collection pixels as archived numpy array.

    Args:
      base_dir: Base data directory on the current vm e.g. /mnt/disks/ssd/data
      collection_id: collection address e.g. '0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d'
      X_train: np array with flattened pixels form entire collection e.g. [collection_length, 224 * 224]
    Returns:
      True if collection was saved as numpy.
    """
    path = base_dir + '/{}'.format(collection_id) + '/numpy'
    if not os.path.exists(path):
        os.system('sudo mkdir {}'.format(path))

    # Save pixels.
    filename = path + '/pixels.npz'
    savez_compressed(filename, X_train)
    print('Saving pixels as numpy to {}'.format(filename))

    # Save ids.
    filename = path + '/ids.npz'
    savez_compressed(filename, ids)
    print('Saving ids as numpy to {}'.format(filename))
    return True


def save_labels_numpy(base_dir, collection_id, y_train, ids):
    """Saves nft collection pixels as archived numpy array.

    Args:
      base_dir: Base data directory on the current vm e.g. /mnt/disks/ssd/data
      collection_id: collection address e.g. '0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d'
      X_train: np array with flattened pixels form entire collection e.g. [collection_length, 224 * 224]
    Returns:
      True if collection was saved as numpy.
    """
    path = base_dir + '/{}'.format(collection_id) + '/numpy'
    if not os.path.exists(path):
        os.system('sudo mkdir {}'.format(path))

    # Save pixels.
    filename = path + '/labels.npz'
    savez_compressed(filename, y_train)
    print('Saving pixels as numpy to {}'.format(filename))
    filename = path + '/labels.npz'

    return True


def img_to_array(img_path):
    print('Loading image from {}'.format(img_path))
    img = Image.open(img_path)
    width, height = img.size
    print('Opened image with dimensions {} by {}'.format(width, height))
    img = img.convert('RGB')
    # Resize since images may have different sizes still even when loaded.
    img = img.resize((EFFICIENTNET_IMAGE_SIZE, EFFICIENTNET_IMAGE_SIZE))
    img_array = np.array(img)
    print(img_array.shape)
    #img_batch = np.expand_dims(img_array, axis=0)
    del img
    gc.collect()
    return img_array


def collection_to_array(base_dir, collection_id):
    # Convert images to numpy
    collection_folder = base_dir + '/{}'.format(collection_id) + '/resized'
    output_array = []
    ids = []
    count = 0
    for f in os.listdir(collection_folder):
        path = collection_folder + '/{}'.format(f)
        try:
            image_array = img_to_array(path)
            output_array.append(image_array)
            ids.append(f)
            print(len(output_array))
        except BaseException:
            print('Unable to load image from: {}, skipping'.format(path))
        count += 1
        if count > MAX_EXAMPLES:
            break
    X_train = np.array(output_array)
    print(X_train.shape)
    return X_train, ids


def scores_hist(base_dir, collection_id):
    """ Colelct raw global pixel score from all scored collections

    and put that into hist
    return True
    """
    all_scores = []
    for id in FLAGS.collections_whitelist:
        df = pd.read_csv(
            base_dir + '/{}/pixelscore/global_raw_pixelscore.csv'.format(id))
        scores = df['pixelScore']
        all_scores.append(scores)
    all_scores = np.array(all_scores).flatten()
    fig = plt.hist(all_scores, bins=100)
    plt.title('Global pixelscore hist all collections')
    plt.xlabel("pixelscore")
    plt.ylabel("Frequency")
    plt.savefig('all_collections_hist.png')
    print('Hist Successfully saved')
    return True


def bucketize_scores(base_dir):
    """ Bucketize final scores into bins

    bins: 1 to 10
    """

    df = pd.read_csv('merged_sorted_global_raw_pixelscore.csv')
    bin_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # What quantiles should we select for scores?
    quantiles = [0.17, 0.34, 0.49, 0.62, 0.73,
                 0.82, 0.89, 0.94, 0.97, 0.99, 1.0]
    df['bin_pixelScore'] = pd.qcut(df['pixelScore'],
                                   q=quantiles,
                                   labels=bin_labels)
    print(df.head(100)['bin_pixelScore'])
    # Check hist of that.
    binned_scores = df['bin_pixelScore'].values
    print(binned_scores)
    fig = plt.hist(binned_scores, bins=10)
    plt.title('Binned pixel score')
    plt.xlabel("pixelscore")
    plt.ylabel("Frequency")
    plt.savefig('binned_pixelscore_hist.png')
    print('Binned hist Successfully saved')
    return True


def merge_results(base_dir, collection_id, do_sort=True):
    """ Merge global_raw_pixelscore.csv results

    Done for all collections at once.
    """
    whitelist = pd.read_csv(FLAGS.scored_collections_whitelist)[
        'collection_id'].values
    id = whitelist[0]
    all_df = pd.read_csv(
        base_dir + '/{}/pixelscore/global_raw_pixelscore.csv'.format(id))
    whitelist = whitelist[1:]
    for id in whitelist:
        df = pd.read_csv(
            base_dir + '/{}/pixelscore/global_raw_pixelscore.csv'.format(id))
        all_df = pd.concat([all_df, df])
        print('Merged collection {}'.format(id))
    all_df.sort_values(by = 'pixelScore', ascending=True, inplace=True)
    all_df.to_csv(FLAGS.merged_scores_file)

def plot_3d_pixel_histogram():
    """Plots 3D histogram of RGB for specific pixel.
    
    Loads global hist object
    """
    i = 10
    j = 15
    global_hist = np.load(FLAGS.global_hist_path)['arr_0']
    hist = get_global_hist(global_hist, i, j)

    return True
    
def analyze_merged_scores(merged_scores_file, post_processing_dir):
    """Stats analysis of merged scores from all collections."""

    # Part 1: Line plot.
    top_k = 10000
    df = pd.read_csv(merged_scores_file)
    y = df['pixelScore'][:top_k]
    x = range(len(y))
    # Smoothing to yhat
    #from scipy.signal import savgol_filter
    #yhat = savgol_filter(y, 1001, 2)
    fig = plt.plot(x, y)
    plt.title('Sorted PIXELSCORE chart')
    plt.xlabel("token id")
    plt.ylabel("PIXELSCORE")
    plt.savefig(post_processing_dir + '/merged_scores_plot.png', dpi = 512)

    #Part 2: Binned histogram.

    return True

def main(argv):
    if FLAGS.collection_id is not None:
        print('Preprocess for collection {}'.format(FLAGS.collection_id))
    if FLAGS.mode == 'scores_hist':
        print('All scores hist.')
        scores_hist(FLAGS.base_dir, FLAGS.collection_id)
    if FLAGS.mode == 'merge_results':
        print('Merge results.')
        merge_results(FLAGS.base_dir, FLAGS.collection_id)
    if FLAGS.mode == 'bucketize_scores':
        print('Bucketizing scores.')
        bucketize_scores(FLAGS.base_dir)
    if FLAGS.mode == 'analyze_merged_scores':
        print('Analyzing merged scores.')
        analyze_merged_scores(FLAGS.merged_scores_file, FLAGS.post_processing_dir)
    if FLAGS.mode == 'plot_3d_pixel_histogram':
        print('Plotting 3d pixel histogram.')
        plot_3d_pixel_histogram(FLAGS.post_processing_dir)
    print('Success')


if __name__ == '__main__':
    app.run(main)
