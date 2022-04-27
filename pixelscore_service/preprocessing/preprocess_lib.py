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


def is_empty(base_dir, collection_id):
    """Is col has empty np array, then

    save it's id to temp dir
    return True
    """
    print('Checking is empty')
    path = base_dir + '/{}'.format(collection_id) + '/numpy/pixels.npz'
    x = np.load(path)['arr_0']
    is_empty = False
    if len(x) == 0:
        is_empty = True
        # flush log
        # /mnt/disks/additional-disk/raw_logs/tmp_preprocess/is_pre_reveal
        filename = FLAGS.is_empty_logs_dir + '/{}'.format(collection_id)
        #print('Saving to {}'.format(filename))
        os.system('sudo touch {}'.format(filename))
        print('Is empty: {}'.format(collection_id))
    return is_empty


def is_pre_reveal(base_dir, collection_id):
    """Is col is pre-reveal (all images equal), then

    save it's id to temp dir
    return True
    """
    print('Checking is pre reveal')
    sample_size = 10
    path = base_dir + '/{}'.format(collection_id) + '/numpy/pixels.npz'
    x = np.load(path)['arr_0']
    # Get random images
    ind = np.random.choice(range(len(x)), sample_size)
    x_sample = x[ind, :]
    # Check equality
    n_equal = 1
    first = x_sample[0]
    x_list = x_sample[1:]
    for i in x_list:
        if np.array_equiv(first, i):
            n_equal += 1
    if n_equal == sample_size:
        is_reveal = True
        # flush log
        # /mnt/disks/additional-disk/raw_logs/tmp_preprocess/is_pre_reveal
        filename = FLAGS.pre_reveal_logs_dir + '/{}'.format(collection_id)
        #print('Saving to {}'.format(filename))
        os.system('sudo touch {}'.format(filename))
        print('Is pre-reveal: {}'.format(collection_id))
    return is_pre_reveal


def main(argv):
    if FLAGS.collection_id is not None:
        print('Preprocess for collection {}'.format(FLAGS.collection_id))
    if FLAGS.mode == 'is_pre_reveal':
        print('CHECK pre reveal')
        is_pre_reveal(FLAGS.base_dir, FLAGS.collection_id)
    if FLAGS.mode == 'is_empty':
        print('CHECK is empty')
        is_empty(FLAGS.base_dir, FLAGS.collection_id)
    print('Success')


if __name__ == '__main__':
    app.run(main)
