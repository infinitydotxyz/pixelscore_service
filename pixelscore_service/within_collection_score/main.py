"""Computes PixelScore for a single NFT colelction.

example run:
python3 pixelscore_service/within_collection_score/main.py
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
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from keras import backend as K
from numpy import savez_compressed

# Functions for loading model and scoring one collection of NFTs.
FLAGS = flags.FLAGS
PIXELSCORE_SCALING_MIN = 0.0
PIXELSCORE_SCALING_MAX = 10.0
LAYER_FROM_TOP = 2
N_CLASSES = 10
N_CLASSES_STANDARD_MODEL = 1000
# Used for debug.
# Read only MAX_EXAMPLES from collection, to always read full collection
# set to 100K
MAX_EXAMPLES = 100000
EFFICIENTNET_IMAGE_SIZE = 224
# Number of bins for pixel rarity score, must be less than collection size.
PIXEL_SCORE_BINS = 10
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
    'Local base directory containing images.')
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
    True,
    'Whether to use model checkpoint transfer learned for the given collection. If False, base EfficientNet with imagenet weights is used.')


def update_results(results_file, stats_dict):
    """Update results file with stats - by colleciton id.

    stats_dict: e.g. {'model_accuracy': 0.4, collection_id: '0x333'}
    """
    df_base = pd.read_csv(results_file)
    if df_base.empty:
        stats_dict_fordf = dict()
        for key, value in stats_dict.items():
            stats_dict_fordf[key] = [value]
        df_update = pd.DataFrame.from_dict(stats_dict_fordf)
        df_base = df_update
        print('DF results updated')
        print(df_base)
        df_base.to_csv(results_file)
        return True
    if stats_dict['collection_id'] in df_base['collection_id'].values:
        for key, value in stats_dict.items():
            df_base.loc[df_base.collection_id == stats_dict['collection_id'],key] = value
    else:
        stats_dict_fordf = dict()
        for key, value in stats_dict.items():
            stats_dict_fordf[key] = [value]
        df_update = pd.DataFrame.from_dict(stats_dict_fordf)
        df_base = pd.concat([df_base, df_update])
    print('DF results updated')
    print(df_base)
    df_base.to_csv(results_file, columns = [
        'collection_id',
        'model_accuracy',
        'corr_rarityScore',
        'corr_rarityRank'])
    return True

def rename_columns(df):
    """Rename and reorder columns in the output pixelscore df."""
    df.rename(columns={
        'id':'tokenId',
        'rarityScore':'rarityScore',
        'rarityRank':'rarityRank',
        'PixelScore':'pixelScore',
        'PixelScoreRank':'pixelScoreRank',
        'url':'imageUrl',
        'collectionAddress':'collectionAddress'
        },inplace=True)
    df = df[['collectionAddress','tokenId','pixelScore','pixelScoreRank','rarityScore','rarityRank','imageUrl']]
    return df

def load_collection_numpy(base_dir, collection_id):
    """Loads nft collection pixels as archived numpy array.

    Args:
      base_dir: Base data directory on the current vm e.g. /mnt/disks/ssd/data
      collection_id: collection address e.g. '0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d'
    Returns:
      X_train: np array with flattened pixels form entire collection e.g. [collection_length, 224 * 224]
    """
    # Load pixels.
    path = base_dir + '/{}'.format(collection_id) + '/numpy'
    filename = path + '/pixels.npz'
    X_train = np.load(filename)['arr_0']
    print('Loading pixels as numpy from {}'.format(filename))
    # Load ids.
    filename = path + '/ids.npz'
    ids = np.load(filename)['arr_0']
    print('Loading ids as numpy from {}'.format(filename))
    return X_train, ids


def save_collection_numpy(base_dir, collection_id, X_train):
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
    filename = path + '/dnn_layers.npz'
    savez_compressed('dnn_layers.npz', X_train)
    print('Saving layers as numpy to {}'.format(filename))
    os.system('sudo mv dnn_layers.npz {}'.format(filename))
    return True


def save_collection_scores(base_dir, collection_id, results_file, df):
    """Saves pixel scores for the given collection in .csv.

    Args:
      base_dir: Base data directory on the current vm e.g. /mnt/disks/ssd/data
      collection_id: collection address e.g. '0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d'
      df: dataframe with columns at least 'id' and 'PixelScore'

    Returns:
      True if collection was saved as numpy.
    """
    path = base_dir + '/{}'.format(collection_id) + '/pixelscore'
    if not os.path.exists(path):
        os.system('sudo mkdir {}'.format(path))
    filename = path + '/pixelscore.csv'
    
    # Merge with original metadata by left_join on id
    # i.e. All wors in original metadata are preserved.
    df_metadata = load_metadata(base_dir, collection_id)
    print(df_metadata.dtypes)
    print(df.dtypes)
    df = pd.merge(df_metadata, df, how = 'left', on = ['id'])
    print('Merged df metadata and scores.')
    print(df.head(10))    
    # Add pixelscore rank column, rank 1 is most rare.
    df['pixelScoreRank'] = df['PixelScore'].rank(ascending = False)
    # Correlation of pixelscore with rarityScore ground truth
    corr = df['rarityScore'].corr(df['PixelScore'])
    print('Correlation between ground truth rarityScore and pixelScore: {}'.format(corr))
    corr_rank = df['rarityRank'].corr(df['pixelScoreRank'])
    print('Correlation between ground truth rarityScoreRank and pixelScoreRank: {}'.format(corr_rank))
    update_results(results_file,
        {'collection_id': collection_id,
        'corr_rarityScore': corr,
        'corr_rarityRank': corr_rank,
        })
    # Rename columns in output format and save to disk.
    df = rename_columns(df)
    df.to_csv('pixelscore.csv')
    print('Saving collection scores to {}'.format(filename))
    os.system('sudo mv pixelscore.csv {}'.format(filename))
    # Save scores histogram.
    filename = path + '/hist.png'
    scores = df['pixelScore'].values
    fig = plt.hist(scores, bins=28)
    plt.title('Hist pixelscore')
    plt.xlabel("pixelscore")
    plt.ylabel("Frequency")
    plt.savefig('hist.png')
    os.system('sudo mv hist.png {}'.format(filename))
    return True


def load_metadata(base_dir, collection_id):
    """Loads metadata file.

    Args:
      base_dir: Base data directory on the current vm e.g. /mnt/disks/ssd/data
      collection_id: collection address e.g. '0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d'
    Returns:
      df: pandas Dataframe with loaded metadata and appended collection_ids.
    """
    # Load metadata with ground truth rarity scores.
    path = base_dir + '/{}'.format(collection_id) + '/metadata'
    filename = path + '/metadata.csv'
    df = pd.read_csv(filename, header=None, low_memory=False)
    df.columns = ['id', 'rarityScore', 'rarityRank', 'url']
    df['collectionAddress'] = collection_id
    print(df.head(10))
    return df


def load_standard_model():
    # print(help(tf.keras.applications))
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(
            EFFICIENTNET_IMAGE_SIZE,
            EFFICIENTNET_IMAGE_SIZE,
            3),
        weights="imagenet",
        classes=N_CLASSES_STANDARD_MODEL)
    return base_model


def load_checkpoint(base_dir, collection_id):
    model_path = base_dir + '/{}'.format(collection_id) + '/tf_logs/model'
    model = tf.keras.models.load_model(model_path)
    # Check its architecture
    print(model.summary())
    return model


def get_layer_output_nft(img_path, model):
    print('Loading image from {}'.format(img_path))
    img = Image.open(img_path)
    img = img.convert('RGB')
    img_array = np.array(img)
    #img = img.resize((EFFICIENTNET_IMAGE_SIZE,EFFICIENTNET_IMAGE_SIZE))
    # display(img)
    #img_array = image.img_to_array(img)
    print(img_array.shape)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    # with a Sequential model
    # get_3rd_layer_output = K.function([model.layers[0].input],
    #                                  [model.layers[3].output])
    layer_name = 'dense_3'
    intermediate_layer_model = keras.models.Model(
        inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(img_batch)
    # print(intermediate_output)
    layer_output = intermediate_output
    print('Obtained Layer output with shape: {}'.format(layer_output.shape))
    del img
    gc.collect()
    print('Layer output shape {}'.format(layer_output.shape))
    return layer_output


def get_layer_output_collection(base_dir, collection_id, model):
    # Take each image
    collection_folder = base_dir + '/{}'.format(collection_id) + '/resized'
    output_array = []
    ids = []
    count = 0
    for f in os.listdir(collection_folder):
        path = collection_folder + '/{}'.format(f)
        layer_output = get_layer_output_nft(path, model).flatten()
        output_array.append(layer_output)
        ids.append(f)
        print(len(output_array))
        count += 1
        if count > MAX_EXAMPLES:
            break
    X_train = np.array(output_array)
    print(X_train.shape)
    save_collection_numpy(base_dir, collection_id, X_train)
    return X_train, ids


def get_layer_output_collection_from_numpy(base_dir, collection_id, model):
    # Loads entire collection as np and passes to model to get output layer.
    X_train, ids = load_collection_numpy(base_dir, collection_id)
    # TODO(dstorcheus): if needed process layer outputs per batch.
    print('Getting model layer output for the entire collection at once, takes a few mins.')
    layer_name = 'dense_3'
    intermediate_layer_model = keras.models.Model(
        inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(X_train)
    # print(intermediate_output)
    layer_output = intermediate_output
    print('Obtained Layer output with shape: {}'.format(layer_output.shape))
    gc.collect()
    save_collection_numpy(base_dir, collection_id, layer_output)
    return layer_output, ids


def get_scores_collection(X_train, ids, sum_neurons = True):
    # Score is created as new column in df.
    if sum_neurons:
        scores = np.mean(X_train, axis = 1)
    else:
        pixel_scores = []
        # Obtain histograms.
        est = KBinsDiscretizer(
            n_bins=PIXEL_SCORE_BINS,
            encode='ordinal',
            strategy='kmeans')
        print('Fitting KBinsDiscretizer to break layer values into bins.')
        est.fit(X_train)
        Xt = est.transform(X_train)
        # Xt are the actual scores.
        print(Xt)
        scores = np.mean(Xt, axis=1)
    print('Scores {}'.format(scores))
    # Scale the scores for fixed range
    scaler = MinMaxScaler(
        feature_range=(
            PIXELSCORE_SCALING_MIN,
            PIXELSCORE_SCALING_MAX))
    scores = scaler.fit_transform(scores.reshape(-1, 1))
    print(
        'Scores scaled to ({},{})  {}'.format(
            PIXELSCORE_SCALING_MIN,
            PIXELSCORE_SCALING_MAX,
            scores))
    df = pd.DataFrame()
    df['id'] = ids.astype(np.int64)
    df['PixelScore'] = scores
    print(df.head(10))
    # (unique, counts) = np.unique(Xt, return_counts=True)
    # Counts of each bin value.
    # frequencies = np.asarray((unique, counts)).T
    # print(frequencies)
    # Xt = Xt.flatten()
    return df


def main(argv):
    if FLAGS.collection_id is not None:
        print('Generating Scres for collection {}'.format(FLAGS.collection_id))
    if FLAGS.use_checkpoint:
        model = load_checkpoint(FLAGS.base_dir, FLAGS.collection_id)
    else:
        model = load_standard_model()
    X_train, ids = get_layer_output_collection_from_numpy(
        FLAGS.base_dir, FLAGS.collection_id, model)
    df = get_scores_collection(X_train, ids)
    save_collection_scores(FLAGS.base_dir, FLAGS.collection_id, FLAGS.results_file, df)
    print(
        'Completed Score generation for collection {}'.format(
            FLAGS.collection_id))
    print('Success')


if __name__ == '__main__':
    app.run(main)
