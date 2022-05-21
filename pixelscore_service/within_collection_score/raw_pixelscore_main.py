"""Computes PixelScore for a single NFT colelction.

raw_pixelscore is based only on pixel histogram bins.

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
import math
import os
import gc
import sys
import numpy as np
from PIL import Image
from absl import app
from absl import flags
# import multiprocessing as mp
import time

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

# Standard image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Histogram Bins for each pixel.
N_BINS_R = 10
N_BINS_G = 10
N_BINS_B = 10

# Hardcoded hist edges.
EDGES = [np.array([0.,  25.5,  51.,  76.5, 102., 127.5, 153., 178.5, 204.,
                   229.5, 255.]), np.array([0.,  25.5,  51.,  76.5, 102., 127.5, 153., 178.5, 204.,
                                            229.5, 255.]), np.array([0.,  25.5,  51.,  76.5, 102., 127.5, 153., 178.5, 204.,
                                                                     229.5, 255.])]

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
flags.DEFINE_string(
    'hist_dir',
    '/mnt/disks/additional-disk/histograms',
    'Dir to save histograms.')
flags.DEFINE_boolean(
    'use_checkpoint',
    True,
    'Whether to use model checkpoint transfer learned for the given collection. If False, base EfficientNet with imagenet weights is used.')
flags.DEFINE_boolean(
    'save_collection_counts',
    False,
    'Whether to save collection counts in the process.')
flags.DEFINE_boolean(
    'save_pixels_hist',
    False,
    'Whether to save pixels hist in the process.')
flags.DEFINE_boolean(
    'global_score',
    True,
    'Whether to apply global score scaling.')
flags.DEFINE_string(
    'raw_logs_dir',
    '/mnt/disks/additional-disk/raw_logs/tmp',
    'Raw logs.')
flags.DEFINE_boolean(
    'use_log_scores',
    False,
    'Whether to use scores as -log(1+prob).')

GLOBAL_HIST_PATH = '/mnt/disks/additional-disk/global_hist/global_hist.npz'
GLOBAL_HIST = np.load(GLOBAL_HIST_PATH)['arr_0']

def get_global_hist(i, j):
    """Quick get global hist."""
    index = i + IMG_HEIGHT + j
    hist = GLOBAL_HIST[index, :, :, :]
    return hist

# Global smallest pixel count
MIN_PIXEL_COUNT = np.min(GLOBAL_HIST[np.nonzero(GLOBAL_HIST)])
PIXEL_SUMS = dict()
# For each pixel (i,j) contains its total count from all bins.
for i in range(IMG_HEIGHT):
    for j in range(IMG_WIDTH):
            PIXEL_SUMS[(i,j)] = np.sum(get_global_hist(i,j))



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
            df_base.loc[df_base.collection_id ==
                        stats_dict['collection_id'], key] = value
    else:
        stats_dict_fordf = dict()
        for key, value in stats_dict.items():
            stats_dict_fordf[key] = [value]
        df_update = pd.DataFrame.from_dict(stats_dict_fordf)
        df_base = pd.concat([df_base, df_update])
    print('DF results updated')
    print(df_base)
    df_base.to_csv(results_file, columns=[
        'collection_id',
        'corr_rarityScore',
        'corr_rarityRank'])
    # Also fluch raw log as filename
    return True


def rename_columns(df):
    """Rename and reorder columns in the output pixelscore df."""
    df.rename(columns={
        'id': 'tokenId',
        'rarityScore': 'rarityScore',
        'rarityRank': 'rarityRank',
        'PixelScore': 'pixelScore',
        'PixelScoreRank': 'pixelScoreRank',
        'url': 'imageUrl',
        'collectionAddress': 'collectionAddress'
    }, inplace=True)
    df = df[['collectionAddress', 'tokenId', 'pixelScore',
             'pixelScoreRank', 'rarityScore', 'rarityRank', 'imageUrl']]
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
        os.system('mkdir {}'.format(path))
    filename = path + '/dnn_layers.npz'
    savez_compressed('dnn_layers.npz', X_train)
    print('Saving layers as numpy to {}'.format(filename))
    os.system('mv dnn_layers.npz {}'.format(filename))
    return True


def save_collection_scores(base_dir, collection_id, results_file, df, global_score=True, use_log_scores=False):
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
        os.system('mkdir {}'.format(path))
    if global_score:
        if use_log_scores:
            filename_score = path + '/global_raw_pixelscore_log.csv'
            filename_hist = path + '/global_raw_pixelscore_log_hist.png'
        else:
            filename_score = path + '/global_raw_pixelscore.csv'
            filename_hist = path + '/global_raw_pixelscore_hist.png'
    else:
        filename_score = path + '/raw_pixelscore.csv'
        filename_hist = path + '/raw_pixelscore_hist.png'

    # Merge with original metadata by left_join on id
    # i.e. All wors in original metadata are preserved.
    df_metadata = load_metadata(base_dir, collection_id)
    print(df_metadata.dtypes)
    print(df.dtypes)
    df = pd.merge(df_metadata, df, how='left', on=['id'])
    print('Merged df metadata and scores.')
    print(df.head(10))
    # Add pixelscore rank column, rank 1 is most rare.
    df['pixelScoreRank'] = df['PixelScore'].rank(ascending=False)
    # Correlation of pixelscore with rarityScore ground truth
    corr = df['rarityScore'].corr(df['PixelScore'])
    print('Correlation between ground truth rarityScore and pixelScore: {}'.format(corr))
    corr_rank = df['rarityRank'].corr(df['pixelScoreRank'])
    print('Correlation between ground truth rarityScoreRank and pixelScoreRank: {}'.format(corr_rank))
    # Rename columns in output format and save to disk.
    df = rename_columns(df)
    df.to_csv(filename_score)
    print('Saving raw collection scores to {}'.format(filename_score))
    # Save scores histogram.
    scores = df['pixelScore'].values
    fig = plt.hist(scores, bins=100)
    plt.title('Hist pixelscore')
    plt.xlabel("pixelscore")
    plt.ylabel("Frequency")
    plt.savefig(filename_hist)

    # Raw log flush to tempdir.
    n_files = len(os.listdir(FLAGS.raw_logs_dir)) + 1
    filestring = FLAGS.raw_logs_dir + \
        '/{},{},{},{}.txt'.format(n_files, collection_id, corr, corr_rank)
    os.system('touch {}'.format(filestring))

    update_results(results_file,
                   {'collection_id': collection_id,
                    'corr_rarityScore': corr,
                    'corr_rarityRank': corr_rank,
                    })
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


def get_bin_count(point, H, edges):
    """ Bin count for single 3D point.
    Binretrieval for a new 3D point (x,y,z) is easy:
    check index i in edges[0] where x fall into
    check index j in edges[1] where y falls into
    check index k in edges[2] where x falls into
    """
    x = point[0]
    y = point[1]
    z = point[2]

    x_edges = edges[0]
    x_bin = len(x_edges) - 2  # Init to last bin to make 255 included
    # Left bin edge included - double check that.
    # TODO(dstorcheus): check inclusion of right edge
    for i in range(1, len(x_edges)):
        if x_edges[i-1] <= x and x < x_edges[i]:
            x_bin = i-1
            break
    y_edges = edges[1]
    y_bin = len(y_edges) - 2
    # TODO(dstorcheus): check inclusion of right edge
    for i in range(1, len(y_edges)):
        if y_edges[i-1] <= y and y < y_edges[i]:
            y_bin = i-1
            break
    z_edges = edges[2]
    z_bin = len(z_edges) - 2
    # TODO(dstorcheus): check inclusion of right edge
    for i in range(1, len(z_edges)):
        if z_edges[i-1] <= z and z < z_edges[i]:
            z_bin = i-1
            break
    #print('Found bins')
    # print((x_bin,y_bin,z_bin))
    count = H[x_bin, y_bin, z_bin]
    return count


def get_bin_count_opt(pixel, H, edges):
    """Optimized bet bin counts for single pixel acxross entire collection."""
    ind_0 = np.digitize(pixel[:, 0], edges[0]) - 1
    ind_1 = np.digitize(pixel[:, 1], edges[1]) - 1
    ind_2 = np.digitize(pixel[:, 2], edges[2]) - 1
    # The last bin intex must be again -1 to match 255 pixel
    ind_0 = np.where(ind_0 == 10, 9, ind_0)
    ind_1 = np.where(ind_1 == 10, 9, ind_1)
    ind_2 = np.where(ind_2 == 10, 9, ind_2)
    counts_opt = H[(ind_0, ind_1, ind_2)]
    return counts_opt


def get_single_pixel_score(base_dir, collection_id, pixel, H, edges, save_hist=False):
    """Score number for one pixel.


    It's count is in Hist[i,j,k]

    pixel: [n_tokens, 3]

    """
    #start = time.time()
    if save_hist:
        filename = FLAGS.hist_dir + \
            '/{}/p_{}_{}_hist.npz'.format(collection_id, i, j)
        # savez with two arrays: H and edges
        #np.savez_compressed('p_{}_{}_hist.npz'.format(i ,j), H = H, edges = edges)
        np.savez_compressed(filename, H)
        print('Saving histogrm and edges to {}'.format(filename))
        #os.system('sudo mv p_{}_{}_hist.npz {}'.format(i, j, filename))
    #stop = time.time()
    #print("Histogram time:", stop - start)
    # print(H.shape)
    # Flattened hist corresponds to counts for all 10x10x10=1000 3D bin boxes.
    #counts = H.flatten()
    # TODO(dstorcheus): get the list of 3D intervals for bins (cubes), needed for new collections placing.
    # Now we get binned frequencies for this particular pixel across the entire collection nfts
    # Try to get opimized counts
    #stop = time.time()
    counts_opt = get_bin_count_opt(pixel, H, edges)
    #stop = time.time()
    #print("Counts opt time time:", stop - start)
    """
    start = time.time()
    counts = []
    for x in pixel:
        x_count = get_bin_count(x, H, edges)
        counts.append(x_count)
    stop = time.time()
    print("Bin count time:", stop - start)
    #print(counts_opt)
    #print(counts)
    """
    return counts_opt


def get_raw_pixelscores_collection(base_dir, collection_id, X_train, ids, save_collection_counts=False, save_pixels_hist=False, global_score=True, use_log_scores = False):
    # Score is created as new column in df.
    # For each pixel
    collection_scores = []
    pixels_hist = []
    pixel_sums = []
    for i in range(IMG_HEIGHT):
        for j in range(IMG_WIDTH):
            pixel = X_train[:, i, j]
            # TODO: uncomment if local scores are needed.
            H, edges = np.histogramdd(pixel,
                                      bins=(N_BINS_R, N_BINS_G, N_BINS_B),
                                      range=[(0, 255), (0, 255), (0, 255)])
            pixels_hist.append(H)

            # Replace hist by the global hist if we are running global score compute.
            if global_score:
                H = get_global_hist(i, j)
            # Counts returned.
            pixel_scores = get_single_pixel_score(
                base_dir, collection_id, pixel, H, edges)
            collection_scores.append(pixel_scores)
            pixel_sums.append(PIXEL_SUMS[(i,j)])
            #print('Processed pixel ({}, {}) of (224,224)'.format(i,j))
    if save_collection_counts:
        collection_counts = np.array(collection_scores).astype('int32')
        filename = base_dir + \
            '/{}/numpy/pixel_counts.npz'.format(collection_id)
        savez_compressed('pixel_counts.npz', X_train)
        print('Saving pixel counts as numpy to {}'.format(filename))
        os.system('mv pixel_counts.npz {}'.format(filename))
    if save_pixels_hist:
        folder = FLAGS.hist_dir + '/{}'.format(collection_id)
        if not os.path.exists(folder):
            os.system('mkdir {}'.format(folder))
        pixels_hist_ = np.array(pixels_hist).astype('int32')
        filename = FLAGS.hist_dir + '/{}/pixels_hist.npz'.format(collection_id)
        print(pixels_hist_.shape)
        savez_compressed(filename, pixels_hist_)
        print('Saving pixels hist as numpy to {}'.format(filename))
        #os.system('sudo mv pixels_hist.npz {}'.format(filename))
    # Now collection_scores are [224 *224 , n_tokens] and represent count numbers.
    # Aggrerage them
    # Aggregate over bins.
    # Check scores of pixel 100x100
    # For global score inversion should happen before summation.
    # TODO(dstorcheus): check the above.
    collection_scores = np.array(collection_scores)
    if global_score:
        if use_log_scores:
            # These are probability estimates, pixel_sums actually has the same value repeated, so we just take it using the mean.
            collection_scores = numpy.divide(collection_scores, np.mean(pixel_sums))
            collection_scores = 1.0 - np.log(1.0 + collection_scores)
            #collection_scores = -1.0 * np.log(collection_scores)
            collection_scores = np.mean(collection_scores, axis=0)
        else:
            collection_scores = PIXELSCORE_SCALING_MAX * \
                MIN_PIXEL_COUNT * 1.0 / collection_scores
            collection_scores = np.mean(collection_scores, axis=0)
    else:
        collection_scores = np.mean(collection_scores, axis=0)
        collection_scores = 1.0 / collection_scores
    # Now recall that rarity is inversely proportional to frequency.

    # Also the global score shoul dbe done without min-max rescaler, since it is global!
    if global_score:
        scores = collection_scores
    else:
        scaler = MinMaxScaler(
            feature_range=(
                PIXELSCORE_SCALING_MIN,
                PIXELSCORE_SCALING_MAX))
        scores = scaler.fit_transform(collection_scores.reshape(-1, 1))
    df = pd.DataFrame()
    df['id'] = ids.astype(np.int64)
    df['PixelScore'] = scores
    print(df.head(10))
    return df


def get_single_pixel_global_counts(pixel, global_counts):
    """Optimized bet bin counts for single pixel acxross entire collection."""
    ind_0 = np.digitize(pixel[:, 0], edges[0]) - 1
    ind_1 = np.digitize(pixel[:, 1], edges[1]) - 1
    ind_2 = np.digitize(pixel[:, 2], edges[2]) - 1
    # The last bin intex must be again -1 to match 255 pixel
    ind_0 = np.where(ind_0 == 10, 9, ind_0)
    ind_1 = np.where(ind_1 == 10, 9, ind_1)
    ind_2 = np.where(ind_2 == 10, 9, ind_2)
    counts_opt = H[(ind_0, ind_1, ind_2)]
    return counts_opt


def get_raw_global_pixelscores_collection(base_dir, collection_id, X_train, ids, global_counts):
    # DEPRECATED!!!!
    """Global pixelscore using global counts."""
    collection_scores = []
    for i in range(IMG_HEIGHT):
        for j in range(IMG_WIDTH):
            pixel = X_train[:, i, j]
            # Counts returned.
            pixel_scores = get_single_pixel_global_counts(pixel)
            collection_scores.append(pixel_scores)
            #print('Processed pixel ({}, {}) of (224,224)'.format(i,j))
    if save_collection_counts:
        collection_counts = np.array(collection_scores).astype('int32')
        filename = base_dir + \
            '/{}/numpy/pixel_counts.npz'.format(collection_id)
        savez_compressed('pixel_counts.npz', X_train)
        print('Saving pixel counts as numpy to {}'.format(filename))
        os.system('mv pixel_counts.npz {}'.format(filename))
    # Now collection_scores are [224 *224 , n_tokens] and represent count numbers.
    # Aggrerage them
    collection_scores = np.mean(collection_scores, axis=0)
    # Now refall that rarity is inversely proportional to frequency.
    collection_scores = 1.0 / collection_scores
    scaler = MinMaxScaler(
        feature_range=(
            PIXELSCORE_SCALING_MIN,
            PIXELSCORE_SCALING_MAX))
    scores = scaler.fit_transform(collection_scores.reshape(-1, 1))
    df = pd.DataFrame()
    df['id'] = ids.astype(np.int64)
    df['PixelScore'] = scores
    print(df.head(10))
    return df


def main(argv):
    if FLAGS.collection_id is not None:
        print('Generating Scres for collection {}'.format(FLAGS.collection_id))
    collection_id = FLAGS.collection_id
    base_dir = FLAGS.base_dir
    X_train, ids = load_collection_numpy(base_dir, collection_id)
    df = get_raw_pixelscores_collection(base_dir = base_dir, collection_id = collection_id, X_train = X_train, ids = ids, save_collection_counts=FLAGS.save_collection_counts, save_pixels_hist=FLAGS.save_pixels_hist, global_score = FLAGS.global_score, use_log_scores = FLAGS.use_log_scores)
    save_collection_scores(
        FLAGS.base_dir, FLAGS.collection_id, FLAGS.results_file, df, FLAGS.global_score, FLAGS.use_log_scores)
    print(
        'Completed Score generation for collection {}'.format(
            FLAGS.collection_id))
    print('Success')


if __name__ == '__main__':
    app.run(main)
