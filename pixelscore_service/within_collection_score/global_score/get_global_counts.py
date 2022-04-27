"""Produces aggregated global counts of each pixel bin across all collections."""

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

N_BINS_R = 10
N_BINS_G = 10
N_BINS_B = 10
IMG_HEIGHT = 224
IMG_WIDTH = 224

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'base_dir',
    '/mnt/disks/ssd/data',
    'Local base directory containing images, resized, metadata, etc.')

flags.DEFINE_string(
    'hist_dir',
    '/mnt/disks/additional-disk/histograms',
    'Dir to save histograms.')

flags.DEFINE_string(
    'global_hist_dir',
    '/mnt/disks/additional-disk/global_hist',
    'Dir to save global histograms.')

flags.DEFINE_string(
    'global_hist_shards_dir',
    '/mnt/disks/additional-disk/global_histograms/shards',
    'Dir to save global histogram shards.')

flags.DEFINE_string(
    'collection_whitelist',
    '/mnt/disks/ssd/pixelscore_service/whitelists_blacklists/global_hist_ready_10_Apr_2022.csv',
    'Path to .csv file with whitelist of collection_id')

flags.DEFINE_boolean(
    'use_whitelist',
    True,
    'Whether to use collections whitelist or score all colelctions found in base_dir.')

flags.DEFINE_boolean(
    'use_blacklist',
    False,
    'Collections in blacklist wont be scored.')

flags.DEFINE_string(
    'blacklist',
    '/mnt/disks/ssd/pixelscore_service/within_collection_score/blacklist_1.csv',
    'Path to .csv file with blacklist of collection_id')

flags.DEFINE_string(
    'results_dir',
    '/mnt/disks/ssd/pixelscore_service/within_collection_score/scoring_results',
    'Folder that stores csv with scoring results {col_id, time_finished, model_acc, corr}.')


def get_global_counts(base_dir, whitelist):
    """Get global counts from all collections."""
    global_count = np.zeros(
        [IMG_HEIGHT * IMG_WIDTH, N_BINS_R * N_BINS_G * N_BINS_B])
    for id in whitelist:
        print('Processing counts for collection {}'.format(id))
        filename = base_dir + \
            '/{}/numpy/pixel_counts.npz'.format(collection_id)
        counts = np.load(filename)['arr_0']
        global_count = counts + global_count
    return global_count


def save_global_count(base_dir, whitelist, global_count):
    """Saves global count to dir."""
    filename = base_dir + '/global_count/global_counts.npz'
    savez_compressed('global_counts.npz', global_count)
    os.system('sudo mv global_counts.npz {}'.format(filename))
    return True


def save_global_hist(base_dir, whitelist, global_hist):
    """Saves global count to dir."""
    filename = FLAGS.global_hist_dir + '/global_hist.npz'
    np.savez_compressed(filename, global_hist)
    return True


def get_global_hist(base_dir, whitelist):
    """Computes global hist for each pixel by suming over collections in the whitelist.

    Global histogram shoul dbe contained in a singe .npz format

    pixels_hist.npz has dimensions (50176, 10, 10, 10), where 50176 is total pixels.
    """
    # Pre-load histograms for all collecitons:
    id_to_hist = dict()
    whitelist_post = []
    for id in whitelist:
        try:
            filename = FLAGS.hist_dir + '/{}/pixels_hist.npz'.format(id)
            id_to_hist[id] = np.load(filename)['arr_0']
            whitelist_post.append(id)
            print('Successfully loaded hist for collection {}'.format(id))
        except:
            print('Failed to load hist for collection {}'.format(id))

    # Sum over collections for every pixel
    super_global_hist = []
    for i in range(IMG_HEIGHT):
        for j in range(IMG_WIDTH):
            print('Processing pixel {}, {}'.format(i, j))
            global_pixel_hist = np.zeros([N_BINS_R, N_BINS_G, N_BINS_B])
            # get proper indices
            index = i * IMG_HEIGHT + j
            for id in whitelist_post:
                global_pixel_hist = global_pixel_hist + id_to_hist[id][index]
            super_global_hist.append(global_pixel_hist)
    return np.array(super_global_hist), whitelist_post

# Yield successive n-sized
# chunks from l.


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def save_global_hist_shards(global_hist, whitelist_post, shard_id):
    """Saves global hist shard with corresponsing whitelist."""
    shard_path = FLAGS.global_hist_shards_dir + '/{}'.format(shard_id)
    if not os.path.exists(shard_path):
        os.system('sudo mkdir {}'.format(shard_path))
        os.system('sudo chmod ugo+rwx {}'.format(shard_path))
    filename_hist = FLAGS.global_hist_shards_dir + \
        '/{}/pixels_hist.npz'.format(shard_id)
    np.savez_compressed(filename_hist, global_hist)
    filename_whitelist = FLAGS.global_hist_shards_dir + \
        '/{}/whitelist.csv'.format(shard_id)
    df = pd.DataFrame()
    df['collection_id'] = whitelist_post
    df.to_csv(filename_whitelist)


def get_global_hist_sharded(base_dir, whitelist):
    """Processes in chunks of 500 collections."""
    SHARD_SIZE = 250
    whitelist_shards = list(divide_chunks(whitelist, SHARD_SIZE))
    shard_count = 0
    for whitelist_ in whitelist_shards:
        global_hist, whitelist_post = get_global_hist(base_dir, whitelist_)
        save_global_hist_shards(global_hist, whitelist_post, shard_count)
        shard_count += 1
    return True


def sum_global_hist():
    """Sums all sharded global hist to get the final one."""
    print('Summing up global hist.')
    shards = os.listdir(FLAGS.global_hist_shards_dir)
    shard = shards[0]
    global_hist = np.load(FLAGS.global_hist_shards_dir +
                          '/{}/pixels_hist.npz'.format(shard))['arr_0']
    for shard in shards[1:]:
        hist = np.load(FLAGS.global_hist_shards_dir +
                       '/{}/pixels_hist.npz'.format(shard))['arr_0']
        global_hist = global_hist + hist
        del hist
        gc.collect()
    return global_hist


def main(argv):
    if FLAGS.collection_whitelist is None:
        print('Collection whitelist not specified.')
    #results_file = create_results_file(FLAGS.results_dir)
    if FLAGS.use_whitelist:
        df = pd.read_csv(FLAGS.collection_whitelist)
        whitelist = df['collection_id'].values
    else:
        whitelist = os.listdir(FLAGS.base_dir)
    # Subtract blacklist form whitelist
    if FLAGS.use_blacklist and FLAGS.blacklist is not None:
        df_black = pd.read_csv(FLAGS.blacklist)
        blacklist = df_black['collection_id'].values
        whitelist = set(whitelist) - set(blacklist)
        whitelist = list(whitelist)
    print('Using whitelist of size {} with the following:'.format(len(whitelist)))
    print(whitelist)
    # Now get the global counts and hist.
    get_global_hist_sharded(FLAGS.base_dir, whitelist)
    global_hist = sum_global_hist()
    save_global_hist(FLAGS.base_dir, whitelist, global_hist)
    print('Success')


if __name__ == '__main__':
    app.run(main)
