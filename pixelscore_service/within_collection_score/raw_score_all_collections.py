"""Scores all collections in the whitelist.

Applies raw_pixelscore_main.py in parallel chunks to all collections from thd whitelist.

Input:
.csv file with collections whitelist, must have column 'colelction_id'

Output:

Saves raw pixelscore in .csv file base_dir/<collection_id>/pixelscore/raw_pixelscore.csv
"""
import os
import gc
import sys
import numpy as np
from PIL import Image
from absl import app
from absl import flags
import pandas as pd
from datetime import timezone
import datetime
import multiprocessing as mp

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'collection_whitelist',
    '/mnt/disks/ssd/pixelscore_service/whitelists_blacklists/global_hist_ready_10_Apr_2022.csv',
    'Path to .csv file with whitelist of collection_id')
flags.DEFINE_string(
    'collection_id',
    '0x9a534628b4062e123ce7ee2222ec20b86e16ca8f',
    'Collection id.')
flags.DEFINE_string(
    'base_dir',
    '/mnt/disks/ssd/data',
    'Local base directory containing images.')
flags.DEFINE_boolean(
    'use_checkpoint',
    True,
    'Whether to use model checkpoint transfer learned for the given collection. If False, base EfficientNet with imagenet weights is used.')
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
flags.DEFINE_string(
    'code_path',
    '/mnt/disks/ssd/git/pixelscore_service',
    'Root oath to project with code')
flags.DEFINE_string(
    'hist_dir',
    '/mnt/disks/additional-disk/histograms',
    'Dir to save pixel histograms.')
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
def get_numpy_ready_collections(base_dir, whitelist):
    """Return ids for which numpy already generated."""
    ready_list = []
    for col_id in whitelist:
        x_path = base_dir + '/{}/numpy/pixels.npz'.format(col_id)
        y_path = base_dir + '/{}/numpy/labels.npz'.format(col_id)
        exists_y = os.path.exists(x_path)
        exists_x = os.path.exists(y_path)
        if exists_y and exists_x:
            ready_list.append(col_id)
    print('Numpy ready whitelist of size {} with the following:'.format(len(ready_list)))
    return ready_list


def create_results_file(results_dir):
    """Makes empty results file for the current scoring round.

    Results file should contain:

    collection_id (can be repeated)
    finish_timestamp
    model_acc
    model_epochs
    corr_rarityScore
    corr_rarityRank

    """
    dt = datetime.datetime.now(timezone.utc)
    utc_time = dt.replace(tzinfo=timezone.utc)
    utc_timestamp = int(utc_time.timestamp())
    filename = results_dir + '/{}_results.csv'.format(utc_timestamp)
    if not os.path.exists(filename):
        df = pd.DataFrame(columns=[
            'collection_id',
            'model_accuracy',
            'model_epochs',
            'corr_rarityScore',
            'corr_rarityRank'])
        df.to_csv('results.csv')
        os.system('mv results.csv {}'.format(filename))
    print('File with scoring results stats created {}'.format(filename))
    return filename


def run_process(collection_id, base_dir, results_file):
    """Single process run of getting raw pixel scores function."""
    print('Start computing RAW pixelscores for collection {}'.format(collection_id))
    try:
        os.system('python3 {}/pixelscore_service/within_collection_score/raw_pixelscore_main.py --collection_id={} --base_dir={} --results_file={} --hist_dir={} --save_pixels_hist={} --global_score={} --raw_logs_dir={}'.format(
            FLAGS.code_path, collection_id, base_dir, results_file, FLAGS.hist_dir, FLAGS.save_pixels_hist, FLAGS.global_score, FLAGS.raw_logs_dir))
    except:
        print('Unable to compute pixelscores for collection {}, trying next one'.format(
            collection_id))
    return True


def main(argv):
    if FLAGS.collection_whitelist is None:
        print('Collection whitelist not specified.')
    results_file = create_results_file(FLAGS.results_dir)
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

    # Define parallel execution.
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(run_process, args=(
        collection_id, FLAGS.base_dir, results_file)) for collection_id in whitelist]
    pool.close()
    pool.join()
    print(results)

    print('Success. Score all collections complete.')


if __name__ == '__main__':
    app.run(main)
