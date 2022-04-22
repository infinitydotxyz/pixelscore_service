"""Analysis and post processinf of scoring results

Use cases
1) Find idsfrom metadata file that were not cores due to missing URLS.
2) For each collection total_ids, how many were scored, how many not scored
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
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from keras import backend as K
from numpy import savez_compressed

# Functions for loading model and scoring one collection of NFTs.

N_CLASSES = 10
# Num classes to binarize ground truth rarity score.
GROUND_TRUTH_N_CLASSES = 10
# Default classes in pre-trained EfficientNet.
N_CLASSES_STANDARD_MODEL = 1000
# Read only MAX_EXAMPLES from collection, set to 100K to read everything.
MAX_EXAMPLES = 100000
# Image dimension for EfficientNet.
EFFICIENTNET_IMAGE_SIZE = 224
# Number of bins for pixel rarity score, must be less than collection size.
PIXEL_SCORE_BINS = 10
# Params for fine tuning EfficientNet on groundtruth rarityScore.
EPOCHS = 10
BATCH_SIZE = 32
LR = 0.001

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'collection_id',
    '0x9a534628b4062e123ce7ee2222ec20b86e16ca8f',
    'Collection id.')
flags.DEFINE_string(
    'base_dir',
    '/mnt/disks/ssd/data',
    'Local base directory containing images.')
flags.DEFINE_string(
    'checkpoints_dir',
    '/mnt/disks/ssd/checkpoints',
    'Local dire where model checkpoints for each collection are stored.')
flags.DEFINE_boolean(
    'use_checkpoint',
    False,
    'Whether to use model checkpoint transfer learned for the given collection. If False, base EfficientNet with imagenet weights is used.')
flags.DEFINE_string(
    'collection_whitelist',
    '/mnt/disks/ssd/pixelscore_service/within_collection_score/blacklist_1.csv',
    'Whitelist  of collection_id to run analysis on.')

def find_not_scored_ids(base_dir, whitelist):
    """Examples ids and % examples not scored.
    
    images are matched by token ids, 
    which is first column in metadata, 
    reided image filename and tokenId column in pixelscore file.
    """
    for col_id in whitelist:
        pixelscore_f = base_dir + '/{}/pixelscore/pixelscore.csv'.format(col_id)
        metadata_f = base_dir + '/{}/metadata/metadata.csv'.format(col_id) 
        try:
            pixelscore_df = pd.read_csv(pixelscore_f)
            metadata_df = pd.read_csv(metadata_f, header=None, low_memory=False)
            metadata_df.columns = ['id', 'rarityScore', 'rarityRank', 'url']
        except:
            print('NO pixelscore file for collection {}'.format(col_id))
            continue
        
        # Sanity check
        len_metadata = len(metadata_df['id'].values)
        len_pixelscore = len(pixelscore_df['tokenId'].values)
        if len_metadata != len_pixelscore:
            print('Warning: len metadata {} not equal len pixelscore {}'.format(len_metadata, len_pixelscore))
        
        # Count {total, scored, missing}
        total_ids = metadata_df['id'].values 
        df_nan = pixelscore_df[pixelscore_df['rarityScore'].isna()]
        not_scored_ids = df_nan['tokenId'].values
        scored_ids = list(set(total_ids) - set(not_scored_ids))
        # Count
        n_total = len(total_ids)
        n_scored = len(scored_ids)
        n_not_scored = len(not_scored_ids)
        percent_scored = 100.0 * float(n_scored) / float(n_total)
        percent_not_scored = 100.0 * float(n_not_scored) / float(n_total)
        if percent_not_scored >0:
            print('Not scored ids percent {} in collection {}'.format(percent_not_scored, col_id))
        # Record this data.
        not_scored_df = pd.DataFrame()
        not_scored_df['notScoredIds'] = not_scored_ids
        stats_df = pd.DataFrame.from_dict({
            'nTotalTokens': [n_total],
            'nScoredTokens': [n_scored],
            'nNotScoredTokens': [n_not_scored],
            'percentScoredTokens': [percent_scored],
            'percentNotScoredTokens': [percent_not_scored]
            })
        print(not_scored_df)
        print(stats_df)
        not_scored_ids_path = base_dir + '/{}/pixelscore/not_scored_ids.csv'.format(col_id)
        stats_path = base_dir + '/{}/pixelscore/stats.csv'.format(col_id)
        not_scored_df.to_csv('not_scored_ids.csv')
        stats_df.to_csv('stats.csv')
        os.system('sudo mv not_scored_ids.csv {}'.format(not_scored_ids_path))
        os.system('sudo mv stats.csv {}'.format(stats_path))
    return True

def main(argv):
    df = pd.read_csv(FLAGS.collection_whitelist)
    whitelist = df['collection_id'].values
    find_not_scored_ids(FLAGS.base_dir, whitelist)
    print('Completed scoring post-processing.')
    print('Success')


if __name__ == '__main__':
    app.run(main)
