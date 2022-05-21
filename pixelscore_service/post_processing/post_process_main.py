"""Converts all colelctions in the folder to numpy
Runs the subscripts
img_to_numpy.py for all available collections
Update - uses multiprocessing (12 cores) to run img_to_numpy scripts in parallel
over 12 different collections ids at the same time.
"""
import os
import gc
import sys
import numpy as np
from PIL import Image
from absl import app
from absl import flags
import pandas as pd
import multiprocessing as mp

debug_whitelist = ['0xd3cd44f07744da3a6e60a4b5fda1370400ad515b', '0x6206d330d018cfdca00c7e9e210c79d51c6b1d07', '0xc0a2632641449400ec2c54b94fa371c1d85406fc', '0xd08b02a76552fa54d8616f8d42c3cf0de3c0a9ec', '0xc8100dd81e0d8d0901b7b5831e575b03e1489057', '0x892555e75350e11f2058d086c72b9c94c9493d72', '0x8460bb8eb1251a923a31486af9567e500fc2f43f', '0x91a7c7ca17f9f8579ac0a34fa0203f99fa37cf48', '0x80581ea8339d682ca5a71e3f561e3ab0e270398e', '0x35b0ecc952cef736c12a7ef3a830f438f67912b3',
                   '0x0574c34385b039c2bb8db898f61b7767024a9449', '0xbe588550f82aa9207e84d07b305767cf33249dc9', '0x85f06f0dc7ac62f006ab09227e81709b7c39f50c', '0x4f438a2e7d060733f90775566af040576bd441b0', '0x880644ddf208e471c6f2230d31f9027578fa6fcc', '0xb6329bd2741c4e5e91e26c4e653db643e74b2b19', '0x63fa29fec10c997851ccd2466dad20e51b17c8af', '0x2070f7821d9c8ebfdecc1bc981d1308cc0d93843', '0x714a3c939a3664c06e3bb0e8315cefff84526f17', '0x4530ed5907ceb4e14b0550a28e7d300cc773b92e']

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'collection_whitelist',
    '/mnt/disks/ssd/pixelscore_service/whitelists_blacklists/np_ready_ids_8_Apr_2022.csv',
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
    'mode',
    'is_pre_reveal',
    'Mode to run preprocessing')
flags.DEFINE_string(
    'code_path',
    '/mnt/disks/ssd/git/pixelscore_service',
    'Root oath to project with code')
flags.DEFINE_string(
    'scored_collections_whitelist',
    '/mnt/disks/ssd/pixelscore_service/whitelists_blacklists/scored_ids_11_Apr_2022.csv',
    'Scored collections.')
flags.DEFINE_string(
    'merged_scores_file',
    '',
    'File with all scores for all collections merged.')
flags.DEFINE_boolean(
    'use_log_scores',
    False,
    'Whether to use scores as -log(1+prob).')

def run_process(collection_id, base_dir):
    """Single process run of img to numpy function."""
    print('Converting collection {}'.format(collection_id))
    try:
        os.system('python3 {}/pixelscore_service/preprocessing/preprocess_lib.py --collection_id={} --base_dir={} --mode="is_pre_reveal"'.format(
            FLAGS.code_path, collection_id, base_dir))
    except:
        print('Unable to compute pixelscores for collection {}, trying next one'.format(
            collection_id))
    return True


def run_process_is_empty(collection_id, base_dir):
    """Single process run of img to numpy function."""
    print('Converting collection {}'.format(collection_id))
    try:
        os.system('python3 {}/pixelscore_service/preprocessing/preprocess_lib.py --collection_id={} --base_dir={} --mode="is_empty"'.format(
            FLAGS.code_path, collection_id, base_dir))
    except:
        print('Unable to compute pixelscores for collection {}, trying next one'.format(
            collection_id))
    return True


def main(argv):
    base_dir = FLAGS.base_dir
    if FLAGS.collection_whitelist is None:
        print('Collection whitelist not specified.')
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
    # print(whitelist)
    # Scores hist mode is single thresded.
    if FLAGS.mode == 'scores_hist':
        os.system('python3 {}/pixelscore_service/post_processing/post_process_lib.py --base_dir={} --mode="scores_hist"'.format(FLAGS.code_path, base_dir))
    if FLAGS.mode == 'merge_results':
        os.system('python3 {}/pixelscore_service/post_processing/post_process_lib.py --mode="merge_results" --base_dir={} --scored_collections_whitelist={} --merged_scores_file={} --use_log_scores={}'.format(FLAGS.code_path, base_dir, FLAGS.scored_collections_whitelist, FLAGS.merged_scores_file, FLAGS.use_log_scores))
    if FLAGS.mode == 'analyze_merged_scores':
        os.system('python3 {}/pixelscore_service/post_processing/post_process_lib.py --mode="analyze_merged_scores" --base_dir={} --scored_collections_whitelist={} --merged_scores_file={} --use_log_scores={}'.format(FLAGS.code_path, base_dir, FLAGS.scored_collections_whitelist, FLAGS.merged_scores_file, FLAGS.use_log_scores))
    if FLAGS.mode == 'bucketize_scores':
        os.system('python3 {}/pixelscore_service/post_processing/post_process_lib.py --mode="bucketize_scores" --base_dir={} --scored_collections_whitelist={} --merged_scores_file={} --use_log_scores={}'.format(FLAGS.code_path, base_dir, FLAGS.scored_collections_whitelist, FLAGS.merged_scores_file, FLAGS.use_log_scores))
    if FLAGS.mode == 'plot_rarest_colors':
        os.system('python3 {}/pixelscore_service/post_processing/post_process_lib.py --mode="plot_rarest_colors" --base_dir={} --scored_collections_whitelist={} --merged_scores_file={} --use_log_scores={}'.format(FLAGS.code_path, base_dir, FLAGS.scored_collections_whitelist, FLAGS.merged_scores_file, FLAGS.use_log_scores))
    if FLAGS.mode == 'plot_rarest_collections':
        os.system('python3 {}/pixelscore_service/post_processing/post_process_lib.py --mode="plot_rarest_collections" --base_dir={} --scored_collections_whitelist={} --merged_scores_file={} --use_log_scores={}'.format(FLAGS.code_path, base_dir, FLAGS.scored_collections_whitelist, FLAGS.merged_scores_file, FLAGS.use_log_scores))
    if FLAGS.mode == 'plot_log':
        os.system('python3 {}/pixelscore_service/post_processing/post_process_lib.py --mode="plot_log" --base_dir={} --scored_collections_whitelist={} --merged_scores_file={} --use_log_scores={}'.format(FLAGS.code_path, base_dir, FLAGS.scored_collections_whitelist, FLAGS.merged_scores_file, FLAGS.use_log_scores))
    if FLAGS.mode == 'plot_bucketized_hist':
        os.system('python3 {}/pixelscore_service/post_processing/post_process_lib.py --mode="plot_bucketized_hist" --base_dir={} --scored_collections_whitelist={} --merged_scores_file={} --use_log_scores={}'.format(FLAGS.code_path, base_dir, FLAGS.scored_collections_whitelist, FLAGS.merged_scores_file, FLAGS.use_log_scores))
    # Define parallel execution.
    if FLAGS.mode == 'is_pre_reveal' or FLAGS.mode == 'is_empty':
        pool = mp.Pool(mp.cpu_count())
        if FLAGS.mode == 'is_pre_reveal':
            results = [pool.apply_async(run_process, args=(
                collection_id, base_dir)) for collection_id in whitelist]
        if FLAGS.mode == 'is_empty':
            results = [pool.apply_async(run_process_is_empty, args=(
                collection_id, base_dir)) for collection_id in whitelist]
        pool.close()
        pool.join()
        print(results)
    print('Success')


if __name__ == '__main__':
    app.run(main)
