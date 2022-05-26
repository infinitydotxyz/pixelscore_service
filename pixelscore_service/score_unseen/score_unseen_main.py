""" Computes global pixelscore and pixelrank for any (unseen) collection.

Use case:


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
from datetime import timezone
import datetime

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'collection_id',
    '0x9a534628b4062e123ce7ee2222ec20b86e16ca8f',
    'Collection id.')
flags.DEFINE_string(
    'base_dir',
    '/mnt/disks/additional-disk/data',
    'Local base directory containing images.')
flags.DEFINE_string(
    'mode',
    'is_pre_reveal',
    'Mode to run preprocessing')
flags.DEFINE_string(
    'code_path',
    '/mnt/disks/ssd/git/pixelscore_service',
    'Root path to project with code')
flags.DEFINE_string(
    'merged_scores_file',
    '/mnt/disks/additional-disk/merged_scores/log_merged_scores_10_May_2022.csv',
    'File with all scores for all collections merged.')
flags.DEFINE_string(
    'score_unseen_logs',
    '/mnt/disks/additional-disk/raw_logs/score_unseen',
    'Base log dir for score unseen scripts.')
flags.DEFINE_string(
    'hist_dir',
    '/mnt/disks/additional-disk/histograms',
    'Dir to save pixel histograms.')

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
        df.to_csv(filename)
    print('File with scoring results stats created {}'.format(filename))
    return filename

def make_logdir(mode):
    """Makes proper log dir, separate folder per day."""
    today = str(datetime.date.today())
    if mode == 'img_to_numpy':
        logdir = FLAGS.score_unseen_logdir + '/img_to_numpy/' + today
    if mode == 'raw_pixelscore_main':
        logdir = FLAGS.score_unseen_logdir + '/raw_pixelscore_main/' + today
    if mode == 'scoring_results':
        logdir = FLAGS.score_unseen_logdir + '/scoring_results/' + today
    if not os.path.exists(logdir):
        os.run('mkdir ' + logdir)
    return logdir

def binned_pixelscore_unseen(base_dir, collection_id):
    """Computes binned pixel score on unseen nft collection.
    
    Uses saved bin edges from latest binarizing of merged log pixelscores.
    """
    filename = FLAGS.post_processing_dir + '/pixelscore_bins/pixelscore_bins.npz'
    bins = np.load(filename)['arr_0']
    path = base_dir + '/{}'.format(collection_id) + '/pixelscore'
    filename_score = path + '/unseen_global_raw_pixelscore_log.csv'
    df = pd.read_csv(filename_score)
    """
    bin examples:
    [6.81321656e-05 4.01425345e-03 6.83831863e-03 9.67101284e-03
    1.30270915e-02 1.62251006e-02 1.94639000e-02 2.28212615e-02
    2.88390565e-02 4.67239450e-02 1.31064342e+00]
    """
    #Then pandas qcut (cut with the bins above and labels)
    bin_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    df['bin_pixelScore'], bins = pd.cut(df['pixelScore'],
                                   bins=bins,
                                   labels=bin_labels,
                                   retbins = True)
    print(bins)
    sys.exit()
    df.to_csv(filename_score)
    return True
    


def main(argv):
    base_dir = FLAGS.base_dir
    collection_id = FLAGS.collection_id
    # Convert collection to numpy (if not converted yet)
    print('START convert img to numpy for collection {}'.format(collection_id))
    img_to_numpy_logs_dir = make_logdir(mode = 'img_to_numpy')
    cmd='python3 {}/pixelscore_service/within_collection_score/img_to_numpy.py " \
        "--collection_id={} " \
        "--base_dir={} " \
        "--img_to_numpy_logs_dir={}'.format(
            FLAGS.code_path,
            collection_id,
            base_dir,
            img_to_numpy_logs_dir)
    os.run(cmd)
    print('FINISH convert img to numpy for collection {}'.format(collection_id))
    # Data cleaning?
    # Compute raw scores.
    print('START computing pixelscores for collection {}'.format(collection_id))
    results_file = create_results_file(make_logdir(mode = 'scoring_results'))
    raw_logs_dir = make_logdir(mode = 'raw_pixelscore_main')
    save_pixels_hist = False
    global_score = True
    use_log_scores = True
    cmd='python3 {}/pixelscore_service/within_collection_score/raw_pixelscore_main.py " \
        "--collection_id={} " \
        "--base_dir={} " \
        "--results_file={} " \
        "--hist_dir={} " \
        "--save_pixels_hist={} " \
        "--global_score={} " \
        "--raw_logs_dir={} " \
        "--use_log_scores={}'.format(
            FLAGS.code_path,
            collection_id,
            base_dir,
            results_file, 
            FLAGS.hist_dir,
            save_pixels_hist,
            global_score,
            raw_logs_dir,
            use_log_scores)
    os.run(cmd)
    print('FINISH computing pixelscores for collection {}'.format(collection_id))
    # Compute binned pixel scores.
    print('START computing binned pixelscores for collection {}'.format(collection_id))
    binned_pixelscore_unseen(base_dir, collection_id)
    print('FINISH computing binned pixelscores for collection {}'.format(collection_id))
    print('SUCCESS')
if __name__ == '__main__':
    app.run(main)