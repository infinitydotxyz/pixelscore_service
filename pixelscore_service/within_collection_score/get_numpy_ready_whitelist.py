"""Gets a list of collections, for which pixels.npz and lablels.npz files are created
and can be opened without errors.

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

debug_whitelist = ['0xd3cd44f07744da3a6e60a4b5fda1370400ad515b', '0x6206d330d018cfdca00c7e9e210c79d51c6b1d07', '0xc0a2632641449400ec2c54b94fa371c1d85406fc', '0xd08b02a76552fa54d8616f8d42c3cf0de3c0a9ec', '0xc8100dd81e0d8d0901b7b5831e575b03e1489057', '0x892555e75350e11f2058d086c72b9c94c9493d72', '0x8460bb8eb1251a923a31486af9567e500fc2f43f', '0x91a7c7ca17f9f8579ac0a34fa0203f99fa37cf48', '0x80581ea8339d682ca5a71e3f561e3ab0e270398e', '0x35b0ecc952cef736c12a7ef3a830f438f67912b3', '0x0574c34385b039c2bb8db898f61b7767024a9449', '0xbe588550f82aa9207e84d07b305767cf33249dc9', '0x85f06f0dc7ac62f006ab09227e81709b7c39f50c', '0x4f438a2e7d060733f90775566af040576bd441b0', '0x880644ddf208e471c6f2230d31f9027578fa6fcc', '0xb6329bd2741c4e5e91e26c4e653db643e74b2b19', '0x63fa29fec10c997851ccd2466dad20e51b17c8af', '0x2070f7821d9c8ebfdecc1bc981d1308cc0d93843', '0x714a3c939a3664c06e3bb0e8315cefff84526f17', '0x4530ed5907ceb4e14b0550a28e7d300cc773b92e']

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'collection_whitelist',
    '',
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
    False,
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
    'whitelists_blacklists_dir',
    '/mnt/disks/ssd/pixelscore_service/whitelists_blacklists',
    'Path to dir will all whitelists and blacklists etc.')
def run_process(collection_id, base_dir):
    """Single process run of img to numpy function."""
    print('Converting collection {}'.format(collection_id))
    try:
        os.system('python3 pixelscore_service/within_collection_score/img_to_numpy.py --collection_id={} --base_dir={}'.format(collection_id, base_dir))
    except:
        print('Unable to compute pixelscores for collection {}, trying next one'.format(collection_id))
    return True

def get_numpy_ready_collections(base_dir, whitelist):
    """Return ids for which numpy already generated."""
    ready_list = []
    not_ready_list = []
    for col_id in whitelist:
        try:
            x_path = base_dir + '/{}/numpy/pixels.npz'.format(col_id)
            y_path = base_dir + '/{}/numpy/labels.npz'.format(col_id)
            exists_y = os.path.exists(x_path)
            exists_x = os.path.exists(y_path)
            x = np.load(x_path)['arr_0']
            y = np.load(y_path)['arr_0']
            ready_list.append(col_id)
            del x
            del y
            gc.collect()
            print('NP arrays verified for collection {}'.format(col_id))
        except:
            print('Unable to load np for collection {}'.format(col_id))
            not_ready_list.append(col_id)
    print('Numpy ready whitelist of size {} with the following:'.format(len(ready_list)))
    return ready_list, not_ready_list

def save_numpy_ready_ids(base_dir, ready_list, not_ready_list):
    """Save ids to memo."""
    # Ready.
    df = pd.DataFrame()
    df['collection_id'] = ready_list
    filename = FLAGS.whitelists_blacklists_dir = '/np_ready_ids_8_Apr_2022.csv' 
    df.to_csv(filename)
    # Not ready.
    df = pd.DataFrame()
    df['collection_id'] = not_ready_list
    filename = FLAGS.whitelists_blacklists_dir = '/np_not_ready_ids_8_Apr_2022.csv' 
    df.to_csv(filename)

    return True

def main(argv):
    base_dir = FLAGS.base_dir
    if FLAGS.collection_whitelist is None:
        print('Collection whitelist not specified.')
    if FLAGS.use_whitelist:    
        df = pd.read_csv(FLAGS.collection_whitelist)
        whitelist = df['colelction_id'].values
    else:
        whitelist = os.listdir(FLAGS.base_dir)
    # Subtract blacklist form whitelist
    if FLAGS.use_blacklist and FLAGS.blacklist is not None:
        df_black = pd.read_csv(FLAGS.blacklist)
        blacklist = df_black['collection_id'].values
        whitelist = set(whitelist) - set(blacklist)
        whitelist = list(whitelist)
    print('Using whitelist of size {} with the following:'.format(len(whitelist)))
    #print(whitelist)

    ready_list, not_ready_list = get_numpy_ready_collections(base_dir, whitelist)
    save_numpy_ready_ids(base_dir, ready_list, not_ready_list)
    print('Success')


if __name__ == '__main__':
    app.run(main)
