"""Train DNN model for particular collection.

Init from EfficientNet imagenet weights.

example run:
python3 pixelscore_service/within_collection_score/train_model.py
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

from keras.models import Sequential  # Model type to be used

# Types of layers to be used in our model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import GlobalAveragePooling2D

from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from keras import backend as K
from numpy import savez_compressed

# Functions for loading model and scoring one collection of NFTs.
FLAGS = flags.FLAGS
N_CLASSES = 10
EPOCHS = 27
# Num classes to binarize ground teruth rarity score.
GROUND_TRUTH_N_CLASSES = 10
N_CLASSES_STANDARD_MODEL = 1000
# Used for debug.
# Read only MAX_EXAMPLES from collection, to always read full collection
# set to 100K
MAX_EXAMPLES = 100000
EFFICIENTNET_IMAGE_SIZE = 224
# Number of bins for pixel rarity score, must be less than collection size.
PIXEL_SCORE_BINS = 100
BATCH_SIZE = 32
# Stop model training when accuracy reaches threshold.
ACCURACY_THRESHOLD = 0.4
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
    False,
    'Whether to use model checkpoint transfer learned for the given collection. If False, base EfficientNet with imagenet weights is used.')

# Callbacks.

class AccThresholdCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > ACCURACY_THRESHOLD):
            print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))
            self.model.stop_training = True

def tensorboard_callback(directory, name):
    log_dir = directory + "/" + name
    t_c = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    return t_c


def reduce_on_plateau():
    r_c = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
            factor = 0.1,
            patience = 2,
            verbose = 1,
            cooldown = 1)
    return r_c


def model_checkpoint(directory, name):
    log_dir = directory + "/" + name
    m_c = tf.keras.callbacks.ModelCheckpoint(filepath=log_dir,
                                             monitor="val_accuracy",
                                             save_best_only=True,
                                             save_weights_only=True,
                                             verbose=1)
    return m_c


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


def load_labels(base_dir, collection_id, ids):
    """Loads labels based on ground-truth rarity.score for a specific nft collection.

    Args:
      base_dir: Base data directory on the current vm e.g. /mnt/disks/ssd/data
      collection_id: collection address e.g. '0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d'
    Returns:
      y_train: np array with labels for  entire collection e.g. [collection_length]
    """
    # Load labels.
    path = base_dir + '/{}'.format(collection_id) + '/numpy'
    filename = path + '/labels.npz'
    y_train = np.load(filename)['arr_0']
    print('Loading labels as numpy from {}'.format(filename))
    return y_train


def save_collection_scores(base_dir, collection_id, df):
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
    df.to_csv('pixelscore.csv')
    print('Saving layers as numpy to {}'.format(filename))
    os.system('sudo mv pixelscore.csv {}'.format(filename))
    return True


def load_metadata():
    df = pd.read_csv(METADATA_FILE, header=None, low_memory=False)
    df.columns = [
        'id',
        'rarityScore',
        'rarityRank',
        'url',
        'contenttype',
        'length']
    # print(df.head(10)['rarityScore'])
    # print(df.describe())
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


def create_architecture_small_cnn():
    """Small cnn from scratch.."""
    model = Sequential()
    model.add(
        keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(
                EFFICIENTNET_IMAGE_SIZE, EFFICIENTNET_IMAGE_SIZE, 3)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(N_CLASSES, activation=('softmax')))
    model.summary()
    return model


def create_architecture_regression():
    """Small cnn from scratch.."""
    model = Sequential()
    model.add(
        keras.layers.Dense(
            28,
            activation='relu',
            input_shape=(
                EFFICIENTNET_IMAGE_SIZE,
                EFFICIENTNET_IMAGE_SIZE,
                3)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(N_CLASSES, activation=('softmax')))
    model.summary()
    return model


def create_architecture():
    """Feature Extractor on top of EffNet init from imagenet.

    Recommended lr = TBD
    ok to train on CPU for 10 epochs takes 1h.
    """
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(
            EFFICIENTNET_IMAGE_SIZE,
            EFFICIENTNET_IMAGE_SIZE,
            3),
        weights="imagenet",
        classes=N_CLASSES_STANDARD_MODEL)
    base_model.trainable = False

    # Now trainable layers.
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    # model.add(Dense(128,activation=('relu')))
    # model.add(Dense(N_CLASSES,activation=('softmax')))
    model.add(Flatten())
    # Adding the Dense layers along with activation and batch normalization
    model.add(Dense(1024, activation=('relu'), input_dim=512))
    model.add(Dense(512, activation=('relu')))
    model.add(Dense(256, activation=('relu')))
    # model.add(Dropout(.3))
    model.add(Dense(128, activation=('relu')))
    # model.add(Dropout(.2))
    model.add(Dense(N_CLASSES, activation=('softmax')))
    # Model summary
    model.summary()
    return model


def load_checkpoint(checkpoints_dir, collection_id):
    tf.keras.models.load_model(checkpoint)
    # Check its architecture
    print(new_model.summary())
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
    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[3].output])
    get_last_layer_output = K.function([model.layers[0].input],
                                       [model.layers[-1].output])
    layer_output = get_last_layer_output([img_preprocessed])[0]
    print('Obtained Layer output with shape: {}'.format(layer_output.shape))
    #print(decode_predictions(prediction, top=3)[0])
    del img
    gc.collect()
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


def get_scores_collection(X_train, ids):
    # Score is created as new column in df.
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
    df = pd.DataFrame()
    df['id'] = ids
    df['PixelScore'] = scores
    print('Scores {}'.format(scores))
    df = pd.DataFrame()
    df['id'] = ids
    df['PixelScore'] = scores

    print(df.head(10))
    (unique, counts) = np.unique(Xt, return_counts=True)
    # Counts of each bin value.
    frequencies = np.asarray((unique, counts)).T
    print(frequencies)
    Xt = Xt.flatten()
    #plt.hist(Xt, bins = HIST_BINS)
    #plt.plot(unique, counts)
    # plt.show()
    # Score of each collection element is the sum (average) of bins?
    return df


def train_model(base_dir, collection_id, results_file, model, X_train, y_train):
    """Trains (fine-tunes) base model on the given collection."""
    tf_logs = base_dir + '/{}'.format(collection_id) + '/tf_logs'
    if not os.path.exists(tf_logs):
        os.system('sudo mkdir {}'.format(tf_logs))
    os.system('sudo chmod -R ugo+rwx {}'.format(tf_logs))
    # Compile.
    # Recommended lr = 0.001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'])

    steps_per_epoch = len(y_train) // BATCH_SIZE
    validation_steps = len(y_train) // BATCH_SIZE
    # Train.
    callbacks_ = [tensorboard_callback(tf_logs, "model"),
                  model_checkpoint(tf_logs, "model.ckpt"),
                  AccThresholdCallback(),
                  reduce_on_plateau()]
    hist = model.fit(
        x=X_train, y=y_train,
        epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
        validation_data=(X_train, y_train), callbacks=callbacks_).history
    model_epochs = len(hist['val_accuracy'])
    model_accuracy = hist['val_accuracy'][-1]
    update_results(results_file,
        {'collection_id': collection_id,
        'model_accuracy': model_accuracy,
        'model_epochs': model_epochs,
        })
    model.save(tf_logs + '/model')
    return model


def main(argv):
    if FLAGS.collection_id is not None:
        print('Training model for collection {}'.format(FLAGS.collection_id))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    model = create_architecture()
    X_train, ids = load_collection_numpy(FLAGS.base_dir, FLAGS.collection_id)
    y_train = load_labels(FLAGS.base_dir, FLAGS.collection_id, ids)
    y_train_cat = tf.keras.utils.to_categorical(y_train, N_CLASSES)
    trained_model = train_model(
        FLAGS.base_dir,
        FLAGS.collection_id,
        FLAGS.results_file,
        model,
        X_train,
        y_train_cat)
    # Save checkpoint.
    print(
        'Completed model training for collection {}'.format(
            FLAGS.collection_id))
    print('Success')


if __name__ == '__main__':
    app.run(main)
