# pixelscore_service

## Install Deps

```sh
sudo apt-get install build-essential gfortran libatlas-base-dev python-pip python-dev
sudo apt-get install pkg-config
sudo pip install --upgrade pip
sudo pip3 install --upgrade pip3
sudo apt-get install gfortran
sudo apt-get install libatlas-base-dev
pip3 install -U pip
pip3 install -U tensorflow==2.8.0
pip3 install -r requirements.txt
```

## Make proper dir structure

base_dir, typically /mnt/disks/ssd/data or /mnt/disks/ssd/data_backup  

base_dir contains a folder per collection_id e.g.
/mnt/disks/ssd/data/0x9a534628b4062e123ce7ee2222ec20b86e16ca8f  

each colelction folder contrains the following:  
metadata -  a .csv file with ground truth rarityScores (not pixel scores)  
numpy - images and labels in np format (X_train, y_train). 
resized - raw nft images 224x224  
tf_logs - model checkpoint trained on the given collection, write access must be given to tf_logs  
pixelscore - a .csv file with newly computed pixelscores
raw_pixelscore - a .csv file with newly computed raw pixelscores (without DNN, only based on pixels).
hist.png - histgram with pixel scores for this collection  

## Workflow

### Scrape collections

With external node.js scripts, collect resized 224-by-224 NFT images for a set of collections and save them in base_dir/<collection_id>/resized

### Preprocessing

Some scraped collections can have corrupt data and shoud not be used for further pixelscore computations, data corruptions include:
/*:
  - Pre-reveal (all images are the same copies of pre-reveal thumbnail)
  - Empty (no images)
 */
To get the list of collections that have corrupt data, run the following (can replace with custom logs dir)
```sh
python3 pixelscore_service/preprocessing/preprocess_main.py --mode="is_pre_reveal" --pre_reveal_logs_dir="/mnt/disks/additional-disk/raw_logs/tmp_preprocess/is_pre_reveal"
```
```sh
python3 pixelscore_service/preprocessing/preprocess_main.py --mode="is_empty" --is_empty_logs_dir="/mnt/disks/additional-disk/raw_logs/tmp_preprocess/is_empty"
```
The scripts run paralell across collections and the output will be the list of empty files in pre_reveal_logs_dir or is_empty_logs_dir, where the name of each file is the <collection_id>. These colelction ids can be copied to blacklist and excluded from further processing.

### Convert images to numpy arrays

The resized 224-by-224 NFT images scraped from the previous step need to be converted to np.array for faster subsequent processing. This is done in parallel (one collection per process). It is recommended to put the ids of non-corrupt scraped collections after the preprocessing stage into a whitelist i.e. pixelscore_service/whitelists_blacklists/np_ready_ids_8_Apr_2022.csv, then pass this whitelist into the script below

```sh
python3 pixelscore_service/within_collection_score/convert_all_collections_to_numpy.py --base_dir= --code_path= --whitelist="pixelscore_service/whitelists_blacklists/np_ready_ids_8_Apr_2022.csv" --img_to_numpy_logs_dir=/mnt/disks/additional-disk/raw_logs/tmp_preprocess/img_to_numpy
```

This will do image conversion for all collections in the provided whitelist. Conversion to numpy may fail for a small number of collections, so it's important to get the whitelist of collection ids that were successfully converted to numpy with the convert_all_collections_to_numpy.py script. Like before, the whitelisted collection ids will be dumped as empty filenames in --img_to_numpy_logs_dir.

### Global histograms

Global hist is the most crutial step for pixelscore computation, as this hist contains key rarity info for each pixel in the collections that we scraped.

One should take the whitelist of collections for which numpy files have been successfully generated in the previous step i.e. pixelscore_service/whitelists_blacklists/global_hist_ready_10_Apr_2022.csv and use them to generate the global pixel histogram.

## Command shortcuts for convenience