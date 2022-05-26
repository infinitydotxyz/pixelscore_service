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

### Get pixel histogrms for each collection

Global hist is the most crutial step for pixelscore computation, as this hist contains key rarity info for each pixel in the collections that we scraped.

One should take the whitelist of collections for which numpy files have been successfully generated in the previous step i.e. pixelscore_service/whitelists_blacklists/global_hist_ready_10_Apr_2022.csv and use them to generate the global pixel histogram.

```sh
python3 pixelscore_service/within_collection_score/raw_score_all_collections.py --save_pixels_hist=True --global_score=False --hist_dir=/mnt/disks/additional-disk/histograms
```

The script above (with given flags) generates pixel histograms and saves them to hist_dir/<collection_id>/pixels_hist.npz

### Global global pixels histogram

Collection histograms obtained from the previous step need to be aggregates into the global histogram, this is a single threaded process

```sh
pixelscore_service/within_collection_score/global_score/get_global_counts.py --hist_dir=/mnt/disks/additional-disk/histograms --global_hist_dir=/mnt/disks/additional-disk/global_hist --global_hist_shards_dir=/mnt/disks/additional-disk/global_histograms/shards
```

The final global histogram will be saved to <global_hist_dir>/global_hist.npz.

Congrats! The hardest work is done, now we can use all the info from the global hist to score the collections.

### Score collections using global histogram

```sh
python3 pixelscore_service/within_collection_score/raw_score_all_collections.py --save_pixels_hist=False --global_score=True
```
The global scores will be saved to <base_dir>/<collection_id>/pixelscore/global_raw_pixelscore.csv
The histogram of pixelscores is available in <base_dir>/<collection_id>/pixelscore/global_raw_hist.png

## Command shortcuts for convenience

### Source venv
```sh
source /mnt/disks/ssd/pixelscore_service/pixelscore_service/bin/activate
```
### Convert to numpy
```sh
python3 /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/within_collection_score/convert_all_collections_to_numpy.py --base_dir=/mnt/disks/additional-disk/data --code_path=/mnt/disks/ssd/git/pixelscore_service --use_whitelist=False --img_to_numpy_logs_dir=/mnt/disks/additional-disk/raw_logs/tmp_preprocess/img_to_numpy_2
```
with pm2
```sh
pm2 start /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/within_collection_score/convert_all_collections_to_numpy.py --name convert_all_collections_to_numpy.py --interpreter=python3 --no-autorestart -- --base_dir=/mnt/disks/additional-disk/data --code_path=/mnt/disks/ssd/git/pixelscore_service --use_whitelist=False --img_to_numpy_logs_dir=/mnt/disks/additional-disk/raw_logs/tmp_preprocess/img_to_numpy_2
```
### Check is_empty
```sh
python3 /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/preprocessing/preprocess_main.py --base_dir=/mnt/disks/additional-disk/data --mode="is_empty" --use_whitelist=False --is_empty_logs_dir=/mnt/disks/additional-disk/raw_logs/tmp_preprocess/is_empty_2
```
with pm2
```sh
pm2 start /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/preprocessing/preprocess_main.py --name check_is_empty --interpreter=python3 --no-autorestart -- --base_dir=/mnt/disks/additional-disk/data --mode="is_empty" --use_whitelist=False --is_empty_logs_dir=/mnt/disks/additional-disk/raw_logs/tmp_preprocess/is_empty_2
```
### Check is_pre_reveal

```sh
python3 /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/preprocessing/preprocess_main.py --base_dir=/mnt/disks/additional-disk/data --use_whitelist=False --mode="is_pre_reveal" --pre_reveal_logs_dir=/mnt/disks/additional-disk/raw_logs/tmp_preprocess/is_pre_reveal_2
```
with pm2
```sh
pm2 start /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/preprocessing/preprocess_main.py --name check_is_pre_reveal --interpreter=python3 --no-autorestart -- --base_dir=/mnt/disks/additional-disk/data --use_whitelist=False --mode="is_pre_reveal" --pre_reveal_logs_dir=/mnt/disks/additional-disk/raw_logs/tmp_preprocess/is_pre_reveal_2
```
### Get pixel histograms (for each collection)
Note, the histograms are saved for each collection separately, so they don't need to be re-computed for old collections.
```sh
python3 /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/within_collection_score/raw_score_all_collections.py --base_dir=/mnt/disks/additional-disk/data --code_path=/mnt/disks/ssd/git/pixelscore_service --collection_whitelist=/mnt/disks/ssd/git/pixelscore_service/pixelscore_service/whitelists_blacklists/global_hist_ready_ready_5_May_2022.csv --save_pixels_hist=True --global_score=False --hist_dir=/mnt/disks/additional-disk/histograms
```
with pm2
```sh
pm2 start /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/within_collection_score/raw_score_all_collections.py --name compute_collection_hist --interpreter=python3 --no-autorestart -- --base_dir=/mnt/disks/additional-disk/data --code_path=/mnt/disks/ssd/git/pixelscore_service --collection_whitelist=/mnt/disks/ssd/git/pixelscore_service/pixelscore_service/whitelists_blacklists/global_hist_ready_ready_5_May_2022.csv --save_pixels_hist=True --global_score=False --hist_dir=/mnt/disks/additional-disk/histograms
```
### Get global histogram

Don't forget to merge any old collection ids and new ids into one whitelist.
```sh
python3 /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/within_collection_score/global_score/get_global_counts.py --code_path=/mnt/disks/ssd/git/pixelscore_service --collection_whitelist=/mnt/disks/ssd/git/pixelscore_service/pixelscore_service/whitelists_blacklists/global_hist_ready_merged_8_May_2022.csv --hist_dir=/mnt/disks/additional-disk/histograms --global_hist_dir=/mnt/disks/additional-disk/global_hist_2 --global_hist_shards_dir=/mnt/disks/additional-disk/global_histograms/shards_2
```

with pm2
```sh
pm2 start /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/within_collection_score/global_score/get_global_counts.py --name get_global_hist --interpreter=python3 --no-autorestart -- --code_path=/mnt/disks/ssd/git/pixelscore_service --collection_whitelist=/mnt/disks/ssd/git/pixelscore_service/pixelscore_service/whitelists_blacklists/global_hist_ready_merged_8_May_2022.csv --hist_dir=/mnt/disks/additional-disk/histograms --global_hist_dir=/mnt/disks/additional-disk/global_hist_2 --global_hist_shards_dir=/mnt/disks/additional-disk/global_histograms/shards_2
```
### Compute global scores using global histogram

```sh
python3 /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/within_collection_score/raw_score_all_collections.py --base_dir=/mnt/disks/additional-disk/data --code_path=/mnt/disks/ssd/git/pixelscore_service --collection_whitelist=/mnt/disks/ssd/git/pixelscore_service/pixelscore_service/whitelists_blacklists/global_hist_ready_merged_8_May_2022.csv --save_pixels_hist=False --global_score=True --hist_dir=/mnt/disks/additional-disk/histograms --raw_logs_dir=/mnt/disks/additional-disk/raw_logs/tmp_2
```
with pm2
```sh
pm2 start /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/within_collection_score/raw_score_all_collections.py --name raw_score_all_collections --interpreter=python3 --no-autorestart -- --base_dir=/mnt/disks/additional-disk/data --code_path=/mnt/disks/ssd/git/pixelscore_service --collection_whitelist=/mnt/disks/ssd/git/pixelscore_service/pixelscore_service/whitelists_blacklists/global_hist_ready_merged_8_May_2022.csv --save_pixels_hist=False --global_score=True --hist_dir=/mnt/disks/additional-disk/histograms --raw_logs_dir=/mnt/disks/additional-disk/raw_logs/tmp_2
```

### Merge scores into one csv
```sh
python3 /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/post_processing/post_process_main.py --mode="merge_results" --code_path=/mnt/disks/ssd/git/pixelscore_service --base_dir=/mnt/disks/additional-disk/data --scored_collections_whitelist=/mnt/disks/ssd/git/pixelscore_service/pixelscore_service/whitelists_blacklists/scored_10_May_2022.csv --merged_scores_file=/mnt/disks/additional-disk/merged_scores/merged_scores_10_May_2022.csv
```

### Analyze merged scores
```sh
python3 /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/post_processing/post_process_lib.py --mode="analyze_merged_scores"
```

### Some temp logs for scoring results can be found here:
/mnt/disks/ssd/pixelscore_service/within_collection_score/scoring_results/1652105068_results.csv

### Other (e.g. use_log_scores).
### Log scores.
pm2 start /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/within_collection_score/raw_score_all_collections.py --name log_raw_score_all_collections --interpreter=python3 --no-autorestart -- --base_dir=/mnt/disks/additional-disk/data --code_path=/mnt/disks/ssd/git/pixelscore_service --collection_whitelist=/mnt/disks/ssd/git/pixelscore_service/pixelscore_service/whitelists_blacklists/global_hist_ready_merged_8_May_2022.csv --save_pixels_hist=False --global_score=True --hist_dir=/mnt/disks/additional-disk/histograms --raw_logs_dir=/mnt/disks/additional-disk/raw_logs/tmp_3 --use_log_scores=True

### Merge log scores.
python3 /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/post_processing/post_process_main.py --mode="merge_results" --code_path=/mnt/disks/ssd/git/pixelscore_service --base_dir=/mnt/disks/additional-disk/data --scored_collections_whitelist=/mnt/disks/ssd/git/pixelscore_service/pixelscore_service/whitelists_blacklists/scored_10_May_2022.csv --merged_scores_file=/mnt/disks/additional-disk/merged_scores/log_merged_scores_10_May_2022.csv --use_log_scores=True

### Analyze merged log scores.
```sh
python3 /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/post_processing/post_process_main.py --mode="analyze_merged_scores" --code_path=/mnt/disks/ssd/git/pixelscore_service --base_dir=/mnt/disks/additional-disk/data --scored_collections_whitelist=/mnt/disks/ssd/git/pixelscore_service/pixelscore_service/whitelists_blacklists/scored_10_May_2022.csv --merged_scores_file=/mnt/disks/additional-disk/merged_scores/log_merged_scores_10_May_2022.csv --use_log_scores=True
```

### Bucketize (binarize, get binned pixelscore) merged log scores.
```sh
python3 /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/post_processing/post_process_main.py --mode="bucketize_scores" --code_path=/mnt/disks/ssd/git/pixelscore_service --base_dir=/mnt/disks/additional-disk/data --scored_collections_whitelist=/mnt/disks/ssd/git/pixelscore_service/pixelscore_service/whitelists_blacklists/scored_10_May_2022.csv --merged_scores_file=/mnt/disks/additional-disk/merged_scores/log_merged_scores_10_May_2022.csv --use_log_scores=True
```

### Plot Rarest Colors.
```sh
python3 /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/post_processing/post_process_main.py --mode="plot_rarest_colors" --code_path=/mnt/disks/ssd/git/pixelscore_service --base_dir=/mnt/disks/additional-disk/data --scored_collections_whitelist=/mnt/disks/ssd/git/pixelscore_service/pixelscore_service/whitelists_blacklists/scored_10_May_2022.csv --merged_scores_file=/mnt/disks/additional-disk/merged_scores/log_merged_scores_10_May_2022.csv --use_log_scores=True
```

### Plot Log.
```sh
python3 /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/post_processing/post_process_main.py --mode="plot_log" --code_path=/mnt/disks/ssd/git/pixelscore_service --base_dir=/mnt/disks/additional-disk/data --scored_collections_whitelist=/mnt/disks/ssd/git/pixelscore_service/pixelscore_service/whitelists_blacklists/scored_10_May_2022.csv --merged_scores_file=/mnt/disks/additional-disk/merged_scores/log_merged_scores_10_May_2022.csv --use_log_scores=True
```

### Plot bucketized hist.
```sh
python3 /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/post_processing/post_process_main.py --mode="plot_bucketized_hist" --code_path=/mnt/disks/ssd/git/pixelscore_service --base_dir=/mnt/disks/additional-disk/data --scored_collections_whitelist=/mnt/disks/ssd/git/pixelscore_service/pixelscore_service/whitelists_blacklists/scored_10_May_2022.csv --merged_scores_file=/mnt/disks/additional-disk/merged_scores/log_merged_scores_10_May_2022.csv --use_log_scores=True
```

### Plot Rarest Collections.
```sh
python3 /mnt/disks/ssd/git/pixelscore_service/pixelscore_service/post_processing/post_process_main.py --mode="plot_rarest_collections" --code_path=/mnt/disks/ssd/git/pixelscore_service --base_dir=/mnt/disks/additional-disk/data --scored_collections_whitelist=/mnt/disks/ssd/git/pixelscore_service/pixelscore_service/whitelists_blacklists/scored_10_May_2022.csv --merged_scores_file=/mnt/disks/additional-disk/merged_scores/log_merged_scores_10_May_2022.csv --use_log_scores=True
```