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

## Make proper dirt structure

base_dir, typically /mnt/disks/ssd/data or /mnt/disks/ssd/data_backup  

base_dir contains a folder per collection_id e.g.
/mnt/disks/ssd/data/0x9a534628b4062e123ce7ee2222ec20b86e16ca8f  

each colelction folder contrains the following:  
metadata -  a .csv file with ground truth rarityScores (not pixel scores)  
numpy - images and labels in np format (X_train, y_train). 
resized - raw nft images 224x224  
tf_logs - model checkpoint trained on the given collection, write access must be given to tf_logs  
pixelscore - a .csv file with newly computed pixelscores  
hist.png - histgram with pixel scores for this collection  

## How to run .py scripts

Call for 1 colelction at a time, takes around 10% of RAM, completes in 2 hours. For example, take colelction with chain id 0x004f5683e183908d0f6b688239e3e2d5bbb066ca

## Convert nft images to numpy arrays

```sh
python3 pixelscore_service/within_collection_score/img_to_numpy.py --collection_id=0x004f5683e183908d0f6b688239e3e2d5bbb066ca
```

## Train model on the given collection

```sh
python3 pixelscore_service/within_collection_score/train_model.py --collection_id=0x004f5683e183908d0f6b688239e3e2d5bbb066ca
```

## Run main.py from root dir to compute rarity scores

```
python3 pixelscore_service/within_collection_score/main.py --collection_id=0x004f5683e183908d0f6b688239e3e2d5bbb066ca
```

## Check rarity scores.

Must be written to /mnt/disks/ssd/data/<COLLELCTION_ID>/pixelscore/pixelscore.csv  
Histogram of pixelscores available at /mnt/disks/ssd/data/<COLLELCTION_ID>/pixelscore/hist.png.

## Script that runs all of the tools above for all collections in the whitelist.

```
python3 pixelscore_service/within_collection_score/score_all_collections.py --collections_whtelist=whitelist.csv
```

## Run scripts using pm2 from venv.
```
pm2 flush
pm2 start pixelscore_service/within_collection_score/score_all_collections.py --name score_all_collections --interpreter=python3
```
