***Install Deps***
sudo apt-get install build-essential gfortran libatlas-base-dev python-pip python-dev
sudo apt-get install pkg-config
sudo pip install --upgrade pip
sudo pip3 install --upgrade pip3
sudo apt-get install gfortran
sudo apt-get install libatlas-base-dev
pip3 install -U pip
pip3 install -U tensorflow==2.8.0
pip3 install -r requirements.txt
***Make proper dirt structure***
base_dir, typically /mnt/disks/ssd/data or /mnt/disks/ssd/data_backup
base_dir contains a folder per collection_id e.g.
/mnt/disks/ssd/data/0x9a534628b4062e123ce7ee2222ec20b86e16ca8f 
each colelction folder contrains the following:
metadata -  a .csv file with ground truth rarityScores (not pixel scores)
numpy - images and labels in np format (X_train, y_train)
resized - raw nft images 224x224
tf_logs - model checkpoint trained on the given collection, write access must be given to tf_logs
pixelscore - a .csv file with newly computed pixelscores
***How to run py scripts.***
Call for 1 colelction at a time, takes around 10% of RAM, completes in 2 hours.
***Convert nft images to numpy arrays***
python3 pixelscore_service/within_collection_score/img_to_numpy.py
***Train model on the given collection***
python3 pixelscore_service/within_collection_score/train_model.py
***Run main.py from root dir to compute rarity scores***
python3 pixelscore_service/within_collection_score/main.py
***Check rarity scores.***
Must be written to 
/mnt/disks/ssd/data/<COLLELCTION_ID>/pixelscore/pixelscore.csv
