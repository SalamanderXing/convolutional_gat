### Required packages:
- h5py
- pytorch 
- numpy
- ipdb
- matplotlib

### Run by:
From outside this folder:
```
python -m preprocessing.kmni_dataset preprocess \
         -i <location of raw data> \
         -o <location where to write data to> \
         (optional) -r <minimum relative ammount of rain pixels in each frame>
```
### Show help:
```
python -m preprocess.kmni_dataset -h
```
