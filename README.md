## Video Classification using R2Plus1D PyTorch model on UCF-101 dataset


### Dataset:
- Download the dataset from [here](https://www.crcv.ucf.edu/research/data-sets/ucf101/)
- Prepare the folders as shown below:

```
├── UCF-101 
    ├── annotations
         ├── classInd.txt
         ├── testlist01.txt
         ├── trainlist01.txt
             ...
          
    ├── videos
         ├── ApplyEyeMakeup
         ├── ApplyLipstick
         ├── Archery
             ...
```

### Train
```commandline
bash train.sh
```
`train.sh`:
```
torchrun --nproc_per_node=$NUM_GPU train.py --amp --cache-dataset
```

### Test
```commandline
bash test.sh
```
`test.sh`:
```
python test.py --cache-dataset
```
