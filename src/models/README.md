
# Training Wave-U-Net Model

## On demand-16k dataset
The dataset is available on huggingface, I modified the code to read the dataset directly form huggingface and train the model.

### Useful Commands
```
conda deactivate
conda deactivate
conda activate aml
cd src/models/Wave-U-Net-Pytorch

screen -xr aml

rm -rf logs/waveunet/*
rm -rf checkpoints/waveunet/*
rm -rf hdf/*hdf5
```

### Start the training
```
python -i train.py --instruments "clean" --cuda --dataset "demand" --dataset_dir "/storage/hdd0/data/mayank_dataset" --sr 16000 --channels 1 --separate 0 --patience 10
```