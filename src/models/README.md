
# Training Wave-U-Net Model

## On demand-16k dataset
The dataset is available on huggingface, I modified the code to read the dataset directly form huggingface and train the model.

### Start the training
```
python train.py --instruments "clean" --cuda --dataset "demand" --dataset_dir "/storage/hdd0/data/mayank_dataset" --sr 16000 --channels 1 --separate 0 --patience 10
```

### For predicting
```
python predict.py --load_model checkpoints/waveunet/checkpoint --instruments "clean" --cuda --sr 16000 --channels 1 --separate 0 --input "./audio_outputs/input_151557.wav"
```