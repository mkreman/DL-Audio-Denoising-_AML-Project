
# Training Wave-U-Net Model

## On demand-16k dataset
The dataset is available on huggingface, I modified the code to read the dataset directly form huggingface and train the model.

```
python train.py --instruments "clean, noisy" --cuda --dataset "demand" --dataset_dir "/storage/hdd0/data/mayank_dataset" --sr 16000 --channels 1
```