# ICW-fMRI-GAN
An improved conditional wasserstein generative adversarial network (ICW-GAN) that is trained to generate synthetic fMRI data samples.  The model is trained on NeuroVault collection 1952.

## Running the Code
```
# If on Ubuntu 16.04 and CUDA installation desired:
bash setup.sh

# Otherwise
pip3 install -r requirements.txt

# Run!
python3 train.py

python3 generate.py <path to generator> <num samples to generate> <output directory>

python3 evaluation/train_classifiers.py <path to generator> <path to real data> <path to synthetic data> <output directory>

python3
```

## Examples
### Training
![Training](examples/training.gif)

### Samples
![Sample 1000](examples/sample.png)
