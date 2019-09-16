# Autoencoder Weight Transfer Network (AE-WTN)

This is the code to replicate the AE-WTN experiments in the **Scaling Object Detection by Transferring Classification Weights** paper accepted as an oral paper at ICCV 2019.



Please consider citing this project in your publications if it helps your research.

```
@inproceedings{kuen2019scaling,
   title = {Scaling Object Detection by Transferring Classification Weights},
   author = {Kuen, Jason and Perazzi, Federico and Lin, Zhe and Zhang, Jianming and Tan, Yap-Peng},
   booktitle = {ICCV},
   year = {2019}
}
```



## Installation

- Prerequisites: [PyTorch v1.2.0](https://github.com/pytorch/pytorch/tree/v1.2.0), [pandas](https://github.com/pandas-dev/pandas)
- For others, please refer to [INSTALL.md](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md) of the official [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) repo.
- `python setup.py build develop`



## Dataset Preparation

```bash
cd AE-WTN/datasets/openimages

# download the Open Images training annotations 
wget https://storage.googleapis.com/openimages/challenge_2018/train/challenge-2018-train-annotations-bbox.csv

## create symlinks (in datasets/openimages) to image directories of training and evaluation datasets

# Open Images (challenge/V4/V5) training images directory (about 1.58M images with all download parts combined)
# https://www.figure-eight.com/dataset/open-images-annotated-with-bounding-boxes/ (train_00.zip, train_01.zip, ...)
ln -s train /path_to_openimages_images/train

# Open Images (V4/V5) validation images (41,620 images)
# https://datasets.figure-eight.com/figure_eight_datasets/open-images/zip_files_copy/validation.zip
ln -s val_600 /path_to_openimages_images/validation

# Visual Genome (Version 1.2) images (108,079 images with part 1 & 2 combined)
# https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
# https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
ln -s VG_100K /path_to_visualgenome_images

cd ../..
```



## Training & Evaluation

By default, 4 GPU cards are utilized for training and evaluation.

Training checkpoints are stored in the same directory. Evaluation results are stored in the `inference` subdirectory.


```bash
cd experiment

# training
sh train.sh

# evaluate on the 3 evaluation datasets
sh test.sh
```

**Pretrained model**: [download link](https://entuedu-my.sharepoint.com/:u:/g/personal/jkuen001_e_ntu_edu_sg/ESgnD2nsp9hPnlUJ38-PzHwBjkfxyoLGmWEdZtT2Wwbe2w?e=Jh7G8C) (place it in the same directory before running evaluation)



## License

AE-WTN is released under the MIT license. See [LICENSE](LICENSE) for additional details.
