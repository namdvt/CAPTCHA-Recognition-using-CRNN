# CRNN for captcha recognition
## General
This is a simple PyTorch implementation of OCR system using CNN + RNN + CTC loss for captcha recognition.
## Dataset
I used CAPTCHA Images dataset which was downloaded from https://www.kaggle.com/fournierp/captcha-version-2-images
## Files

```
.
├── data
│   └── CAPTCHA Images
│       ├── test
│       ├── train
│       └── val
├── dataset.py
├── model.py
├── output
│   ├── log.txt
│   ├── loss.png
│   └── weight.pth
├── predict.py
├── README.md
├── split_train_val_test.py
├── train.py
└── utils.py

```
### Training
```
python train.py
```
Training and validation loss:

![Image description](output/loss.png)
### Testing
```
python predict.py
```
accuracy = 0.897