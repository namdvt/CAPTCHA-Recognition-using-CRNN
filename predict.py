import string

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from model import CRNN
import os
from tqdm import tqdm
import glob
from dataset import CaptchaImagesDataset
from utils import LabelConverter
from tqdm import tqdm


if __name__ == '__main__':
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    label_converter = LabelConverter(char_set=string.ascii_lowercase + string.digits)
    vocab_size = label_converter.get_vocab_size()

    model = CRNN(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load('output/weight.pth', map_location=device))
    model.eval()

    correct = 0.0
    image_list = glob.glob('data/CAPTCHA Images/test/*')
    for image in tqdm(image_list):
        ground_truth = image.split('/')[-1].split('.')[0]
        image = Image.open(image).convert('RGB')
        image = F.to_tensor(image).unsqueeze(0).to(device)

        output = model(image)
        encoded_text = output.squeeze().argmax(1)
        decoded_text = label_converter.decode(encoded_text)

        if ground_truth == decoded_text:
            correct += 1

    print('accuracy =', correct/len(image_list))