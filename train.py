import torch
import torch.optim as optim
from utils import write_log, write_figure, LabelConverter
import numpy as np
from dataset import get_loader
from tqdm import tqdm
from model import CRNN
import string
import torch.nn as nn


def calculate_loss(inputs, texts, label_converter, device):
    criterion = nn.CTCLoss(blank=0)

    inputs = inputs.log_softmax(2)
    input_size, batch_size, _ = inputs.size()
    input_size = torch.full(size=(batch_size,), fill_value=input_size, dtype=torch.int32)

    encoded_texts, text_lens = label_converter.encode(texts)
    loss = criterion(inputs, encoded_texts.to(device), input_size.to(device), text_lens.to(device))
    return loss


def fit(epoch, model, optimizer, label_converter, device, data_loader, phase='training'):
    if phase == 'training':
        model.train()
    else:
        model.eval()

    running_loss = 0

    for images, labels in tqdm(data_loader):
        images = images.to(device)

        if phase == 'training':
            optimizer.zero_grad()
            outputs = model(images)
        else:
            with torch.no_grad():
                outputs = model(images)

        loss = calculate_loss(outputs, labels, label_converter, device)
        running_loss += loss.item()

        if phase == 'training':
            loss.backward()
            optimizer.step()

    epoch_loss = running_loss / len(data_loader)
    print('[%d][%s] loss: %.4f' % (epoch, phase, epoch_loss))
    return epoch_loss


def train():
    print('start training ...........')
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.1

    label_converter = LabelConverter(char_set=string.ascii_lowercase + string.digits)
    vocab_size = label_converter.get_vocab_size()

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = CRNN(vocab_size=vocab_size).to(device)
    # model.load_state_dict(torch.load('output/weight.pth', map_location=device))

    train_loader, val_loader = get_loader('data/CAPTCHA Images/', batch_size=batch_size)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2)

    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        train_epoch_loss = fit(epoch, model, optimizer, label_converter, device, train_loader, phase='training')
        val_epoch_loss = fit(epoch, model, optimizer, label_converter, device, val_loader, phase='validation')
        print('-----------------------------------------')

        if epoch == 0 or val_epoch_loss <= np.min(val_losses):
            torch.save(model.state_dict(), 'output/weight.pth')

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)

        write_figure('output', train_losses, val_losses)
        write_log('output', epoch, train_epoch_loss, val_epoch_loss)

        scheduler.step(val_epoch_loss)
        # scheduler.step(epoch)


if __name__ == "__main__":
    train()
