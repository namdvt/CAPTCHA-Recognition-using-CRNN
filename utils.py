import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch


def get_dict():
    NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']
    ALL_CHAR_SET = NUMBER + ALPHABET

    char = sorted(set(''.join(ALL_CHAR_SET)))
    int2char = dict(enumerate(char))
    char2int = {char: ind for ind, char in int2char.items()}
    return int2char, char2int


def write_figure(location, train_losses, val_losses):
    plt.plot(train_losses, label='training loss')
    plt.plot(val_losses, label='validation loss')
    plt.legend()
    plt.savefig(location + '/loss.png')
    plt.close('all')


def write_log(location, epoch, train_loss, val_loss):
    if epoch == 0:
        f = open(location + '/log.txt', 'w+')
        f.write('epoch\t\ttrain_loss\t\tval_loss\n')
    else:
        f = open(location + '/log.txt', 'a+')

    f.write(str(epoch) + '\t' + str(train_loss) + '\t' + str(val_loss) + '\n')

    f.close()
