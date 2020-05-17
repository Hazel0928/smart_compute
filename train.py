from __future__ import print_function
import argparse
import os
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np
import time
import torch
import torch.nn as nn
from torch.nn import init
import torch.utils.data
import matplotlib.pyplot as plt
from bp_network import bp_network_basic

parser = argparse.ArgumentParser(description='baseline')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size for training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--gpu_ids', type=str, default='0', help='which device the model is trained on')
parser.add_argument('--test_batch_size', type=int, default=1, help='batch_size used for test')
parser.add_argument('--initial_lr', type=float, default=0.001, help='initial learning rate for pretrain')
parser.add_argument('--saving_freq', type=int, default=200, help='denote the frequency to save the training model')
parser.add_argument('--savemodel', default='./', help='save model')
parser.add_argument('--model', type=str, default='wine', help='denote the model name')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
torch.cuda.set_device(int(args.gpu_ids))

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# load data
if args.model == 'wine':
    from csv_reader import DataFolder, att_split, label_split
elif args.model == 'car':
    from data_process import DataFolder, att_split, label_split
else:
    raise Exception('not a proper model')


train_data, test_data = att_split()
train_label, test_label = label_split()
TrainLoader = torch.utils.data.DataLoader(
    DataFolder(train_data, train_label), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    drop_last=False
)
TestLoader = torch.utils.data.DataLoader(
    DataFolder(test_data, test_label), batch_size=args.test_batch_size, shuffle=True, num_workers=args.num_workers,
    drop_last=False
)

print('create new summary file')
if not os.path.exists(args.savemodel):
    os.mkdir(args.savemodel)
logger = SummaryWriter(args.savemodel)

if args.model == 'car':
    # car quality classification
    model = bp_network_basic(input_dim=6, hidden_dim=16, num_class=4)
elif args.model == 'wine':
    model = bp_network_basic(input_dim=11, hidden_dim=32, num_class=11)
else:
    raise Exception('not a proper model')
model = model.cuda()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.initial_lr, betas=(0.9, 0.999))


def train(train_data, train_label):
    model.train()
    trainData = Variable(torch.FloatTensor(train_data))
    trainLabel = Variable(train_label)
    trainLabel = torch.squeeze(trainLabel, dim=1)
    
    if args.cuda:
        trainData = trainData.cuda()
        trainLabel = trainLabel.cuda()
    
    # optim part
    optimizer.zero_grad()
    output = model(trainData)
    loss = loss_function(output, trainLabel)
    loss.backward()
    optimizer.step()
    
    return loss.item()

def test(test_data, test_label):
    model.eval()
    testData = Variable(torch.FloatTensor(test_data))
    testLabel = Variable(test_label)
    testLabel = torch.squeeze(testLabel, dim=1)
    if args.cuda:
        testData = testData.cuda()
        testLabel = testLabel.cuda()
    
    with torch.no_grad():
        output = model(testData)
        loss = loss_function(output, testLabel).data.cpu()
        _, preds = output.max(1)
        precision = preds.eq(testLabel).sum().data.cpu()

    return loss, precision

def adjust_learning_rate(optimizer, epoch):
    lr = args.initial_lr
    # linear decay
    lr = lr - epoch / args.epochs * lr
    print('learning rate is %.6f' % (lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    epoch_train_loss = []
    epoch_test_loss = []
    epoch_acc = []
    epoch_list = []
    max_acc = 0
    min_train_loss = 10
    min_test_loss = 10
    for epoch in range(args.epochs):
        # adjust_learning_rate(optimizer, epoch)
        print('this is %d epoch' %(epoch+1))
        total_train_loss = 0.
        for batch_idx, (att, label) in enumerate(TrainLoader):
            loss = train(att, label)
            total_train_loss += loss
        avg_train_loss = total_train_loss / len(TrainLoader)
        if avg_train_loss < min_train_loss:
            min_train_loss = avg_train_loss
        epoch_list.append(epoch)
        epoch_train_loss.append(avg_train_loss)
        print('Train epoch %d avg train loss is %.3f' % (epoch+1, avg_train_loss))
        
        logger.add_scalar('train_loss', avg_train_loss, epoch)
        savename = args.savemodel + '/checkpoint_{}.tar'.format(epoch)
        torch.save({
            'epoch':epoch,
            'state_dict':model.state_dict(),
            'train_loss':avg_train_loss
        }, savename)

        test_loss = 0.
        preds = 0.
        for batch_idx, (att, label) in enumerate(TestLoader):
            loss, pred = test(att, label)
            test_loss += loss
            preds += pred
        avg_test_loss = test_loss / len(TestLoader)
        preds = preds / len(TestLoader)
        if preds > max_acc:
            max_acc = preds
        if avg_test_loss< min_test_loss:
            min_test_loss = avg_test_loss
        epoch_acc.append(preds)
        epoch_test_loss.append(avg_test_loss)
        print('Test epoch {:} avg test loss is{:}, Accuracy is {:.3f}'.format(
            epoch, avg_test_loss, preds
        ))
        logger.add_scalar('test_loss', avg_test_loss, epoch)
        logger.add_scalar('Pred accuracy', preds, epoch)
        print('--------------------------->')
    
    # visualize
    print('visualize')
    print('max_acc', max_acc)
    print('min_train_loss', min_train_loss)
    print('min_test_loss', min_test_loss)
    plt.plot(epoch_list, epoch_train_loss, linewidth=1, marker='.', markersize=8, label='TrainLoss')
    plt.plot(epoch_list, epoch_test_loss, linewidth=1, marker='.', markersize=8, label='TestLoss')
    plt.plot(epoch_list, epoch_acc, linewidth=1, marker='.', markersize=8, label='Acc')
    plt.legend()
    plt.xlabel('epoch num')
    plt.title('{:} layers {:} lr {:} epochs result'.format(7, args.initial_lr, args.epochs))
    plt.savefig('vis.png')


if __name__ == '__main__':
    main()