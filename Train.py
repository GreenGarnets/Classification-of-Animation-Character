from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import os
import argparse
import numpy as np

import shuffleNetV2

import matplotlib.pyplot as plt
import utills
import sys 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
data_transforms = {
    'train': transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.5, 0.5, 0.5])
    ]),
}

data_dir = 'data_crop'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=12,
                                              shuffle=True, num_workers=0)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
classes = image_datasets['train'].classes

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model_conv.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloaders['train']):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model_conv(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        utills.progress_bar(batch_idx, len(dataloaders['train']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    model_conv.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloaders['val']):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_conv(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Epoch : %d | val_Loss: %.3f | val_ Acc: %.3f%% (%d/%d)' % (epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model_conv.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('model'):
            os.mkdir('model')
        torch.save(state, './model/model.t7')
        best_acc = acc

# 이미지 출력 함수
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# 일부 예제에 대하여 예측한 값을 보여준다.
def visualize_model(model, num_images=90):
	checkpoint = torch.load('./model/model.t7')
	model.load_state_dict(checkpoint['net'])
	best_acc = checkpoint['acc']
	start_epoch = checkpoint['epoch']

	was_training = model.training
	model.eval()
	images_so_far = 0
	fig = plt.figure()

	with torch.no_grad():
		for i, (inputs, labels) in enumerate(dataloaders['val']):
			inputs = inputs.to(device)
			labels = labels.to(device)

			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)

			for j in range(inputs.size()[0]):
				images_so_far += 1
				ax = plt.subplot(num_images//15, 15, images_so_far)
				ax.axis('off')
				ax.set_title('tag: {}'.format(classes[preds[j]]))
				imshow(inputs.cpu().data[j])

				if images_so_far == num_images:
					model.train(mode=was_training)
					return

if __name__ == "__main__":
	
	# Model
	print('==> Building model..')
	model_conv = shuffleNetV2.ShuffleNetV2(0.5)
	model_conv = model_conv.to(device)
	if device == 'cuda':
		model_conv = torch.nn.DataParallel(model_conv)
		cudnn.benchmark = True

	if True:
		# Load checkpoint.
		print('==> Resuming from checkpoint..')
		assert os.path.isdir('model'), 'Error: no checkpoint directory found!'
		checkpoint = torch.load('./model/model.t7')
		model_conv.load_state_dict(checkpoint['net'])
		best_acc = checkpoint['acc']
		start_epoch = checkpoint['epoch']

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
	
	# for param in model_conv.parameters():
	#	param.requires_grad = False

	for epoch in range(start_epoch, start_epoch+250):
		train(epoch)
		test(epoch)

	# for param in model_conv.parameters():
	#	param.requires_grad = True

	visualize_model(model_conv)
	print("best Acc : {:2f}".format(best_acc))

	plt.ioff()
	plt.show()