import copy
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloaders, num_epochs=25, lr=0.00025, weight_decay=0.0001, instNorm=False):
    """
      trains model and plot validation accuracies
      Parameters:
      -----------------
      model: model to train
      dataloaders: dictionary of 'train'/'val' dataloaders
      num_epochs: number of epochs
      lr: learning rate for the optimizer Adagrad
      lr_decay: learning rate decay for the optimizer Adagrad
      weight_decay: weight decay for the optimizer Adagrad
      instNorm: if True applies torch.nn.InstanceNorm2d to input instances (recommended for 'mnist/svhn')

      Returns:
      -----------------
      trained model and validation accuracies history
    """
    since = time.time()
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    if instNorm and phase == 'train':
                        m = nn.InstanceNorm2d(model.fc2.out_features)
                        inputs = m(inputs)
                        _, _, outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    else:

                        _, _, outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # plot validation accuracy
    val_acc_history = [tensor.tolist() for tensor in val_acc_history]
    plt.plot(val_acc_history)
    plt.xlabel('epoch')
    plt.ylabel('validation accuracy')
    plt.show()
    return model, val_acc_history

def predict_batchwise(model, dataloader):
    model.eval()   # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0
    criterion = nn.CrossEntropyLoss()

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        _, _, outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    print('Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return epoch_acc, epoch_loss
