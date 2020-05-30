import torch
import argparse
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms

from torch import nn
from torchvision import datasets
from tqdm.notebook import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from network import Net
from send_mail import SendMail


def check_gpu():
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
    return train_on_gpu

train_on_gpu = check_gpu()

def load_data(batch_size):
    # Loading the dataset
    print("Loading the dataset ...\n")

    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = batch_size
    # percentage of training set to use as validation
    valid_size = 0.2

    # convert data to a normalized torch.FloatTensor
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # choose the training and test datasets
    train_data = datasets.CIFAR10('data', train=True,
                                  download=True, transform=transform)
    test_data = datasets.CIFAR10('data', train=False,
                                 download=True, transform=transform)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
        num_workers=num_workers)

    # specify the image classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    return train_loader, valid_loader, test_loader, classes


def define_model():
    # Define the Network Architecture
    print("Defining the Network Architecture...\n")
    # create a complete CNN
    model = Net()

    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        model.cuda()
    return model

def loss_and_optim(model, lr, opt):

    # Specify Loss Functiona and Optimizer
    print("Defining the loss function and optimizer...\n")

    # specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()
    # specify optimizer
    if opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    return criterion, optimizer

def train(n_epochs, model, train_loader, valid_loader, criterion, optimizer):
    # Train the Network
    # print(f"Training the Network with {n_epochs} epochs...\n")

    # number of epochs to train the model
    n_epochs = n_epochs

    valid_loss_min = np.Inf # track change in validation loss

    # For early stopping
    epochs_to_stop = 5
    epochs_no_change = 0

    for epoch in range(1, n_epochs+1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)
        
        # calculate average losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), 'model_cifar.pt')
            valid_loss_min = valid_loss
            epochs_no_change = 0
        else:
            epochs_no_change += 1

        if epochs_no_change == epochs_to_stop:
            print("Early stopping...")
            break

    model.load_state_dict(torch.load('model_cifar.pt'))

    return model


def test(batch_size, model, test_loader, classes, criterion, optimizer):


    # Test the Trained Network
    print("Testing the Trained Network...\n")

    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    model.eval()
    # iterate over test data
    for data, target in test_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)    
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # average test loss
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

    return (100. * np.sum(class_correct) / np.sum(class_total))

def start():
    # Command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--tune', type=bool, default=False, help="pass -t True for hyperparameter tuning")
    args = vars(ap.parse_args())

    batch_size=20
    n_epochs = 1
    # loads data and returns train, valid and test data
    train_load, valid_load, test_load, classes = load_data(batch_size)
    if not args['tune']:
        learning_rate = 0.01
        optimizer = 'SGD'

        model = define_model()
        cr, op = loss_and_optim(model, learning_rate, optimizer)

        print("Training the network without hyperparameter tuning")
    
        train(n_epochs, model, train_load, valid_load, cr, op)
        accuracy = test(batch_size, model, test_load, classes, cr, op)
        accuracy = int(accuracy)
        with open('accuracy.txt', 'w') as f:
            f.write(str(accuracy))
    else:
	n_epochs = 2
        learning_rates = [0.01, 0.001]
        optimizers = ['Adam', 'SGD']
        acc = []
        max_acc = 0
        best_lr = 0
        best_opt = ''

        for learning_rate in learning_rates:
            for optimizer in optimizers:
                print("\n ** Training with {} optimizer and {} learning rate **\n".format(optimizer, learning_rate))
                model = define_model()
                cr, op = loss_and_optim(model, learning_rate, optimizer)
                model = train(n_epochs, model, train_load, valid_load, cr, op)
                accuracy = test(batch_size, model, test_load, classes, cr, op)
                if accuracy > max_acc:
                    max_acc = accuracy
                    best_opt = optimizer
                    best_lr = learning_rate

                    # Save the best performing model
                    torch.save(model.state_dict(), 'model_cifar.pt')
                    print("Saving best model...")
        print("\nBest Learning Rate : {}\nBest Optimizer : {}\nBest Accuracy: {}".format(best_lr, best_opt,
                                                                                       str(max_acc)))
        # Send email user about the model performance and best hyperparameter so far
        print("Sending email to the user...")
        s = SendMail(str(max_acc) + '%', best_lr, best_opt)
        s.send_mail()
        max_acc = int(max_acc)
        with open('accuracy.txt', 'w') as f:
            f.write(str(max_acc))

# Start the training process
start()