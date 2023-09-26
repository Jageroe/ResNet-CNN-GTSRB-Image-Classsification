import os

import torch
import torch.nn as nn

from tqdm.auto import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


def _train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer) -> tuple:
    
    """
    Trains a PyTorch model for a single epoch.

    Args:
        model: The PyTorch model to be trained.
        dataloader: DataLoader instance for training data.
        loss_fn: Loss function to minimize.
        optimizer: Optimizer to update model parameters.

    Returns:
        tuple: (train_loss, train_acc)
    """

    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def _test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module) -> tuple:


    """Tests a PyTorch model for a single epoch.

    Args:
        model: The PyTorch model to be trained.
        dataloader: DataLoader instance for training data.
        loss_fn: Loss function to minimize.
        device: Device to perform computation. Default is "cuda".

    Returns:
        tuple: (test_loss, test_acc)
    """

    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # Forward pass
            test_pred_logits = model(X)

            # Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc



def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          saving:bool=False) -> dict:

    """
    Trains and tests a PyTorch model.

    Args:
        model: The PyTorch model to be trained and tested.
        train_dataloader: DataLoader for training data.
        test_dataloader: DataLoader for testing data.
        optimizer: Optimizer for model optimization.
        loss_fn: Loss function for calculating loss.
        epochs: Number of training epochs.
        Saving: the best parameters if saving == True.
        
    Returns:
        dict: A dictionary containing loss and acc for train and test sets for each epoch
    """

    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    best_acc = 0.0
    
    # Loop through training and testing steps for the given number of epochs
    for epoch in tqdm(range(epochs)):

        # run one train step
        train_loss, train_acc = _train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer)
        
        # run one test step
        test_loss, test_acc = _test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn
        )
        
        
        # Print out the results of the current epoch
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # storing the best performing parameters and its accuracy score
        if (epoch > 0 and best_acc < results['test_acc'][-1]):
            best_model_weights = model.state_dict()
            best_acc = test_acc

    # Saving the best parameters if saving == True
    if saving:
        if not os.path.exists('saved_weights'):
            os.makedirs('saved_weights')

        # load best model weights
        saving_path = os.path.join('saved_weights',f'model-{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')

        torch.save(best_model_weights,saving_path)

        print(f'{model=}'.split('=')[0])
        print(f'The best epoch has {best_acc:.4f} accuracy. Its parameters are saved in the following path: {saving_path}')
   
    return results



def plot_loss_curves(results: dict):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();