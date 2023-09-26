import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"


def predict_test_loader(model: torch.nn.Module, 
                        dataloader: torch.utils.data.DataLoader) -> tuple:
    
    """  
    Returns the input PyTorch model's predictions with the actual classes

    Args:
        model: The PyTorch model to predict with.
        dataloader: DataLoader instance to predict.

    Returns:
        tuple: A tuple containing the actual classes and predicted classes.
    """
    

    pred_classes = list()
    act_classes = list()
    
    #Turn on model evaluation mode and inference mode
    model.eval() 
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # Forward pass
            test_pred_logits = model(X)
            
            # Convert prediction probabilities -> prediction labels
            test_pred_labels = test_pred_logits.argmax(dim=1)

            # populate the pred_classes and the act_classes lists
            pred_classes.append(test_pred_labels.detach().cpu().item())
            act_classes.append(y.detach().cpu().item())

    return (act_classes, pred_classes)



def pred_and_plot_image(model: torch.nn.Module, 
                        image_path: str,
                        act_class = str, 
                        transform=None,
                        device: torch.device = device) -> None:
    
    """Makes a prediction on an image and plots it with its prediction."""
    
    target_image = Image.open(image_path)
    
    # Transform if necessary
    if transform:
        target_image = transform(target_image)

    model.to(device)
    
    #Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():

        target_image = target_image.unsqueeze(dim=0)

        # Forward pass
        target_image_pred = model(target_image.to(device))
        
    #  Convert logits -> prediction probabilities
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    
    # Plot the image
    plt.figure(figsize=(3,3))
    plt.imshow(target_image.squeeze().permute(1, 2, 0)) # make sure it's the right size for matplotlib

    title = f"""
    Act: {act_class} | Pred: {target_image_pred_label.detach().cpu().item()}
    Prob: {target_image_pred_probs.max().cpu():.3f}
    """

    if act_class != str(target_image_pred_label.detach().cpu().item()):
        color = 'r'
    else:
        color = 'g'

    plt.title(title, {'color' : color})
    plt.axis(False);
    plt.show()
