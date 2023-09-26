import os

from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms



class SignsDataset(Dataset):

    """
    Custom dataset class for handling traffic signs data.

    Args:
        df (pd.DataFrame): A Pandas DataFrame containing data about the pictures such as
            the path of the image or the class
        source_dir (str): The directory where the images are stored.
        transform (transforms.Compose): The transformations which will be applied
            to the images.
    
    """
    
    def __init__(self, df:pd.DataFrame, 
                 source_dir:str,transform:transforms.Compose=None) -> None:
        
        self.df = df
        self.source_dir = source_dir
        self.transform = transform

    def load_image(self,index:int) -> Image.Image:
        """
        Loads an image based on the path from the df.
        """
        image_path = os.path.join(self.source_dir,self.df['Path'].iloc[index])
        return Image.open(image_path)
    
    
    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self,index:int):

        image = self.load_image(index)
        y_class = torch.tensor(self.df['ClassId'].iloc[index])
        
        if self.transform:
            image = self.transform(image)

        return (image, y_class)
    
    def get_classes(self):
        """
        Returns the unique class IDs present in the dataset.

        Returns:
            numpy.ndarray: An array containing the unique class IDs.
        """

        y_classes = self.df['ClassId'].unique()

        return y_classes