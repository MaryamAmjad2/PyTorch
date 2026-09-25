'''
Contains Functionality for Crearing DataLoader for Image Classification Data'''
import os
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

NUM_WORKERS=os.cpu_count()
def create_dataloader(
    train_dir:str,
    test_dir:str,
    transform:transforms.Compose,
    batch_size:int,
    num_workers:int=NUM_WORKERS


):
  '''
  Create Train and test DataLoaders
  '''

  # 1. Create Datasets
  train_data=datasets.ImageFolder(root=train_dir,
                                  transform=transform)

  test_data=datasets.ImageFolder(root=test_dir,
                                transform=transform)


  #2. Get Class Names
  class_names=train_data.classes

  #3. DataLoaders
  train_dataloader=DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=True
                              )

  test_dataloader=DataLoader(dataset=test_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=NUM_WORKERS,
                            pin_memory=True
                            )
  return train_dataloader,test_dataloader,class_names


