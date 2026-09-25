'''
Train a model
'''

import os
import torch
from torch import nn
from torchvision import transforms
from timeit import default_timer as timer


import data_setup, engine,model_builder,utils

try:
    from torchmetrics import Accuracy
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "torchmetrics"])
    from torchmetrics import Accuracy


# Setup Hyperparameters
NUM_EPOCHS=5
BATCH_SIZE=32
HIDDEN_UNITS=10
LEARNING_RATE=0.001

#Setup Directory
train_dir='Data/pizza_steak_sushi/train'
test_dir='Data/pizza_steak_sushi/test'

#Device agnostic code
device='cuda' if torch.cuda.is_available() else 'cpu'

#Create Transform

data_transform=transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])

#Create dataloaders and get Class names
train_dataloder,test_dataloader,class_names=data_setup.create_dataloader(train_dir,test_dir,
                             data_transform,BATCH_SIZE)


#Create Model

model_1=model_builder.TinyVGG(3,10,len(class_names)).to(device)

#Loss and Optimizer and Accuracy
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(params=model_1.parameters(),lr=LEARNING_RATE)
accuracy_fn = Accuracy(task="multiclass", num_classes=len(class_names)).to(device)


start_time=timer()

#Start Training
results=engine.train_loop(model_1,train_dataloder,
                  test_dataloader,loss_fn,
                  optimizer,accuracy_fn,
                  6,device)

end_time=timer()

#Calculate Total Time
total_time=end_time-start_time
print(f'Total Time Taken: {total_time:.3f}')

#Save The Model to File

utils.save_model(model_1,'Models','Model_1.pth')
