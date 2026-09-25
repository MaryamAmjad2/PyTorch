
'''
Turn Our Train_steps into Python Script It Should have acces to train_step and test_step
function
'''



import torch
from typing import Dict,List,Tuple
from tqdm.auto import tqdm

try:
    from torchmetrics import Accuracy
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "torchmetrics"])
    from torchmetrics import Accuracy


device='cuda' if torch.cuda.is_available() else 'cpu'

#1. Train Steps

def train_steps(model:torch.nn.Module,
                dataloader:torch.utils.data.DataLoader,
                loss_fn:torch.nn.Module,
                optimizer:torch.optim.Optimizer,
                accuracy_fn):

  model.train()
  train_loss,train_acc=0,0

  #Loop through dataloader
  for batch, (X,y) in enumerate(dataloader):

    # Send data to target device
    X,y=X.to(device),y.to(device)

    # Forward Pass
    y_pred=model(X) #Output Model Logits

    # Calculate Loss
    loss=loss_fn(y_pred,y)
    train_loss+=loss.item()

    optimizer.zero_grad()

    #Back Propagation
    loss.backward()

    optimizer.step()


    # Calculate Acccuracy
    y_pred_class = torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
    train_acc += accuracy_fn(y_pred_class, y).item()

  # Adjust Metric to Get Avergae Loss
  train_loss=train_loss/len(dataloader)
  train_acc=train_acc/len(dataloader)
  return train_loss,train_acc



#2. Test steps

def  test_steps(model:torch.nn.Module,
                dataloader:torch.utils.data.DataLoader,
                loss_fn:torch.nn.Module,
                optimizer:torch.optim.Optimizer,
                accuracy_fn):

  # put in Eval Mode
  model.eval()

  test_loss,test_acc=0,0
  with torch.inference_mode():

    for batch,(X_test,y_test) in enumerate(dataloader):

      X_test,y_test=X_test.to(device),y_test.to(device)

      test_pred_logits=model(X_test)

      loss=loss_fn(test_pred_logits,y_test)
      test_loss+=loss.item()

      # Cal Accc
      test_pred_label=test_pred_logits.argmax(dim=1)
      test_acc += accuracy_fn(test_pred_label, y_test).item()
      #test_acc+=((test_pred_label==y_test).sum().item()/len(dataloader))
      #test_acc+=((test_pred_label==y_test).sum().item()/len(y_test))

      # Adjust Metric to Get Avergae Loss
  test_loss=test_loss/len(dataloader)
  test_acc=test_acc/len(dataloader)
  return test_loss, test_acc


# 3. Train Loop

def train_loop(model:torch.nn.Module,
                train_dataloader:torch.utils.data.DataLoader,
                test_dataloader:torch.utils.data.DataLoader,
                loss_fn:torch.nn.Module,
                optimizer:torch.optim.Optimizer,
               accurcay_fn,
               epochs:int = 6,
               device=device):

  # Empty Result Dictionary

  results={'train_loss':[],
           'train_acc':[],
           'test_loss':[],
           'test_acc':[]}


  for epoch in tqdm(range(epochs)):
    train_loss,train_acc=train_steps(model,train_dataloader,
                                     loss_fn,optimizer,accurcay_fn
                                    )

    test_loss,test_acc=test_steps(model,test_dataloader,
                                     loss_fn,optimizer,accurcay_fn
                                    )


    # print out

    print(f'Epoch: {epoch} | Train Loss {train_loss:.4f} | Train Accuracy {train_acc:.4f} | Test Loss {test_loss:.4f}  Test Accuracy {test_acc:.4f}')

    # Update Results Dict
    results['train_loss'].append(train_loss)
    results['train_acc'].append(train_acc)
    results['test_loss'].append(test_loss)
    results['test_acc'].append(test_acc)

  # Return the Filled Results
  return results













