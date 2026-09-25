
# import torch
# from pathlib import Path

# def save_model(model:torch.nn.Module,
#                target_dir:str,
#                model_name:str):

#   # Create Target Directory
#   target_dir_path=Path(target_dir)
#   target_dir_path.mkdir(target_dir,exist_ok=True)

#   # Create Model Save Path
#   assert model_name.endswith('.pth') or model_name.endswith('.pt'),'name Should end on pth or pt'
#   model_save_path=target_dir_path/model_name

#   #Save the Model
#   print(f'[INFO] Saving Model to :{model_save_path}')
#   torch.save(obj=model.state_dict(),f=model_save_path)

import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):

  # Create Target Directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)

  # Create Model Save Path
  assert model_name.endswith('.pth') or model_name.endswith('.pt'), 'Name should end with .pth or .pt'
  model_save_path = target_dir_path / model_name

  # Save the Model
  print(f'[INFO] Saving Model to: {model_save_path}')
  torch.save(obj=model.state_dict(), f=model_save_path)
