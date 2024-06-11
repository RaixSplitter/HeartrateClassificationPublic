import numpy as np
import torch
import torch.nn as nn
import hydra
from bachelors_thesis.utils.evaluation import evaluate
from datetime import datetime
import os

from bachelors_thesis.models.model import dummy_model, dummy_rr_encoder, dummy_video_encoder
from bachelors_thesis.models.cnn_proj import CNN_Model
# from bachelors_thesis.models.cnn21 import CNN_Model

from bachelors_thesis.train_model import Trainer
from bachelors_thesis.data.dataloader import get_dataloader

@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg):
    
    print('Initializing model training')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    
    model = CNN_Model()
    model = model.to(device)
    
    print('Model initialized')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.params.lr)
    criterion = nn.BCELoss()
    criterion = criterion.to(device)
    
    trainer_object = Trainer(model, device, optimizer, criterion, cfg)
    
    print('Trainer object initialized')
    # from torch.utils.data import DataLoader
    

    
    train_dataloader = get_dataloader(cfg.paths.train_data_path, batch_size=1, device = device)
    validation_dataloader = get_dataloader(cfg.paths.test_data_path, batch_size=1, device = device)
    test_dataloader = get_dataloader(cfg.paths.test_data_path, batch_size=1, device = device)
    
    # shorten dataloader
    train_dataloader = [next(iter(train_dataloader)) for i in range(1)]
    validation_dataloader = [next(iter(validation_dataloader)) for i in range(1)]
    test_dataloader = [next(iter(test_dataloader)) for i in range(1)]
    
    print('Dataloaders loaded')
    model = trainer_object.train_loop(train_dataloader, validation_dataloader)
    
    print('Evaluating')
    evaluation_scores = evaluate(model, test_dataloader, cfg, NUM_CLASSES = 2, device = device)
    
    for key, value in evaluation_scores.items():
        print(f"{key}: {value}")
        
    # save model
    print('Saving model')
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_filename = f"TestRun_{current_datetime}.pt"
    torch.save(model.state_dict(), os.path.join(cfg.paths.model_path, model_filename))
    
    
if __name__ == "__main__":
    main()