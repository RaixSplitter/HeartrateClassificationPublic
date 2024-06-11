import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
import wandb
import hydra
from bachelors_thesis.utils.evaluation import evaluate
from bachelors_thesis.utils.logging import init_logging, log_message, with_default_logging
from datetime import datetime
import os
import logging
from bachelors_thesis.models.model import dummy_model, dummy_rr_encoder, dummy_video_encoder
# from bachelors_thesis.models.cnn import CNN_Model
from bachelors_thesis.models.cnn_proj import CNN_Proj_Model
# from bachelors_thesis.models.cnn21 import CNN21_Model
from bachelors_thesis.data.dataloader import get_dataloader


class Trainer():
    def __init__(self, model, device, optimizer, criterion, cfg):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.running_train_loss = 0
        self.epochs = cfg.params.epochs
        self.cfg = cfg
        self.class_distribution = {}
    
    def update_class_distribution(self, preds):
        """
        Update the class distribution
        
        ARGS:
            preds: PyTorch tensor
        """
        preds = preds.round()
        for pred in preds:
            pred = pred.item()
            if pred in self.class_distribution:
                self.class_distribution[pred] += 1
            else:
                self.class_distribution[pred] = 1
        
    
        
    def train_batch(self, data_batch):
        """
        Train a model for one batch
        
        ARGS:
            data_batch: Tuple of tensors
        RETURNS:
            model: PyTorch model
        """
        videos1, videos2, hr_features1, hr_features2, labels1, labels2 = data_batch
        
        
        self.optimizer.zero_grad()
        
        # Target person 1
        # Forward pass
        preds = self.model(videos1, videos2, hr_features1)
        preds = torch.reshape(preds, (-1,))
        self.update_class_distribution(preds)
        
        # Calculate loss
        loss = self.criterion(preds, labels1)
        
        # Backward pass to calculate gradients
        loss.backward()
        
        # Get loss
        train_loss = loss.item()
        self.running_train_loss += train_loss
        
        # Update weights
        self.optimizer.step()
        
        # Target person 2
        # Forward pass
        preds = self.model(videos1, videos2, hr_features2)
        preds = torch.reshape(preds, (-1,))
        self.update_class_distribution(preds)
        
        
        # Calculate loss
        loss = self.criterion(preds, labels2)
        
        # Backward pass to calculate gradients
        loss.backward()
        
        # Get loss
        train_loss = loss.item()
        self.running_train_loss += train_loss
        
        # Update weights
        self.optimizer.step()
        
        return None

    @with_default_logging(None)
    def train_step(self, train_dataloader, epoch : int = 0):
        """
        Trains a model for one epoch
        
        ARGS:
            train_dataloader: PyTorch dataloader

        
        RETURNS: 
            model: PyTorch models
            running_train_loss: float
            last_train_loss: float

        """
        

        #Prepares model for training
        self.model.train()
        self.running_train_loss = 0
        
        # Done training
        log_msg = f'--- Epoch {epoch}/{self.cfg.params.epochs} ---'
        wandb_data = {'train_loss': self.running_train_loss / len(train_dataloader)}

        for batch_idx, data_batch in enumerate(train_dataloader):
            self.train_batch(data_batch)
            
                
        return None, (log_msg, wandb_data)
    
    
    @with_default_logging(None)
    def validate_step(self, validation_dataloader, validation_loss = 0):
        """
        Validate a model given the model and a validation dataloader
        
        ARGS:
            validation_dataloader: PyTorch dataloader
            criterion: PyTorch loss function
            validation_loss: float
        
        RETURNS:
            accuracy: float
            validation_loss: float
        """
        
        self.model.eval()
        with torch.no_grad():
            correct_predictions = 0
            total_samples = 0
            for batch_idx, data_batch in enumerate(validation_dataloader):                
                videos1, videos2, hr_features1, hr_features2, labels1, labels2 = data_batch
                
                # Target person 1
                preds = self.model(videos1, videos2, hr_features1)
                preds = torch.reshape(preds, (-1,))
                
                loss = self.criterion(preds, labels1)
                validation_loss += loss
                
                correct_predictions += torch.sum(preds.round() == labels1).item()
                total_samples += len(labels1)
                
                # Target person 2
                preds = self.model(videos1, videos2, hr_features2)
                preds = torch.reshape(preds, (-1,))
                
                loss = self.criterion(preds, labels2)
                validation_loss += loss
                
                correct_predictions += torch.sum(preds.round() == labels2).item()
                total_samples += len(labels1)
                
                

        accuracy = correct_predictions / total_samples
        
        log_msg = f'--- | accuracy: {accuracy} | val loss: {validation_loss / len(validation_dataloader)} ---\n'
        wandb_data = {
            "validation_accuracy" : accuracy,
            "validation_loss" : validation_loss / len(validation_dataloader)
        }

        return (accuracy, validation_loss), (log_msg, wandb_data)
    
    @with_default_logging("Start training")
    def train_loop(self, train_dataloader, validation_dataloader):
        
        
        for epoch in range(self.epochs):

            validation_loss = 0

            # Start training step
            self.train_step(train_dataloader, epoch)
            
            # Start validation step
            accuracy, validation_loss = self.validate_step(validation_dataloader, validation_loss)
        
        log_msg = f"Finished training. Accuracy: {accuracy}\nClass distribution: {self.class_distribution}"
        wandb_data = {f"class_distribution_{label}":count for label, count in self.class_distribution.items()}
        return self.model, (log_msg, wandb_data)
    
@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg):
    
    init_logging(cfg)
    
    log_message(logging.getLogger(__name__), f"CUDA: {torch.cuda.is_available()}", {'cuda': torch.cuda.is_available()})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    
    model = CNN_Proj_Model()
    model = model.to(device)
    
    # Set seed for torch, numpy and random
    torch.manual_seed(cfg.base.seed)
    torch.cuda.manual_seed(cfg.base.seed)
    torch.cuda.manual_seed_all(cfg.base.seed)
    np.random.seed(cfg.base.seed)
    
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.params.lr)
    criterion = nn.BCELoss()
    criterion = criterion.to(device)
    
    trainer_object = Trainer(model, device, optimizer, criterion, cfg)
    
    # from torch.utils.data import DataLoader
    

    
    train_dataloader = get_dataloader(cfg.paths.train_data_path, batch_size=cfg.params.batch_size, device = device)
    validation_dataloader = get_dataloader(cfg.paths.test_data_path, batch_size=cfg.params.batch_size, device = device)
    test_dataloader = get_dataloader(cfg.paths.test_data_path, batch_size=cfg.params.batch_size, device = device)
    
    # shorten dataloader
    # train_dataloader = [next(iter(train_dataloader)) for i in range(2)]
    # validation_dataloader = [next(iter(validation_dataloader)) for i in range(2)]
    # test_dataloader = [next(iter(test_dataloader)) for i in range(2)]
    
    
    model = trainer_object.train_loop(train_dataloader, validation_dataloader)
    
    
    
    
    evaluation_scores = evaluate(model, test_dataloader, cfg, NUM_CLASSES = 2, device = device)
    
    for key, value in evaluation_scores.items():
        wandb.run.summary[key] = value
        
    # save model
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_filename = f"{wandb.run.name}_{current_datetime}.pt"
    torch.save(model.state_dict(), os.path.join(cfg.paths.model_path, model_filename))
    print(summary(model))
    wandb.finish()
    
    
if __name__ == "__main__":
    main()
    