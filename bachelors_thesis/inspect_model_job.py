from torchinfo import summary
import wandb
import hydra
from bachelors_thesis.utils.logging import init_logging, log_message, with_default_logging
import logging
from bachelors_thesis.models.cnn import CNN_Model
from bachelors_thesis.models.cnn_proj import CNN_Proj_Model
from bachelors_thesis.models.cnn21 import CNN21_Model


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg):
    
    init_logging(cfg)
    
    batch_size = cfg.params.batch_size
    input_shape = [(batch_size, 300, 480, 640), (batch_size, 300, 480, 640), (batch_size, 2)]
    
    log_message(logging.getLogger(__name__), summary(CNN_Model(), input_shape), None)
    log_message(logging.getLogger(__name__), summary(CNN_Proj_Model(), input_shape), None)
    log_message(logging.getLogger(__name__), summary(CNN21_Model(), input_shape), None)
    
    wandb.finish()
    
    
if __name__ == "__main__":
    main()
    