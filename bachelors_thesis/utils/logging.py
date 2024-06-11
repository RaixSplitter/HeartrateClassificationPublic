from dotenv import load_dotenv
import wandb
import logging
import functools
import os
import sys



def init_logging(cfg):
    # Create super basic logger
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    # logger = logging.getLogger(__name__)
    
    # only needs to be called for local development
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project = "bachelor HRC",
        config = {
            "learning_rate": cfg.params.lr,
            "epochs": cfg.params.epochs,
            "seed": cfg.base.seed,
            "batch_size": cfg.params.batch_size,
            "architecture": cfg.base.architecture,
            "dataset": cfg.base.dataset,
        },
        mode=cfg.base.wandb_mode
    )

def log_message(logger, msg : str, wandb_data : dict):
    """
    Logs a message to the logger and sends metadata to wandb
    
    ARGS:
        logger: logging object
        msg: str
        wandb_data: dict
        
    Returns: None
    """
    
    # Log
    if msg:
        logger.info(msg)
    
    # Wandb data
    if wandb_data and wandb.run is not None:
        wandb.log(wandb_data)
        
        
def with_default_logging(log_msg : str):
    """
    Decorator that is able to log a message before and after a function is called
    """
    logger = logging.getLogger(__name__)
    if log_msg:
        logger.info(f"{log_msg}")
    def logging_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            
            
            result, logging_data = func(*args, **kwargs)
            log_msg, wandb_data = logging_data
            log_message(logger, log_msg, wandb_data) 
            return result
        
        return wrapper
    return logging_decorator