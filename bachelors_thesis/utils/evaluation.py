import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from bachelors_thesis.utils.logging import with_default_logging
   
class Evaluation_Metrics():
    """
    Class for evaluating the model
    """
    
    def __init__(self, device = 'cpu') -> None:
        self.device = device
        self.y_pred = torch.tensor([])
        self.y_true = torch.tensor([])
    
    
    def update(self, y_pred, y_true):
        """
        Update metrics on the model
        
        ARGS:
            y_pred: PyTorch tensor
            y_true: PyTorch tensor
        
        RETURNS:
            None
        """
        y_pred = y_pred.to('cpu')
        y_true = y_true.to('cpu')
        self.y_pred = torch.cat((self.y_pred, y_pred), 0)
        self.y_true = torch.cat((self.y_true, y_true), 0)
        
    def compute(self):
        """
        Compute metrics on the model
        
        ARGS:
            None
        
        RETURNS:
            dict
        """
        return {
            "test_accuracy": accuracy_score(self.y_true, self.y_pred),
            "test_precision": precision_score(self.y_true, self.y_pred, average='macro'),
            "test_recall": recall_score(self.y_true, self.y_pred, average='macro'),
            "test_f1": f1_score(self.y_true, self.y_pred, average='macro'),
            # "test_confusion_matrix": confusion_matrix(self.y_true, self.y_pred)
        }

@with_default_logging("Start evaluation")
def evaluate(model, test_dataloader, cfg, NUM_CLASSES = 25, device = 'cpu'):
    
    metrics = Evaluation_Metrics(device)
    
    
    with torch.no_grad():
        model.eval()
        

        

        
        for batch_idx, data_batch in enumerate(test_dataloader):
            
            #Target person 1
            videos1, videos2, hr_features1, hr_features2, labels1, labels2 = data_batch = data_batch
            preds = model(videos1, videos2, hr_features1)
            preds = torch.reshape(preds, (-1,))

            metrics.update(preds.round(), labels1)
            
            #Target person 2
            videos1, videos2, hr_features1, hr_features2, labels1, labels2 = data_batch = data_batch
            preds = model(videos1, videos2, hr_features2)
            preds = torch.reshape(preds, (-1,))
            metrics.update(preds.round(), labels2)
            
    evaluation_scores = metrics.compute()
    
    wandb_data = evaluation_scores
    log_msg = "\n".join([
    "\n--- SCORING METRICS ON TEST DATA ---",
    f"--- Accuracy: {evaluation_scores['test_accuracy']} ---",
    f"--- Precision: {evaluation_scores['test_precision']} ---",
    f"--- Recall: {evaluation_scores['test_recall']} ---",
    f"--- F1: {evaluation_scores['test_f1']} ---",
    ])
    
    return (evaluation_scores), (log_msg, wandb_data)