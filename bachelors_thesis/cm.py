from torchinfo import summary
from bachelors_thesis.models.cnn import CNN_Model 
from bachelors_thesis.models.cnn_proj import CNN_Proj_Model
from bachelors_thesis.models.cnn21 import CNN21_Model

import torch
from bachelors_thesis.data.dataloader import get_dataloader
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

print('Loading model...')
# Load in model
cnn_model = CNN_Model()
# proj_model = CNN_Proj_Model()

cnn_model.load_state_dict(torch.load("C:/Programmering/DTU/Heartrate_Classification/models/silver-star-88_2024-06-09_08-37-32.pt"))
# cnn_model.load_state_dict(torch.load("C:/Programmering/DTU/Heartrate_Classification/models/unique-snowball-89_2024-06-09_06-49-49.pt"))

cnn_model.to("cuda")
cnn_model.eval()
print('Model loaded')

test_dataloader = get_dataloader('C:/Programmering/DTU/Heartrate_Classification/data/mappings/test_pairs.csv', batch_size=1, device = 'cuda')
print('Data loaded')

y_preds = np.zeros(2*len(test_dataloader.dataset))
y_true = np.zeros(2*len(test_dataloader.dataset))
for batch_idx, data_batch in tqdm(enumerate(test_dataloader), total = len(test_dataloader)):
            
    #Target person 1
    videos1, videos2, hr_features1, hr_features2, labels1, labels2 = data_batch = data_batch
    preds = cnn_model(videos1, videos2, hr_features1)
    preds = torch.reshape(preds, (-1,))

    y_preds[batch_idx*2] = preds
    y_true[batch_idx*2] = labels1
    
    #Target person 2
    videos1, videos2, hr_features1, hr_features2, labels1, labels2 = data_batch = data_batch
    preds = cnn_model(videos1, videos2, hr_features2)
    preds = torch.reshape(preds, (-1,))
    
    y_preds[batch_idx*2+1] = preds
    y_true[batch_idx*2+1] = labels2
cm = confusion_matrix(y_true, y_preds)

np.save("y_preds.npy", y_preds)
np.save("y_true.npy", y_true)
np.save("confusion_matrix.npy", cm)