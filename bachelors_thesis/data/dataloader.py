import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

class HRV_Dataset(Dataset):
    def __init__(self, dataset_path, device = 'cpu'):
        self.dataset_path = dataset_path
        self.device = device
        
        self.df = pd.read_csv(self.dataset_path)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        sample = self.df.iloc[index] # Get the row at the index
        

    # columns=[
    #     "sample_id1",
    #     "sample_id2",
    #     "video_path1",
    #     "video_path2",
    #     "hr_path1",
    #     "hr_path2"
    #     "rr_interval1_path1",
    #     "rr_interval2_path1",
    #     "rr_interval3_path1",
    #     "rr_interval1_path2",
    #     "rr_interval2_path2",
    #     "rr_interval3_path2",
    #     "times_path1",
    #     "times_path2",

        #Get paths
        video_path1 = sample["video_path1"]
        video_path2 = sample["video_path2"]
        rr_interval_path1 = sample["rr_interval3_path1"]
        rr_interval_path2 = sample["rr_interval3_path2"]
        

        #Load in data
        video1 = torch.load(video_path1)
        video2 = torch.load(video_path2)
        rr_interval1 = np.load(rr_interval_path1)
        rr_interval2 = np.load(rr_interval_path2)
        
        #Compute features
        mean_rr1 = np.mean(rr_interval1)
        std_rr1 = np.std(rr_interval1)
        cv_rr1 = std_rr1/mean_rr1
        
        mean_rr2 = np.mean(rr_interval2)
        std_rr2 = np.std(rr_interval2)
        cv_rr2 = std_rr2/mean_rr2
        
        #Convert to tensors with correct dtype
        video1 = video1.float()
        video2 = video2.float()
        rr_interval1 = torch.tensor(rr_interval1, dtype=torch.float32)
        rr_interval2 = torch.tensor(rr_interval2, dtype=torch.float32)
        label1 = torch.tensor(0, dtype=torch.float32)
        label2 = torch.tensor(1, dtype=torch.float32)
        hr_features1 = torch.tensor([mean_rr1, cv_rr1], dtype=torch.float32)
        hr_features2 = torch.tensor([mean_rr2, cv_rr2], dtype=torch.float32)
        
        video1 = video1.to(self.device)
        video2 = video2.to(self.device)
        hr_features1 = hr_features1.to(self.device)
        hr_features2 = hr_features2.to(self.device)
        label1 = label1.to(self.device)
        label2 = label2.to(self.device)
        return video1, video2, hr_features1, hr_features2, label1, label2
    
    def __len__(self):
        return len(self.df)

def get_dataloader(dataset_path, batch_size = 32, shuffle = True, device = 'cpu'):
    HRV = HRV_Dataset(dataset_path, device = device)
    HRV = DataLoader(HRV, batch_size = batch_size, shuffle = shuffle)
    return HRV
         
if __name__ == "__main__":
    HRV = get_dataloader("C:/Programmering/DTU/Heartrate_Classification/data/mappings/train_pairs.csv")
    data_batch = next(iter(HRV))
    
    for i in data_batch:
        print(i.shape)