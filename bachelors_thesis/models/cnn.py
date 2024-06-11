import torch
import torch.nn as nn

class CNN_Video_Encoder(nn.Module):
    def __init__(self):
        super(CNN_Video_Encoder, self).__init__()
        self.conv1 = nn.Conv3d(1, 4, kernel_size=(5,5, 5), stride = (2,2,1)) 
        self.conv2 = nn.Conv3d(4, 8, kernel_size=(3,3, 5), stride = 1) 
        self.conv3 = nn.Conv3d(8, 16, kernel_size=(3,3, 5), stride = 1) 

        self.pool = nn.MaxPool3d(2, 2)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = x.float()  # Convert input tensor to float type
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.relu(x)
        # Reshape tensor to (batchsize, features)
        x = x.view(x.size(0), -1)
        return x
    
class CNN_RR_Encoder(nn.Module):
    def __init__(self):
        super(CNN_RR_Encoder, self).__init__()

    def forward(self, x):
        return x

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.video_encoder = CNN_Video_Encoder()
        self.rr_encoder = CNN_RR_Encoder()

        self.dense_video = nn.Sequential(
            # nn.Linear(274176, 4),
            nn.Linear(1157632, 4),
            
        )

        self.fc1 = nn.Linear(6, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, video1, video2, hr):
        # Videos shape (batchsize, frames, height, width) to (batchsize, channel, frames, height, width)
        video1 = video1.unsqueeze(1)
        video2 = video2.unsqueeze(1)

        video1 = self.video_encoder(video1)
        video2 = self.video_encoder(video2)
        hr = self.rr_encoder(hr)

        # Concat videos
        videos1 = torch.cat((video1, video2), dim=1)
        videos2 = torch.cat((video2, video1), dim=1)

        del video1, video2
        
        # Dense video layer
        videos1 = self.relu(self.dense_video(videos1))
        videos2 = self.relu(self.dense_video(videos2))


        # videos change view to (batchsize, features)
        videos1 = videos1.view(videos1.size(0), -1)
        videos2 = videos2.view(videos2.size(0), -1)

        x1 = torch.cat((videos1, hr), dim=1)
        x2 = torch.cat((videos2, hr), dim=1)

        del videos1, videos2, hr


        x1 = self.fc1(x1)
        x2 = self.fc1(x2)

        x1 = self.sigmoid(x1)
        x2 = self.sigmoid(x2)

        x = torch.mean(torch.stack([x1, x2]), dim=0)

        return x

