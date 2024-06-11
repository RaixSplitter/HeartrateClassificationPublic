import torch
import torch.nn as nn

class Conv2Plus1D(nn.Module):
    def __init__(self, in_channels, filters : int, kernel_size : tuple[int] = (3,3,3), stride : tuple[int] = (1, 1, 1)):
        super(Conv2Plus1D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, filters, kernel_size=(kernel_size[0], kernel_size[1], 1), stride = (stride[0], stride[1], 1))
        self.conv2 = nn.Conv3d(filters, filters, kernel_size=(1,1, kernel_size[2]), stride = (1, 1, stride[2]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
        
class CNN_Video_Encoder(nn.Module):
    def __init__(self):
        super(CNN_Video_Encoder, self).__init__()
        self.conv1 = Conv2Plus1D(1, 4, kernel_size=(3,3,3), stride = (1,1,1)) 
        self.proj1 = nn.Conv3d(4, 2, kernel_size=(1,1,1), stride = (1,1,1))
        self.conv2 = Conv2Plus1D(2, 8, kernel_size=(3,3, 3), stride = (1,1,1))
        self.proj2 = nn.Conv3d(8, 4, kernel_size=(1,1,1), stride = (1,1,1)) 
        self.conv3 = Conv2Plus1D(4, 16, kernel_size=(3,3, 3), stride = (1,1,1))
        self.proj3 = nn.Conv3d(16, 4, kernel_size=(1,1,1), stride = (1,1,1)) 

        # self.pool = nn.MaxPool3d(2, 2)
        self.pool = nn.AvgPool3d(2, 2)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = x.float()  # Convert input tensor to float type
        x = self.conv1(x)
        x = self.proj1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.proj2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.proj3(x)
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

class CNN21_Model(nn.Module):
    def __init__(self):
        super(CNN21_Model, self).__init__()
        self.video_encoder = CNN_Video_Encoder()
        self.rr_encoder = CNN_RR_Encoder()

        self.dense_video = nn.Sequential(
            # nn.Linear(274176, 4),
            nn.Linear(1266720, 4),
            
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

