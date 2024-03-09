import torch.nn as nn
import torch.nn.functional as F
import torch

class ResNet(nn.Module):

    def __init__(self, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 16
        # Initial convolution
        self.conv1 = nn.Conv2d(1,
                               32,
                               kernel_size=(1, 10),
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 10), stride=1, padding=0)

        self.layer1 = nn.Sequential(ResidualBlock(32, 64, downsample=True), ResidualBlock(64, 128, downsample=True))
        self.layer2 = nn.Sequential(ResidualBlock(128, 256, downsample=True), ResidualBlock(256, 512, downsample=True))
        # self.layer3 = nn.Sequential(ResidualBlock(128, 256, downsample=True), ResidualBlock(256, 256))
        # self.layer4 = nn.Sequential(ResidualBlock(256, 512, downsample=True), ResidualBlock(512, 512))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 256)
        self.lstm = nn.LSTM(256, 128, 2, batch_first=True)
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        x = self.layer1(x)

        x = self.layer2(x)


        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = x.unsqueeze(0)
        x, (hn, cn) = self.lstm(x)
        x = self.fc2(x)
        x = x.squeeze(0)
        

        return x

    
class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample

        if self.downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), bias=False)
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 2), padding=0, bias=False)
        else:
            self.conv1  = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
        self.conv2  = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)


    def forward(self, x):

        conv1_out = self.relu(self.bn1(self.conv1(x)))

        conv2_out = self.bn2(self.conv2(conv1_out))

        if self.downsample:
            residual = self.bn3(self.conv3(x))
        else:
            residual = x


        out = self.relu(conv2_out + residual)

        return out
    

class HybridCNNLSTMModel(nn.Module):
    def __init__(self):
        super(HybridCNNLSTMModel, self).__init__()
        
        # Conv. block 1
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(5, 5), padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=(1,0))
        self.bn1 = nn.BatchNorm2d(25)
        self.dropout1 = nn.Dropout(0.6)
        
        # Conv. block 2
        self.conv2 = nn.Conv2d(25, 50, kernel_size=(5, 5), padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=(1,0))
        self.bn2 = nn.BatchNorm2d(50)
        self.dropout2 = nn.Dropout(0.6)
        
        # Conv. block 3
        self.conv3 = nn.Conv2d(50, 100, kernel_size=(5, 5), padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=(1,0))
        self.bn3 = nn.BatchNorm2d(100)
        self.dropout3 = nn.Dropout(0.6)
        
        # Conv. block 4
        self.conv4 = nn.Conv2d(100, 200, kernel_size=(5, 5), padding=2)
        self.pool4 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=(1,0))
        self.bn4 = nn.BatchNorm2d(200)
        self.dropout4 = nn.Dropout(0.6)
        
        # FC+LSTM layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(57200, 40) # Adjust the input features according to your output shape after the last pooling
        self.lstm = nn.LSTM(40, 10, batch_first=True, dropout=0.4)
        
        # Output layer
        self.fc2 = nn.Linear(10, 4)
        
    def forward(self, x):
        # Apply conv blocks
        x = self.dropout1(F.elu(self.bn1(self.pool1(self.conv1(x)))))
        x = self.dropout2(F.elu(self.bn2(self.pool2(self.conv2(x)))))
        x = self.dropout3(F.elu(self.bn3(self.pool3(self.conv3(x)))))
        x = self.dropout4(F.elu(self.bn4(self.pool4(self.conv4(x)))))
        
        # FC+LSTM
        x = self.flatten(x)
        x = F.elu(self.fc1(x))
        x = x.unsqueeze(1)
        x, (hn, cn) = self.lstm(x)
        x = hn[-1] # Use the last hidden state
        
        # Output layer
        x = self.fc2(x)
        return F.softmax(x, dim=1)