import torch.nn as nn
import torch.nn.functional as F
import torch

# data shape is (B, 1, 22, 1000)

class MiniResNet(nn.Module):

    def __init__(self, num_classes=4):
        super(MiniResNet, self).__init__()
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
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 4)


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
    
# 100 epoch best
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
        self.pool2 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(0,0))
        self.bn2 = nn.BatchNorm2d(50)
        self.dropout2 = nn.Dropout(0.6)
        
        # Conv. block 3
        self.conv3 = nn.Conv2d(50, 100, kernel_size=(5, 5), padding=2)
        self.pool3 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(0,0))
        self.bn3 = nn.BatchNorm2d(100)
        self.dropout3 = nn.Dropout(0.6)
        
        # Conv. block 4
        self.conv4 = nn.Conv2d(100, 200, kernel_size=(5, 5), padding=2)
        self.pool4 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(0,0))
        self.bn4 = nn.BatchNorm2d(200)
        self.dropout4 = nn.Dropout(0.6)
    
        self.fc1 = nn.Linear(4400, 40) # Adjust the input features according to your output shape after the last pooling
       
        self.lstm = nn.LSTM(40, 40, 5, batch_first=True, dropout=0.6)
        
        # Output layer
        self.fc2 = nn.Linear(40, 4)
        
    def forward(self, x):
        x = x[:, :, :, :600]
        x = x.transpose(-1, -2)
        #print(x.shape)
        
        # Apply conv blocks
        x = self.dropout1(F.elu(self.bn1(self.pool1(self.conv1(x)))))
        x = self.dropout2(F.elu(self.bn2(self.pool2(self.conv2(x)))))
        x = self.dropout3(F.elu(self.bn3(self.pool3(self.conv3(x)))))
        x = self.dropout4(F.elu(self.bn4(self.pool4(self.conv4(x)))))
        # FC+LSTM
        x = x.permute(0, 2, 1, 3)
        x = torch.flatten(x, 2)
        x = F.elu(self.fc1(x))
        x, (hn, cn) = self.lstm(x)
        x = hn[-1] # Use the last hidden state
        
        # Output layer
        x = self.fc2(x)
        return x

class EEGNet(nn.Module):
    def __init__(self, num_classes=4):
        super(EEGNet, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=True),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 5), padding=0),
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(8, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 5), padding=0),
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 5), padding=0),
        )
        self.dropout = nn.Dropout(p=0.6, inplace=False)
        self.flatten = nn.Flatten()
        self.elu = nn.ELU(alpha=1.0)
        self.fc1 = nn.Linear(3840, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 512, bias=True)
        self.fc3 = nn.Linear(512, 512, bias=True)
        self.fc4 = nn.Linear(512, 4, bias=True)

    def forward(self, x):
        x = x[:, :, :, 0:600]

        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.dropout(x)
        x = self.flatten(x)
        #k = x
        #print(x.shape)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.elu(x)
        x = self.fc3(x)
        x = self.elu(x)
        x = self.fc4(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.transformer_model = nn.Transformer(nhead=16, 
                                                d_model=256,
                                                num_encoder_layers=6, 
                                                num_decoder_layers=6, 
                                                dim_feedforward=512, 
                                                dropout=0.7, 
                                                activation='gelu')
        self.fc1 = nn.Linear(22, 256)
        self.fc2 = nn.Linear(256*600, 22)
        self.fc3 = nn.Linear(22, 4)

    def forward(self, x):
        x = x.squeeze(1)[:, :, :600]
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.transformer_model(x, x)
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        return x
    

class HybridCNNTransformerModel(nn.Module):
    def __init__(self):
        super(HybridCNNTransformerModel, self).__init__()
        
        # Conv. block 1
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(5, 5), padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=(1,0))
        self.bn1 = nn.BatchNorm2d(25)
        self.dropout1 = nn.Dropout(0.6)
        
        # Conv. block 2
        self.conv2 = nn.Conv2d(25, 50, kernel_size=(5, 5), padding=2)
        self.pool2 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(0,0))
        self.bn2 = nn.BatchNorm2d(50)
        self.dropout2 = nn.Dropout(0.6)
        
        # Conv. block 3
        self.conv3 = nn.Conv2d(50, 100, kernel_size=(5, 5), padding=2)
        self.pool3 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(0,0))
        self.bn3 = nn.BatchNorm2d(100)
        self.dropout3 = nn.Dropout(0.6)
        
        # Conv. block 4
        self.conv4 = nn.Conv2d(100, 200, kernel_size=(5, 5), padding=2)
        self.pool4 = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(0,0))
        self.bn4 = nn.BatchNorm2d(200)
        self.dropout4 = nn.Dropout(0.6)
    
        self.fc1 = nn.Linear(4400, 40) # Adjust the input features according to your output shape after the last pooling
       
        self.transformer_model = nn.Transformer(nhead=4,
                                                d_model=40,
                                                num_encoder_layers=6, 
                                                num_decoder_layers=6, 
                                                dim_feedforward=80, 
                                                dropout=0.6, 
                                                activation='gelu')
        
        # Output layer
        self.fc2 = nn.Linear(1000, 4)
        
    def forward(self, x):
        x = x[:, :, :, :600]
        x = x.transpose(-1, -2)
        #print(x.shape)
        
        # Apply conv blocks
        x = self.dropout1(F.elu(self.bn1(self.pool1(self.conv1(x)))))
        x = self.dropout2(F.elu(self.bn2(self.pool2(self.conv2(x)))))
        x = self.dropout3(F.elu(self.bn3(self.pool3(self.conv3(x)))))
        x = self.dropout4(F.elu(self.bn4(self.pool4(self.conv4(x)))))
        # FC+LSTM
        x = x.permute(0, 2, 1, 3)
        x = torch.flatten(x, 2)
        x = F.elu(self.fc1(x))
        x = self.transformer_model(x, x + torch.rand_like(x))
        x = torch.flatten(x, 1)
        
        # Output layer
        x = self.fc2(x)
        return x
    
class EEGAttentionNet(nn.Module):
    def __init__(self, num_classes=4):
        super(EEGAttentionNet, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=True),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(8, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=1.0),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
        )
        self.dropout = nn.Dropout(p=0.6, inplace=False)
        self.flatten = nn.Flatten()
        self.elu = nn.ELU(alpha=1.0)
        self.k = nn.Linear(960, 512)
        self.q = nn.Linear(960, 512)
        self.v = nn.Linear(960, 512)
        self.layer_norm = nn.LayerNorm(512)
        self.fc1 = nn.Linear(4608, 512, bias=True)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.fc3 = nn.Linear(512, 4, bias=True)


    def forward(self, x):
        x = x[:, :, :, 0:600]

        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.dropout(x)
        x = x.permute(0, 3, 1, 2)
        x = torch.flatten(x, 2)
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)
        attention = torch.matmul(q, k.transpose(-2, -1))
        attention = F.softmax(attention, dim=-1)
        x = torch.matmul(attention, v)
        x = self.layer_norm(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x