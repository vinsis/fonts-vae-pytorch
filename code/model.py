import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VAE(nn.Module):
    def __init__(self, latent_size=20):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        #################
        #ENCODING LAYERS#
        #################
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) #1,32,32 -> 16,32,32
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)  #16,32,32 -> 32,16,16
        self.bn2 = nn.BatchNorm2d(32)

        # self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)    #32,16,16 -> 32,16,16
        # self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)  #32,16,16 -> 64,8,8
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)   #64,8,8 -> 128,4,4
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)    #128,4,4 -> 256,2,2
        self.bn6 = nn.BatchNorm2d(256)

        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.fc1 = nn.Linear(256, latent_size)
        self.fc2 = nn.Linear(256, latent_size)

        #################
        #DECODING LAYERS#
        #################
        self.fc3 = nn.Linear(latent_size, 256)
        self.fc4 = nn.Linear(256, 4096)
        self.conv7 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.Conv2d(16, 1, kernel_size=3, padding=1)


    def encode(self, x):
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.relu(self.bn2(self.conv2(output)))
        # output = self.relu(self.bn3(self.conv3(output)))
        output = self.relu(self.bn4(self.conv4(output)))
        output = self.relu(self.bn5(self.conv5(output)))
        output = self.relu(self.bn6(self.conv6(output)))
        output = self.avgpool(output).view(x.size(0), -1)
        return self.fc1(output), self.fc2(output)

    def decode(self, z):
        output = self.relu(self.fc3(z))
        output = self.relu(self.fc4(output))
        output = output.view(z.size(0), 16, 16, 16)
        output = self.bn7(self.conv7(output))
        output = nn.functional.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
        output = self.sigmoid(self.conv8(output))
        return output

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE(1).to(device)

if __name__  == '__main__':
    x = torch.randn(10, 1, 32, 32)
    with torch.no_grad():
        y = model(x)
        for tensor in y:
            print(tensor.size())
