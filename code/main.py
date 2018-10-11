import torch
from data import loader
from model import model, device
from loss import get_loss
import os

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
current_dir = os.path.dirname(__file__)
weights_dir = os.path.join(current_dir, '..', 'weights')
def train(epoch):
    print('#'*15)
    print('Epoch {}, Latent Size {}'.format(epoch, model.latent_size))
    print('#'*15)
    model.train()
    for index, (x, _) in enumerate(loader):
        x = x.mean(dim=1, keepdim=True).to(device)
        optimizer.zero_grad()
        x_generated, mu, logvar = model(x)
        loss = get_loss(x_generated, x, mu, logvar)
        loss.backward()
        optimizer.step()
        if index % 100 == 0:
            print('Loss at iteration {0}: {1:.4f}'.format(index, loss.item()))
    if epoch == 4:
        filename = 'epoch{}_ls{}.pkl'.format(epoch, model.latent_size)
        torch.save(model.state_dict(), os.path.join(weights_dir, filename))
    if epoch < 4:
        scheduler.step()

if __name__ == '__main__':
    for epoch in range(5):
        train(epoch)
