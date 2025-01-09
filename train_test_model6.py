from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from utils.logger import setup_logger

# Set up logger
logger = setup_logger()

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution: Depthwise conv + Pointwise conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,  # Each input channel is convolved separately
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,  # 1x1 convolution
            bias=False
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Net(nn.Module):
    def __init__(self, dropout_value=0.1):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=10, out_channels=32, kernel_size=(3, 3), padding=0),
            nn.ReLU(),            
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0),
            nn.ReLU(),            
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        self.convblock6 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0),
            nn.ReLU(),            
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        self.convblock7 = nn.Sequential(
            DepthwiseSeparableConv(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),            
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)        
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

# Training and testing functions remain same as model 5
train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = F.nll_loss(y_pred, target)
        train_losses.append(loss)
        loss.backward()
        optimizer.step()
        
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        desc_status = f'Loss={round(loss.item(), 15)} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}'
        pbar.set_description(desc=desc_status)
        train_acc.append(100*correct/processed)
    logger.info(desc_status)


def test_epoch(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    accuracy = 100. * correct / len(test_loader.dataset)
    log_msg = f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)'
    logger.info(log_msg)
    
    test_acc.append(accuracy)
    return accuracy

def save_model(model, optimizer, epoch, train_acc, test_acc, save_dir='model_checkpoints'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_accuracy': train_acc,
        'test_accuracy': test_acc
    }
    
    checkpoint_path = os.path.join(save_dir, f'mnist_model_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    best_model_path = os.path.join(save_dir, 'best_mnist_model.pth')
    if not os.path.exists(best_model_path) or test_acc > checkpoint.get('test_accuracy', 0):
        torch.save(checkpoint, best_model_path)

if __name__ == "__main__":
    # Training setup
    SEED = 1
    use_cuda = torch.cuda.is_available()
    logger.info(f"CUDA Available? {use_cuda}")
    
    torch.manual_seed(SEED)
    if use_cuda:
        torch.cuda.manual_seed(SEED)

    # Data loading
    train_transforms = transforms.Compose([
        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=train_transforms)
    test_dataset = datasets.MNIST('./data', train=False, download=False, transform=test_transforms)
    
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if use_cuda else dict(shuffle=True, batch_size=64)
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)

    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    EPOCHS = 15
    logger.info("Starting training...")
    
    for epoch in range(EPOCHS):
        logger.info(f"EPOCH: {epoch}")
        train_epoch(model, device, train_loader, optimizer, epoch)
        test_accuracy = test_epoch(model, device, test_loader)
        
        save_model(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_acc=train_acc[-1],
            test_acc=test_accuracy,
            save_dir='model_checkpoints'
        )
    
    logger.info("Training completed!")