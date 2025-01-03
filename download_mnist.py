import torchvision
import torchvision.transforms as transforms
import os

def download_mnist_data():
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Train Phase transformations
    train_transforms = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])

    # Test Phase transformations
    test_transforms = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])

    # Download training dataset
    train_dataset = torchvision.datasets.MNIST('./data', 
                                             train=True, 
                                             download=True, 
                                             transform=train_transforms)
    
    # Download test dataset
    test_dataset = torchvision.datasets.MNIST('./data', 
                                            train=False, 
                                            download=True, 
                                            transform=test_transforms)

    print("MNIST dataset downloaded successfully!")
    return train_dataset, test_dataset

if __name__ == "__main__":
    download_mnist_data() 