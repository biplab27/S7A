import torch
from torchsummary import summary
from train_test_mnist import Net

def display_model_summary():
    # Set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create model instance
    model = Net().to(device)
    
    # Display summary with input size matching MNIST dimensions (1x28x28)
    print("\nModel Architecture Summary:")
    print("=" * 80)
    summary(model, input_size=(1, 28, 28))
    
    # Display receptive field calculations
    print("\nReceptive Field Calculations:")
    print("=" * 80)
    print("{:<15} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(
        "Layer", "RF", "n_in", "n_out", "j_in", "j_out", "r_in", "r_out"
    ))
    print("-" * 80)
    
    # Layer-by-layer RF calculations
    rf_data = [
        ("Input", "1x1", "28", "28", "1", "1", "0", "0"),
        ("Conv1 (3x3)", "3x3", "28", "26", "1", "1", "0", "1"),
        ("Conv2 (3x3)", "5x5", "26", "24", "1", "1", "1", "2"),
        ("Conv3 (3x3)", "7x7", "24", "22", "1", "1", "2", "3"),
        ("MaxPool", "14x14", "22", "11", "1", "2", "3", "6"),
        ("Conv4 (1x1)", "14x14", "11", "11", "2", "2", "6", "6"),
        ("Conv5 (3x3)", "18x18", "11", "9", "2", "2", "6", "8"),
        ("Conv6 (3x3)", "22x22", "9", "7", "2", "2", "8", "10"),
        ("Conv7 (1x1)", "22x22", "7", "7", "2", "2", "10", "10"),
        ("Conv8 (7x7)", "28x28", "7", "1", "2", "14", "10", "16")
    ]
    
    for layer_data in rf_data:
        print("{:<15} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(*layer_data))

if __name__ == "__main__":
    display_model_summary() 