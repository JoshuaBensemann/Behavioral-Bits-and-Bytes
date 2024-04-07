import torch
import torch.nn as nn
import torch.optim as optim


def training_setup(model, classes, lr=0.1, opt_method=None):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device}")

    # Freeze all the parameters in the pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Modify the last fully connected layer to match the number of classes in your dataset
    num_classes = len(classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    if opt_method is None:
        optimizer = optim.SGD(model.fc.parameters(), lr=lr)
    else:
        optimizer = opt_method(model.fc.parameters(), lr=lr)

    return model, device, criterion, optimizer


def train_model(model, dataset, device, criterion, optimizer):
    running_loss = 0.0
    correct = 0
    total = 0

    # Set the model to training mode
    model.train()

    for images, labels in dataset:
        # Move the images and labels to the GPU if available
        images = images.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        # Update the running loss
        running_loss += loss.item()

        # Compute the predicted labels
        _, predicted = torch.max(outputs.data, 1)

        # Update the total and correct predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Compute the accuracy
    accuracy = 100 * correct / total
    return accuracy, running_loss
