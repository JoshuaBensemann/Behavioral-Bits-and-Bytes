import torch
from torch.nn.functional import softmax


def evaluate_model(model, eval_dataloader, device, train_classes, eval_classes):
    results = []

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Turn off gradients for prediction, saves memory and computations
        for images, labels in eval_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            softmax_output = softmax(outputs, dim=1).cpu().numpy()[:, predicted]
            result = {
                "label": eval_classes[labels],
                "predicted": train_classes[predicted],
                "prob": round(softmax_output[0] * 100, 2),
            }
            print(
                f"Actual: {result['label']}, Predicted: {result['predicted']} {result['prob']}%"
            )
            results.append(result)

    return results
