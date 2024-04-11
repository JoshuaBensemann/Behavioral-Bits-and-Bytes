import torch
from torch.nn.functional import softmax


def evaluate_model(
    model,
    eval_dataloader,
    device,
    train_classes,
    eval_classes,
    eval_dataset=None,
    verbose=True,
):
    results = []
    filenames = None

    if eval_dataset is not None:
        filenames = [sample[0].split("/")[-1] for sample in eval_dataset.samples]

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Turn off gradients for prediction, saves memory and computations
        for i, (images, labels) in enumerate(eval_dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            softmax_output = softmax(outputs, dim=1).cpu().numpy()[:, predicted]
            result = {
                "label": eval_classes[labels],
                "predicted": train_classes[predicted],
                "match": (
                    "Correct"
                    if train_classes[predicted] in eval_classes[labels]
                    else "Incorrect"
                ),
                "prob": round(softmax_output[0] * 100, 2),
                "filename": filenames[i] if filenames is not None else "",
            }
            if verbose:
                print(
                    f"Label: {result['label']}, Predicted: {result['predicted']}, Prob: {result['prob']}%, Match: {result['match']}, {result['filename']}"
                )
            results.append(result)

    return results
