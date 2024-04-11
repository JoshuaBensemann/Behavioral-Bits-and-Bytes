from .data import get_testing_dataset
from .eval import evaluate_model


def run_evaluation(
    model, eval_dir, device, train_classes, test_type="eval", batch_size=1, verbose=True
):
    eval_dataloader, eval_classes = get_testing_dataset(
        eval_dir, transform_set=test_type, batch_size=batch_size, verbose=verbose
    )
    results = evaluate_model(
        model,
        eval_dataloader,
        device,
        train_classes=train_classes,
        eval_classes=eval_classes,
        verbose=verbose,
    )

    return results
