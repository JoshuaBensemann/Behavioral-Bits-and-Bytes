from .data import get_testing_dataset
from .eval import evaluate_model
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


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


def summarise_results(results):
    label_results = {}
    results_df = pd.DataFrame(results).set_index('filename')

    for label, df in results_df.groupby('label'):
        most_common_mistake = df.loc[df['match'] == 'Incorrect']['predicted'].mode()
        try:
            accuracy = df.match.value_counts().loc["Correct"] / df.match.value_counts().sum()
        except KeyError:
            accuracy = 0

        label_results[label] = {
            "accuracy": accuracy,
            "correct_confidence": df.loc[df['match'] == 'Correct']['prob'].mean(),
            "incorrect_confidence": df.loc[df['match'] == 'Incorrect']['prob'].mean(),
            "common_mistake": most_common_mistake.loc[0] if len(most_common_mistake) > 0 else np.nan,
        }

    results_summary = pd.DataFrame(label_results).T

    return {
        "accuracy": accuracy_score(results_df['label'], results_df['predicted']),
        "precision": precision_score(results_df['label'], results_df['predicted'], average='weighted'),
        "recall": recall_score(results_df['label'], results_df['predicted'], average='weighted'),
        "f1": f1_score(results_df['label'], results_df['predicted'], average='weighted'),
        "results_summary": results_summary,
    }
