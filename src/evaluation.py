"""Module for evaluation utils."""

import numpy as np
import torch


def eval_acc_in_binary_task(model, X_test, y_test, dataset, device) -> float:
    X_test_tensor = torch.from_numpy(X_test)
    y_pred = (
        (1 * (model(X_test_tensor.float().to(device)) > 0.5)).cpu().numpy().squeeze()
    )
    acc = np.sum(y_pred == y_test) / y_pred.shape[0]
    print(f"{dataset} accuracy: {acc:.4f}")
    return y_pred, acc
