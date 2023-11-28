import evaluate
import numpy as np

f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    results = {}
    results.update(f1_metric.compute(predictions=preds, references = labels, average="micro"))
    results.update(precision_metric.compute(predictions=preds, references = labels, average="micro"))
    results.update(recall_metric.compute(predictions=preds, references = labels, average="micro"))
    results.update(accuracy_metric.compute(predictions=preds, references = labels))
    return results