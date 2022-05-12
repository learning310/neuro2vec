import torch
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score, f1_score

def accuracy_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return accuracy_score(pred.cpu().numpy(), target.data.cpu().numpy())


def f1_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
    return f1_score(pred.cpu().numpy(), target.data.cpu().numpy(), average='macro')

def _calc_metrics(pred_labels, true_labels, log_dir, home_path):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # save targets
    labels_save_path = os.path.join(log_dir, "labels")
    os.makedirs(labels_save_path, exist_ok=True)
    np.save(os.path.join(labels_save_path, "predicted_labels.npy"), pred_labels)
    np.save(os.path.join(labels_save_path, "true_labels.npy"), true_labels)


    names = ['W', "N1", "N2", "N3", "REM"]
    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True, target_names=names)

    del (r['accuracy'])
    df = pd.DataFrame(r)
    df.loc["accuracy"] = accuracy_score(true_labels, pred_labels)
    df.loc["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df = df * 100
    df.loc["support"] = df.loc["support"] / 100

    # save classification report
    exp_name = os.path.split(os.path.dirname(log_dir))[-1]
    training_mode = os.path.basename(log_dir)
    file_name = f"{exp_name}_{training_mode}_classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    # save confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    cm_file_name = f"{exp_name}_{training_mode}_confusion_matrix.torch"
    cm_Save_path = os.path.join(home_path, log_dir, cm_file_name)
    torch.save(cm, cm_Save_path)

    print("Validation: Acc:{:.2f}, F1:{:.2f}".format(df.loc["accuracy"]["W"], df["macro avg"]["f1-score"]))