"""Metric.

    Mean class recall.
"""

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import numpy as np


def mean_class_recall(y_true, y_pred):
    """Mean class recall.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    class_recall = []
    target_uniq = np.unique(y_true)
    for label in target_uniq:
        indexes = np.nonzero(label == y_true)[0]
        recall = np.sum(y_true[indexes] == y_pred[indexes]) / len(indexes)
        class_recall.append(recall)
    return np.mean(class_recall)


def eval_matrix(pred, label):
    # 计算混淆矩阵
    cm = confusion_matrix(label, pred)

    # 准确率
    accuracy = accuracy_score(label, pred)

    # 精确度、召回率、F1分数（宏平均）
    precision = precision_score(label, pred, average='macro')
    recall = recall_score(label, pred, average='macro')
    f1 = f1_score(label, pred, average='macro')

    # 特异度（宏平均）
    specificity = np.mean([cm[i, i] / (cm[i, :].sum() - cm[i, i] + cm[:, i].sum()) for i in range(7)])

    # AUROC
    # 注意：对于多类分类，roc_auc_score需要label进行one-hot编码
    label_one_hot = np.eye(7)[label]
    auroc = roc_auc_score(label_one_hot, np.eye(7)[pred], multi_class='ovr')
    return accuracy, precision, recall, f1, specificity, auroc
    # return {
    #     "Accuracy": accuracy,
    #     "Precision": precision,
    #     "Recall": recall,
    #     "Specificity": specificity,
    #     "F1 Score": f1,
    #     "AUROC": auroc
    # }
if __name__ == "__main__":
    y_pred = [0, 0, 1, 1, 2, 2, 2]
    y_true = [0, 0, 0, 0, 1, 2, 2]
    print(mean_class_recall(y_true, y_pred))
