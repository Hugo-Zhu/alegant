import os
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score, classification_report


def cross_entropy(logits, targets):
    """
    Inputs: logits未经过softmax的模型输出
    logits: [batch_size, num_classes]
    targets: [batch_size]

    F.cross_entropy会自动计算softmax，等价于log_softmax加nll_loss
        # 手动进行softmax
        probs = F.softmax(logits, dim=1)
        # 计算交叉熵损失, nll_loss负对数似然函数的均值
        loss_fn = F.nll_loss(torch.log(probs), targets)

    """
    return F.cross_entropy(logits, targets)

def cross_entropy_all(logits_list, targets_list):
    """
    Inputs: logits_list: [logits1, logits2, logits3, logits4]
    其中每个logits包含单个标签的logits

    Return: 所有标签，所有样本的loss的平均值

    """
    num_samples = 0
    loss_list = []
    for i, logits in enumerate(logits_list):
        num_samples += logits.size(0)
        loss = cross_entropy(logits_list[i], targets_list[i])
        loss_list.append(loss)
    return sum(loss_list)/num_samples


def compute_metrics(preds, labels, average):
    """
    sklearn.metrics利用计算metrics: accuracy_score, precision_score, recall_score, f1_score
    参数中average=None的话会返回所有类/标签的指标
    """
    # preds = [1-pred for pred in preds]
    # labels = [1-label for label in labels]
    confusion = confusion_matrix(y_true=labels, y_pred=preds)
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    precision = precision_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    report = classification_report(y_true=labels, y_pred=preds)
    return {
        "confusion_matrix": confusion,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report,
    }

def compute_metrics_all_class(preds_all_class, labels_all_class, average):
    """
    Input:
    preds_all_class: [preds1, preds2, preds3, preds4]
    其中每个preds包含单个标签的预测值列表
    
    Return: results_all_class: [results1, results2, results3, result4]
    每个results包含完整的metrics字典
    """
    acc_all_class = []
    f1_all_class = []
    metrics_all_class = []
    for i, preds in enumerate(preds_all_class):
        metrics = compute_metrics(preds_all_class[i], labels_all_class[i], average=average)
        acc_all_class.append(metrics["accuracy"])
        f1_all_class.append(metrics["f1"])
        metrics_all_class.append(metrics)
    avg_acc = sum(acc_all_class)/len(acc_all_class)
    avg_f1 = sum(f1_all_class)/len(f1_all_class)
    return {"metrics_all_class": metrics_all_class, "avg_acc":avg_acc, "avg_f1": avg_f1}

def cosine_similarity_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def load_data(file_path):
    # data.keys(): ['annotations', 'posts_text', 'posts_num']
    data = pickle.load(open(file_path, 'rb'))
    text = data['posts_text']
    label = data['annotations']
    # print(statistical_analysis(train_label))
    processed_data = process_data(text, label)
    return processed_data

def process_data(poster, label):
    label_lookup = {'E': 1, 'I': 0, 'S': 1, 'N':0, 'T': 1, 'F': 0, 'J': 1, 'P':0}
    poster_data = [{'posts': t, 
                'label0': label_lookup[list(label[i])[0]],
                'label1': label_lookup[list(label[i])[1]],
                'label2': label_lookup[list(label[i])[2]],
                'label3': label_lookup[list(label[i])[3]]} 
               for i,t in enumerate(poster)]
    return poster_data
    
def statistical_analysis(label):
    persona_lookup = {}
    I,E,S,N,T,F,P,J=0,0,0,0,0,0,0,0
    for t in label:
        if 'I' in t:
            I+=1
        if 'E' in t:
            E += 1
        if 'S' in t:
            S+=1
        if 'N' in t:
            N+=1
        if 'T' in t:
            T+=1
        if 'F' in t:
            F+=1
        if 'P' in t:
            P+=1
        if 'J' in t:
            J+=1
        if t not in persona_lookup:
            persona_lookup[t] = 1
        else:
            persona_lookup[t] += 1
    print('I', I)
    print('E', E)
    print('S', S)
    print('N', N)
    print('T', T)
    print('F', F)
    print('P', P)
    print('J', J)
    print("persona number:", persona_lookup)


def check_nan(vector, name=None):
    if isinstance(vector, torch.Tensor):
        if not np.any(np.isnan(vector.cpu().detach().numpy())):
            return True
        print("[] is false".format(name))
        return False
    elif isinstance(vector, np.ndarray):
        if not np.any(np.isnan(vector)):
            return True
        print("[] is false".format(name))
        return False
    elif isinstance(vector, list):
        vector = np.asarray(vector, dtype=np.float32)
        if not np.any(np.isnan(vector)):
            return True
        print("[] is false".format(name))
        return False

def checkpath(file_path):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
