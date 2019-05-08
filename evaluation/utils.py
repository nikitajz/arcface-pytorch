import torch
from PIL import ImageOps
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def img_to_feature(model, img, transformer, input_shape, use_flip=True, device=None):
    if use_flip:
        data = torch.zeros([2, *input_shape])
        data[0] = transformer(img)
        data[1] = transformer(ImageOps.mirror(img))
    else:
        data = torch.zeros([1, *input_shape])
        data[0] = transformer(img)

    if device:
        data = data.to(device)

    with torch.no_grad():
        output = model(data)
    return output.data.cpu().numpy().reshape(-1)


use_metrics = [
    accuracy_score,
    f1_score,
    recall_score,
    precision_score
]


def calc_metrics(y_true, y_pred):
    m = [f(y_true, y_pred) for f in use_metrics]
    return m
