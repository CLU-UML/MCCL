import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import utils
import variables
import torch.nn as nn

def eval_model(model, dataloader, device):
    model.eval()
    predictions = []
    labels = []
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            labels.append(batch.y)
            prediction = model(batch)
            prediction = sigmoid(prediction)
            predictions.append(prediction)
        predictions = torch.cat(predictions, dim=0)
        labels = torch.cat(labels, dim=0)
        predictions_proba = predictions.detach().clone()
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0

        p = precision_score(labels.cpu().numpy(), predictions.cpu().numpy())
        r = recall_score(labels.cpu().numpy(), predictions.cpu().numpy())
        f1 = f1_score(labels.cpu().numpy(), predictions.cpu().numpy())

        return p, r, f1, predictions, predictions_proba


def eval_best_model(args,model, test_loader):
    model_name = utils.get_model_name(args)
    checkpoint = torch.load(variables.dir_model + '/{}_best.pth'.format(model_name))
    model.load_state_dict(checkpoint["state_dict"])
    best_epoch = checkpoint['epoch']
    print('Loading the best model at epoch = {}'.format(best_epoch))

    t_p, t_r, t_f1, predictions, predictions_proba = eval_model(model, test_loader, variables.device)
    print("precision: {:.4f}, recall: {:.4f}, f1: {:.4f} ".format(t_p, t_r, t_f1))
    return t_p, t_r, t_f1, predictions, predictions_proba

