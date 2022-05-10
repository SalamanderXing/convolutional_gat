import torch as t
import torch.nn as nn
from tqdm import tqdm
import ipdb


def test(model: nn.Module, device, loader, *, nregions=6, flag="val"):
    # binarize_thresh = t.mean(val_test_loader.normalizing_mean)
    model.eval()  # We put the model in eval mode: this disables dropout for example (which we didn't use)
    with t.no_grad():  # Disables the autograd engine
        total_length = 0
        # mean = val_test_loader.normalizing_mean
        # var = val_test_loader.normalizing_var
        # threshold = (0.5 - t.mean(val_test_loader.normalizing_mean)) / t.mean(
        #    val_test_loader.normalizing_var
        # )

        factor = 47.83
        threshold = 0.5
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0
        loss_model = 0.0
        y_sum = 0.0
        y_num = 0.0
        # tot_nmse = 0
        # tot_size = 0
        print(f"{nregions=}")
        for i, (x, y) in tqdm(enumerate(loader)):
            x = x[:, :, :, :, :nregions].to(device)
            y = y[:, :, :, :, :nregions].to(device)
            # tot_size += x.shape[0]
            y_true = y
            y_pred = model(x)
            # tot_nmse += nmser(y, y_pred)
            """
            denorm_loss_model += t.nn.functional.mse_loss(
                y_pred.squeeze() * factor, y_true * factor, reduction="sum"
            )
            """
            y_sum += t.sum((y_true.flatten() * factor * 12) ** 2)
            y_num += t.nn.functional.mse_loss(
                y_pred.squeeze() * factor * 12,
                y.squeeze() * factor * 12,
                reduction="sum",
            )
            loss_model += t.nn.functional.mse_loss(
                y_pred.squeeze() * factor, y.squeeze() * factor, reduction="sum"
            )
            y_pred_adj = y_pred.squeeze() * 47.83 * 12
            y_true_adj = y_true.squeeze() * 47.83 * 12
            # convert to masks for comparison
            y_pred_mask = y_pred_adj > threshold
            y_true_mask = y_true_adj > threshold
            # tn, fp, fn, tp = confusion_matrix(y_true_mask.cpu().view(-1), y_pred_mask.cpu().view(-1),
            #                                   labels=[0, 1]).ravel()
            tn, fp, fn, tp = t.bincount(
                y_true_mask.flatten() * 2 + y_pred_mask.flatten(), minlength=4,
            )
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
            total_length += y_true.numel()
            # computes NMSE given loss_model

            # get metrics for sample
            """
            y = t.pow(y, 1 / loader.power)
            y_hat = t.pow(y_hat, 1 / loader.power)
            running_loss += (
                t.sum((y - y_hat) ** 2) / t.prod(t.tensor(y.shape[1:]).to(device))
            ).cpu()

            unique = t.unique(y)
            threshold = 0.5 # unique[int(len(unique) * (1 / 2))].cpu()
            total_length += x.numel()
            acc, prec, rec = get_metrics(
                y.detach(), y_hat.detach(), threshold,  # second_min  # 0.04011
            )
            running_acc += acc
            running_prec += prec
            running_recall += rec
            running_denorm_mse += (
                t.sum(((y.flatten() - y_hat.flatten()) * loader.normalizing_max) ** 2)
            ).cpu()
            ipdb.set_trace()
            """
        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
        f1 = 2 * precision * recall / (precision + recall)
        mse = loss_model / total_length
        # nmse = y_num / y_sum
        nmse = mse / 0.0014030783 * 47.83
        # nmse = tot_nmse / tot_size
        # nmse = denorm_loss_model / total_length

    model.train()
    return {
        "MSE": mse.item(),
        "Accuracy": accuracy.item(),
        "Precision": precision.item(),
        "Recall": recall.item(),
        "f1": f1.item(),
        "NMSE": nmse.item(),
    }
