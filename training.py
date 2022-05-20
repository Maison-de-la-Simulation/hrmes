import numpy as np
import torch
from tqdm.notebook import tqdm
import random
import matplotlib.pyplot as plt
from conf import device, test_length


def train(
    model, epochs, train_dataloader, test_dataloader, optimizer, criterion, bool_mask
):
    epoch_train_losses = []
    epoch_test_losses = []

    for epoch in range(epochs):
        print("Epoch ", epoch + 1)
        accumulated_train_loss = 0.0
        accumulated_test_loss = 0.0
        train_losses = []
        test_losses = []

        pbar = tqdm(train_dataloader)
        for i, (x, y) in enumerate(pbar, start=1):
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
            output = model(x)

            optimizer.zero_grad()
            loss = model.compute_loss(criterion, output, y, bool_mask)
            loss.backward()
            optimizer.step()

            loss = loss.item()
            accumulated_train_loss += loss
            train_losses.append(accumulated_train_loss / i)
            pbar.set_description(f"Loss: {accumulated_train_loss / i:.3f}")
        
        pbar = tqdm(test_dataloader)
        for i, (x, y) in enumerate(pbar, start=1):
            with torch.no_grad():
                x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
                output = model(x)
                loss = model.compute_loss(criterion, output, y, bool_mask).item()
                accumulated_test_loss += loss
                pbar.set_description(f"Test loss: {accumulated_test_loss / i:.3f}")
                test_losses.append(accumulated_test_loss / i)
        
        epoch_train_losses.append(accumulated_train_loss / len(train_dataloader))
        epoch_test_losses.append(accumulated_test_loss / len(test_dataloader))

    if epochs == 1:
        return train_losses, test_losses
    else:
        return epoch_train_losses, epoch_test_losses


def test_sample(model, criterion, test_ds, bool_mask, idx=None):
    if idx is None:
        idx = random.randrange(len(test_ds))
    print("Chosen idx:", idx)
    x, y = test_ds[idx]
    x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
    
    last_month = x[-1]
    average = x.mean(0)
    output = model(x[None])[0]

    loss_last_month = model.compute_loss(criterion, last_month, y, bool_mask).item()
    loss_average = model.compute_loss(criterion, average, y, bool_mask).item()
    loss_prediction = model.compute_loss(criterion, output, y, bool_mask).item()
    
    output = output.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    output[np.isnan(y)] = np.nan
    expectedmean, expectedstd = np.nanmean(y), np.nanstd(y)
    obtainedmean, obtainedstd = np.nanmean(output), np.nanstd(output)
    print(f"Mean of ground truth: {expectedmean:.3f} ; Std of ground truth: {expectedstd:.3f}")
    print(f"Mean of predicted map: {obtainedmean:.3f} ; Std of predicted map: {obtainedstd:.3f}")
    
    # Display
    fig = plt.figure(figsize=(20, 12))
    plt.subplot(2, 2, 1)
    plt.pcolor(last_month.cpu().numpy())
    plt.title(f"Last month (loss: {loss_last_month:.3f})")
    plt.subplot(2, 2, 2)
    plt.pcolor(average.cpu().numpy())
    plt.title(f"Timewise average of input (loss: {loss_average:.3f})")
    plt.subplot(2, 2, 3)
    plt.pcolor(output)
    plt.title(f"Predicted first test month (loss: {loss_prediction:.3f})")
    plt.subplot(2, 2, 4)
    plt.pcolor(y)
    plt.title("Ground truth first test month")


def inverse_pca(pca, bool_mask, map):
    map_ = np.zeros((332, 362), dtype=float)
    map_[bool_mask] = pca.inverse_transform(map)
    map_[~bool_mask] = np.nan
    return map_


def test_sample_pca(model, criterion, test_ds, bool_mask, pca, idx=None):
    if idx is None:
        idx = random.randrange(len(test_ds))
    print("Chosen idx:", idx)
    x, y = test_ds[idx]
    x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
    
    last_month = x[-1]
    average = x.mean(0)
    output = model(x[None])[0]

    loss_last_month = model.compute_loss(criterion, last_month, y, bool_mask).item()
    loss_average = model.compute_loss(criterion, average, y, bool_mask).item()
    loss_prediction = model.compute_loss(criterion, output, y, bool_mask).item()
    
    output = output.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    bool_mask = bool_mask.cpu().numpy()
    output = inverse_pca(pca, bool_mask, output)
    y = inverse_pca(pca, bool_mask, y)
    last_month = inverse_pca(pca, bool_mask, last_month.cpu().numpy())
    average = inverse_pca(pca, bool_mask, average.cpu().numpy())

    expectedmean, expectedstd = np.nanmean(y), np.nanstd(y)
    obtainedmean, obtainedstd = np.nanmean(output), np.nanstd(output)
    print(f"Mean of ground truth: {expectedmean:.3f} ; Std of ground truth: {expectedstd:.3f}")
    print(f"Mean of predicted map: {obtainedmean:.3f} ; Std of predicted map: {obtainedstd:.3f}")
    
    lim = max(expectedstd, obtainedstd) * 3.

    # Display
    fig = plt.figure(figsize=(20, 12))
    plt.subplot(2, 2, 1)
    plt.pcolor(last_month, vmin=-lim, vmax=lim) #.cpu().numpy()
    plt.title(f"Last year (loss: {loss_last_month:.3f})")
    plt.subplot(2, 2, 2)
    plt.pcolor(average, vmin=-lim, vmax=lim) #.cpu().numpy()
    plt.title(f"Timewise average of input (loss: {loss_average:.3f})")
    plt.subplot(2, 2, 3)
    plt.pcolor(output, vmin=-lim, vmax=lim)
    plt.title(f"Predicted first test year (loss: {loss_prediction:.3f})")
    plt.subplot(2, 2, 4)
    plt.pcolor(np.abs(y - output), )#vmin=-lim, vmax=lim)
    plt.colorbar()
    plt.title("Difference between ground truth and prediction")
    # plt.title("Ground truth first test month")



def quantify_quality_one_signal(
    model, criterion, test_ds, bool_mask, first_index, last_index
):
    losses = []
    x, y = test_ds[first_index]
    x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.float32)
    with torch.no_grad():
        for i in range(first_index + 1, last_index):
            output = model(x[None])
            loss = model.compute_loss(criterion, output, y, bool_mask)
            losses.append(loss.item())
            x = torch.cat([x[1:, ...], output], dim=0)
            y = test_ds[i][1].to(device)
    return losses

def quantify_quality_average(model, criterion, test_ds, bool_mask):
    indices = [
        (test_length * i, test_length * (i+1))
        for i in range(len(test_ds) // test_length)
    ]
    losses = None
    for first_idx, last_idx in tqdm(indices):
        loss = quantify_quality_one_signal(
            model, criterion, test_ds, bool_mask, first_idx, last_idx
        )
        if losses is None:
            losses = np.array(loss)
        else:
            losses += np.array(loss)
    return losses / len(indices)