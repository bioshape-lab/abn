"""Training and testing functions for VAE model."""

import copy

# import wandb
import numpy as np


def train_test(model, train_loader, test_loader, optimizer, scheduler, config):
    train_losses = []
    test_losses = []
    lowest_test_loss = np.inf
    for epoch in range(1, config.num_epochs + 1):
        train_loss = run_one_epoch(
            epoch=epoch,
            model=model,
            data_loader=train_loader,
            device=config.device,
            optimizer=optimizer,
            train=True,
        )
        train_losses.append(train_loss)

        test_loss = run_one_epoch(
            epoch=epoch,
            model=model,
            data_loader=test_loader,
            device=config.device,
            optimizer=optimizer,
            train=False,
        )

        if scheduler is not None:
            scheduler.step(test_loss)

        test_losses.append(test_loss)

        if epoch == 1 or test_loss < lowest_test_loss:
            lowest_test_loss = test_loss
            best_model = copy.deepcopy(model)

    return train_losses, test_losses, best_model


def run_one_epoch(epoch, model, data_loader, device, optimizer, train=True):
    """Run one epoch on the train or test set.

    Parameters
    ----------
    epoch : int
        Index of current epoch.
    train : bool, optional
        Whether this is a training epoch (default is True).

    Returns
    -------
    loss : float
        Loss on epoch.
    """
    if train:
        model.train()
        mode = "Train"
    else:
        model.eval()
        mode = "Test"

    epoch_loss = 0

    for batch_idx, batch_data in enumerate(data_loader):
        x_batch, labels = batch_data
        x_batch = x_batch.to(device)
        labels = labels.float().to(device)

        if train:
            optimizer.zero_grad()

        # Forward pass
        z_batch, x_mu_batch, posterior_params = model(x_batch)

        # Compute loss
        loss = model.criterion(x_batch, x_mu_batch, posterior_params, labels, z_batch)
        epoch_loss += loss.item()

        # Backpropagation and optimizer step only during training
        if train:
            loss.backward()
            optimizer.step()

        # Logging during training
        if train and batch_idx % 20 == 0:
            print(
                f"{mode} Epoch: {epoch} [{batch_idx * len(x_batch)}/{len(data_loader.dataset)} "
                f"({100.0 * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item() / len(x_batch):.6f}"
            )

    epoch_loss /= len(data_loader.dataset)

    # # Logging with wandb
    # wandb.log(
    #     {
    #         f"{mode.lower()}_loss": epoch_loss,
    #     },
    #     step=epoch,
    # )

    print(f"====> {mode} Epoch: {epoch} Average loss: {epoch_loss:.4f}")
    return epoch_loss
