class Train:

    def __init__(self, model, data_loader, optim, criterion, device):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.device = device

    def run_epoch(self, lr_updater, iteration_loss=False):
        """Runs an epoch of training.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float).

        """
        self.model.train()
        epoch_loss = 0.0
        for step, batch_data in enumerate(self.data_loader):

            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            # Forward propagation
            mask = (inputs>0).float()
            outputs = self.model(inputs, mask)

            # Loss computation
            loss = (self.criterion(outputs, labels)*mask.detach()).sum()/mask.sum()

            # Backpropagation
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            lr_updater.step()

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader)
