import torch
import matplotlib.pyplot as plt

class Test:

    def __init__(self, model, data_loader, criterion, device):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.device = device

    def run_epoch(self, iteration_loss=False):

        self.model.eval()
        epoch_loss = 0.0
        for step, batch_data in enumerate(self.data_loader):
            # Get the inputs and labels
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            with torch.no_grad():
                # Forward propagation
                mask = (inputs>0).float()
                outputs = self.model(inputs, mask)

                
                plt.figure()
                plt.imshow(inputs[0,0].cpu().detach().numpy())
                plt.figure()
                plt.imshow(outputs[0,0].cpu().detach().numpy())
                plt.figure()
                plt.imshow((outputs*mask)[0,0].cpu().detach().numpy())
                plt.figure()
                plt.imshow(labels[0,0].cpu().detach().numpy())
                plt.show()
                

                # Loss computation
                loss = (self.criterion(outputs, labels)*mask.detach()).sum()/mask.sum()

            # Keep track of loss for current epoch
            epoch_loss += loss.item()

            if iteration_loss:
                print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader)
