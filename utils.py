import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os


def batch_transform(batch, transform):
    transf_slices = [transform(tensor) for tensor in torch.unbind(batch)]
    return torch.stack(transf_slices)


def save_checkpoint(model, optimizer, epoch, loss, args):

    name = args.name
    save_dir = args.save_dir

    assert os.path.isdir(
        save_dir), "The directory \"{0}\" doesn't exist.".format(save_dir)

    # Save model
    model_path = os.path.join(save_dir, name)
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, model_path)

    # Save arguments
    summary_filename = os.path.join(save_dir, name + '_summary.txt')
    with open(summary_filename, 'w') as summary_file:
        sorted_args = sorted(vars(args))
        summary_file.write("ARGUMENTS\n")
        for arg in sorted_args:
            arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
            summary_file.write(arg_str)

        summary_file.write("\nBEST VALIDATION\n")
        summary_file.write("Epoch: {0}\n". format(epoch))
        summary_file.write("Mean IoU: {0}\n". format(loss))


def load_checkpoint(model, optimizer, folder_dir, filename):

    assert os.path.isdir(
        folder_dir), "The directory \"{0}\" doesn't exist.".format(folder_dir)

    # Create folder to save model and information
    model_path = os.path.join(folder_dir, filename)
    assert os.path.isfile(
        model_path), "The model file \"{0}\" doesn't exist.".format(filename)

    # Load the stored model parameters to the model instance
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss
