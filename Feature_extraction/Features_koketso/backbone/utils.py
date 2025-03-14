def load_model_checkpoint(model, checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Load the state_dict into the model
    model.load_state_dict(checkpoint['model_state_dict'])

    # If you have other components to restore, such as the epoch or loss
    epoch_start = checkpoint['epoch']
    loss = checkpoint['loss']
    augmentations = checkpoint['augmentations']

    print(f"Model loaded from epoch {epoch_start} with loss {loss}")

    return model

