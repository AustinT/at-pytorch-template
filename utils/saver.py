import os
import torch
save_format_str = "checkpoint{:08d}.pth"
def save_checkpoint(model_list, optimizer, save_dir, epoch):
    
    checkpoint = {
            'model_states': [model.state_dict() for model in model_list],
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch
            }
    
    torch.save(checkpoint, os.path.join(save_dir, save_format_str.format(epoch)))

def load_checkpoint(model_list, optimizer, save_dir, epoch=-1):
    
    # Search for last checkpoint if no epoch given
    if epoch < 0:
        files = os.listdir(save_dir)
        last_file = sorted(files)[-1]
        full_path = os.path.join(save_dir, last_file)
    else:
        full_path = os.path.join(save_dir, save_format_str.format(epoch))

    checkpoint = torch.load(full_path)
    model_states = checkpoint['model_states']
    assert len(model_states) == len(model_list), (len(model_states), len(model_list))
    for model, state in zip(model_list, model_states):
        model.set_state_dict(state)

    optimizer.set_state_dict(checkpoint['optimizer_state'])

