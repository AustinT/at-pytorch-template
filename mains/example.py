from argparse import ArgumentParser
import sys
import os
import json
import itertools
import pathlib
from tqdm import tqdm, trange
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# Custom code
sys.path.append(os.getcwd())
from utils import saver, ml_logging, config as cfg
# Change this
from data.example import SinxDataset

# Arguments
parser = ArgumentParser(description="Run machine learning training")
parser.add_argument("config_file", help="Path to json config file")
parser.add_argument("--device", default=None)
parser.add_argument("--load", default=None, const=-1, nargs="?", type=int,
        help="Load checkpoint. No argument to load latest checkpoint, "
        "or a number to load checkpoint from a particular epoch")
parser.add_argument("--no-train", action="store_true")
parser.add_argument("--eval", action="store_true")

# Functions to make models
def get_models(configs, device):
    """
    Modify this function to set up your models, ideally using "configs" in some way
    """
    ff = nn.Sequential(
        nn.Linear(1, configs["n_hidden"]),
        nn.ReLU(),
        nn.Linear(configs["n_hidden"], 1)).to(device)
    return ff

# Main program
# Putting training in a main block ensures that model-building functions can be called from elsewhere
if __name__ == "__main__":

    args = parser.parse_args()

    # Load configs
    with open(os.path.join(args.config_file)) as f:
        configs = json.load(f)
    expt_name = cfg.get_expt_name(args.config_file, configs)
    log_dir = ml_logging.get_log_dir(expt_name)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    print(cfg.expt_summary(expt_name, configs))

    # Setup device
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    # Make data
    dataset = SinxDataset(configs)
    dataloader = DataLoader(dataset, batch_size=configs["batch_size"], shuffle=False)

    # Make model
    # Normally would come from model file, but here we are just using a simple file
    # Remember to move models to the correct device!

    ff = get_models(configs, device)
    model_list = [ff]  # Change to add more models

    # Setup optimizer(s) and loss function(s)
    optimizer = torch.optim.Adam(itertools.chain(*[model.parameters() for model in model_list]), lr=configs["learning_rate"])
    loss_fn = nn.MSELoss()
    
    # Learning rate scheduler
    lr_scheduler = configs.get("lr_scheduler", None)
    if lr_scheduler is not None:
        cls_name = lr_scheduler.pop("class_name")
        lr_scheduler = getattr(torch.optim.lr_scheduler, cls_name)(optimizer, **lr_scheduler)


    # Set up logger
    logger = ml_logging.Logger(log_dir)

    # Load model
    if args.load is None:
        start_epoch = 0
    else:
        start_epoch = saver.load_checkpoint(model_list, optimizer, log_dir, epoch=args.load)

    # Do training
    if args.no_train:
        print("SKIPPING TRAINING")
    else:
        print("Starting training...")
        for epoch in trange(start_epoch, configs["num_epochs"], desc="Epoch"):
            for model in model_list:
                model.train()
            
            # Track basic loss for batch
            total_losses = []

            for batch_num, batch_dict in tqdm(enumerate(dataloader), desc="Batch", leave=False):

                # Get data
                x = torch.as_tensor(batch_dict['x'], device=device)
                y_tru = torch.as_tensor(batch_dict['y'], device=device)

                # Run inference
                optimizer.zero_grad()
                y_pred = ff(x)

                # Calculate loss and optimize
                loss_val = loss_fn(y_pred, y_tru)
                loss_val.backward()
                optimizer.step()
                
                # Track losses for log
                total_losses.append(loss_val.item())

            # Write to log files
            logger.scalar_summary('loss', np.average(total_losses), epoch)

            # Saving and testing
            if epoch % configs.get('save_freq', int(1e6)) == 0:
                saver.save_checkpoint(model_list, optimizer, log_dir, epoch)
            
            # PUT ANY TESTING HERE (the kind that happens every epoch)
            for model in model_list:
                model.eval()
        
        
        # Save a final checkpoint
        saver.save_checkpoint(model_list, optimizer, log_dir, configs["num_epochs"])
                
    if args.eval:
        print("Evaluating...")
        for model in model_list:
            model.eval()
        
        # For this example, evaluate 1 whole epoch
        eval_losses = []
        with torch.no_grad():
            for batch_num, batch_dict in tqdm(enumerate(dataloader), desc="Batch", leave=False):

                # Get data
                x = torch.as_tensor(batch_dict['x'], device=device)
                y_tru = torch.as_tensor(batch_dict['y'], device=device)

                # Run inference
                y_pred = ff(x)

                # Calculate loss and optimize
                loss_val = loss_fn(y_pred, y_tru)
                
                # Track losses for log
                eval_losses.append(loss_val.item())

        average_loss_val = np.average(eval_losses)
        print("Loss value: {:.2e}".format(average_loss_val))
                
                
    # Close the logger
    logger.close()
    print("\n\nSUCCESSFUL END OF SCRIPT")


