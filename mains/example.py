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
from utils import saver, logging
# Change this
from data.example import SinxDataset

# Parse arguments
parser = ArgumentParser(description="Run machine learning training")
parser.add_argument("config_file", help="Path to json config file")
parser.add_argument("--device", default="cpu")
parser.add_argument("--load", default=None, const=-1, nargs="?", type=int,
        help="Load checkpoint. No argument to load latest checkpoint, "
        "or a number to load checkpoint from a particular epoch")
args = parser.parse_args()

# Load configs
with open(args.config_file) as f:
    configs = json.load(f)
log_dir = logging.get_log_dir(args.config_file)
pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

# Setup device
device = torch.device(args.device)

# Make data
dataset = SinxDataset(configs)
dataloader = DataLoader(dataset, batch_size=configs["batch_size"], shuffle=False)

# Make model
# Normally would come from model file, but here we are just using a simple file
# Remember to move models to the correct device!
ff = nn.Sequential(
        nn.Linear(1, 20),
        nn.ReLU(),
        nn.Linear(20, 1)).to(device)
model_list = [ff]  # Change to add more models

# Setup optimizer(s) and loss function(s)
optimizer = torch.optim.Adam(itertools.chain(*[model.parameters() for model in model_list]), lr=configs["learning_rate"])
loss_fn = nn.MSELoss()

# Set up logger
logger = logging.Logger(log_dir)

# Load model
if args.load is None:
    start_epoch = 0
else:
    saver.load_checkpoint(model_list, optimizer, log_dir, epoch=args.load)

# Do training
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
    if epoch % configs.get('save_freq', 1000) == 0:
        saver.save_checkpoint(model_list, optimizer, log_dir, epoch)
    # PUT ANY TESTING HERE
    model.eval()

