import os
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from termcolor import cprint

from src.datasets import ThingsMEGDataset
from src.models import resnet18_1d  # ここでインポート
from src.utils import set_seed

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = resnet18_1d(test_set.num_classes, test_set.num_channels, p_drop=0.3).to(args.device)  # ここでドロップアウト率を指定

    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    preds = []
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Evaluation"):
        with torch.no_grad():
            preds.append(model(X.to(args.device)).detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission.npy"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")

if __name__ == "__main__":
    run()
