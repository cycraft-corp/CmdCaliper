import sys
import os
import json
import logging
from pathlib import Path
import yaml

import torch
import numpy as np

HUGGINGFACE_DIRNAME = "./huggingface_model"
MODEL_NAME = "model.pth"
OPTIMIZER_NAME = "optimizer.pth"
LR_SCHEDULER_NAME = "lr_scheduler.pth"

def load_yaml(path_to_data):
    with open(path_to_data, "r") as f:
        data = yaml.safe_load(f)
    return data

def load_json(path_to_data):
    with open(path_to_data, "r") as f:
        data = json.load(f)
    return data

def save_json(data, path_to_data):
    with open(path_to_data, "w") as f:
        json.dump(data, f, indent=2)

def postprocess_data_synthesis_response(response):
    if "```" in response:
        response_list = [r for r in response.split("\n") if "```" not in r]
        response = "\n".join(response_list)
    return response

def extract_cmds(response):
    response = response.replace("<CMD>\n", "<CMD>")

    new_generated_cmd_list = []
    for r in response.split("\n"):
        r = r.strip()
        if "<CMD>" in r:
            cmd = r[len("<CMD>"):]
            if "</CMD>" in r:
                cmd = cmd[:cmd.index("</CMD>")]
            if cmd == "":
                continue
            new_generated_cmd_list.append(cmd.strip())
    return new_generated_cmd_list

class AverageMeter(object):
    def __init__(self, name=''):
        self._name = name
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def __str__(self):
        return "%s: %.5f" % (self._name, self.avg)

    def get_avg(self):
        return self.avg

    def __repr__(self):
        return self.__str__()


def set_random_seed(seed):
    import random
    logging.info("Set seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def rank0_print(*args, level: int = logging.INFO):
    try:
        if dist.get_rank() == 0:
            logging.log(level, *args)
    except:
        logging.log(level, *args)


def initialize_logging(path_to_logging_dir: Path, level: int):
    # Clear original logging setting (e.g., ColossalAI)
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    log_format = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        #filename=os.path.join(path_to_logging_dir, "logger.log"),
        level=level,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p"
    )
    
    path_to_logging_file = path_to_logging_dir / "logger.log"
    path_to_logging_file.touch(exist_ok=True)

    file_handler = logging.FileHandler(str(path_to_logging_file))
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)

def calculate_param_nums(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def resume_checkpoint(
    path_to_checkpoint_dir,
    model,
    criterion=None,
    optimizer=None,
    lr_scheduler=None
):
    path_to_model_checkpoint = os.path.join(path_to_checkpoint_dir, MODEL_NAME)
    path_to_optimizer = os.path.join(path_to_checkpoint_dir, OPTIMIZER_NAME)
    path_to_lr_scheduler = os.path.join(path_to_checkpoint_dir, LR_SCHEDULER_NAME)

    resume_epoch = None
    if os.path.isfile(path_to_model_checkpoint):
        model_checkpoint = torch.load(path_to_model_checkpoint)
        model.load_state_dict(model_checkpoint)

    if os.path.isfile(path_to_optimizer):
        optimizer_checkpoint = torch.load(path_to_optimizer)
        optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
        resume_epoch = optimizer_checkpoint.get("resume_epoch", None)

    if os.path.isfile(path_to_lr_scheduler):
        lr_scheduler_checkpoint = torch.load(path_to_lr_scheduler)
        lr_scheduler.load_state_dict(lr_scheduler_checkpoint)

    return resume_epoch

def save_checkpoint(
    path_to_checkpoint_dir,
    model,
    optimizer=None,
    lr_scheduler=None,
    resume_epoch=None
):
    path_to_model_checkpoint = os.path.join(path_to_checkpoint_dir, MODEL_NAME)
    path_to_huggingface_model_checkpoint = os.path.join(path_to_checkpoint_dir, HUGGINGFACE_DIRNAME)
    os.makedirs(path_to_huggingface_model_checkpoint, exist_ok=True)
    path_to_optimizer = os.path.join(path_to_checkpoint_dir, OPTIMIZER_NAME)
    path_to_lr_scheduler = os.path.join(path_to_checkpoint_dir, LR_SCHEDULER_NAME)

    model_checkpoint = model.state_dict()
    model.transformer.save_pretrained(path_to_huggingface_model_checkpoint)

    torch.save(model_checkpoint, path_to_model_checkpoint)

    if optimizer is not None:
        torch.save({
            "optimizer": optimizer.state_dict(),
            "resume_epoch": resume_epoch
        }, path_to_optimizer)

    if lr_scheduler is not None:
        torch.save(lr_scheduler.state_dict(), path_to_lr_scheduler)
