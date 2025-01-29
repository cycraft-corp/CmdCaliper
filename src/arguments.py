import logging
import pathlib
from typing import List, Union, Optional, Literal
from dataclasses import dataclass, field, fields

@dataclass
class CriterionArguments:
    temperature: float = field(
        default=0.05,
        metadata={
            'help': 'The temperature of InfoNCE loss.'
        }
    )

@dataclass
class DataArguments:
    path_to_train_data_dir: str = field(
        metadata={
            'aliases': '--path-to-train-data-dir',
            'required': True,
            'help': 'Path to data folder, which should contain "train" as child folder.'
        }
    )

    path_to_eval_data_dir: Optional[str] = field(
        default=None,
        metadata={
            'aliases': '--path-to-eval-data-dir',
            'help': 'Path to data folder, which should contain "eval" as child folder.'
        }
    )

    train_percentage: float = field(
        default=1.,
        metadata={
            'aliases': '--train-percentage',
            'help': 'Percentage of spliting data into train_dataset and eval_dataset'
        }
    )

    tokenize_on_the_fly: bool = field(
        default=False,
        metadata={
            'aliases': '--tokenize-on-the-fly',
            'help': 'Whether to tokenize the sentences in each iteration.'
        }
    )

    def __post_init__(self):
        assert 0 < self.train_percentage <= 1, 'training_percentage should be within the range (0, 1]'

@dataclass
class ModelArguments:
    model_max_length: int = field(
        default=512,
        metadata={
            'aliases': ['--max-sequence-len', '--max_sequence_len', '--model-max-length'],
            'help': 'Maximum sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )
    path_to_model_weight: str = field(
        default=None,
        metadata={'aliases': '--path-to-model-weight'}
    )
    load_from_pretrained: bool = field(default=True, metadata={'aliases': '--load-from-pretrained'})
    gradient_checkpointing: bool = field(default=True, metadata={'aliases': '--gradient-checkpointing'})

@dataclass
class TrainingArguments:
    path_to_checkpoint_dir: pathlib.Path = field(
        metadata={
            'aliases': '--path-to-checkpoint-dir',
            'required': True
        }
    )
    device: str = field(default="cuda")

    lr: float = field(default=0.00002)
    epochs: int = field(default=2)

    shuffle: bool = field(default=True)
    per_device_train_batch_size: int = field(
        default=64,
        metadata={
            'aliases': ['--batch-size', '--batch_size', '--per-device-train-batch-size'],
            'help': 'The batch size per GPU/TPU core/CPU for training.'
        }
    )
    per_device_eval_batch_size: int = field(
        default=32,
        metadata={
            'aliases': '--per-device-eval-batch-size',
            'help': 'The batch size per GPU/TPU core/CPU for evaluation.'
        }
    )
    log_level: str = field(
        default='INFO',
        metadata={
            'aliases': '--log-level',
            'help': f'Set logging level. Choices=[{"|".join(logging._nameToLevel.keys())}]'
        }
    )
    log_interval: int = field(
        default=10,
        metadata={'aliases': '--log-interval'},
    )
    eval_interval: int = field(
        default=50,
        metadata={
            'aliases': '--eval-interval',
            'help': 'Do evaluation every eval_interval steps if eval_strategy is steps.'
        },
    )

    random_seed: int = field(
        default=42,
        metadata={'aliases': '--random-seed'}
    )

    def __post_init__(self):
        self.log_level = logging._nameToLevel[self.log_level.upper()]
