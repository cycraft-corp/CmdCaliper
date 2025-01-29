import torch
import transformers

from src.arguments import CriterionArguments, DataArguments, TrainingArguments, ModelArguments
from src.criterion import InfoNCE
from src.data_processor import ContrastDataset
from src.model import CSEBert
from src.utils import initialize_logging, set_random_seed
from src.trainer import Trainer

def main(
    criterion_args: CriterionArguments,
    training_args: TrainingArguments,
    data_args: DataArguments,
    model_args: ModelArguments
):

    device = training_args.device

    set_random_seed(training_args.random_seed)
    training_args.path_to_checkpoint_dir.mkdir(exist_ok=True)

    initialize_logging(
        path_to_logging_dir=training_args.path_to_checkpoint_dir, 
        level=training_args.log_level
    )

    criterion = InfoNCE(criterion_args, device=device)
    model = CSEBert(
        model_args.path_to_model_weight,
        model_args.gradient_checkpointing
    )
    model.to(device)

    tokenizer = model.get_tokenizer(
        padding_side="right", model_max_length=model_args.model_max_length
    )
    train_dataset, eval_dataset = ContrastDataset.initialize_dataset(tokenizer, data_args, device=device)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=train_dataset.collate_fn,
        batch_size=training_args.per_device_train_batch_size
    )
    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, collate_fn=eval_dataset.collate_fn,
            batch_size=training_args.per_device_eval_batch_size
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader)*training_args.epochs)

    trainer = Trainer(
        model=model, optimizer=optimizer,
        criterion=criterion, training_args=training_args,
        train_dataloader=train_dataloader, eval_dataloader=eval_dataloader,
        lr_scheduler=lr_scheduler
    )
    trainer.train()


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((
        CriterionArguments, TrainingArguments, DataArguments, ModelArguments
    ))

    (criterion_args, training_args, data_args, model_args, _) \
        = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    main(criterion_args, training_args, data_args, model_args)
