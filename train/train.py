import argparse
from pathlib import Path
from math import floor
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LRScheduler, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader

import transformers
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, SchedulerType
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTConfig, SFTTrainer

from train_config import TrainConfig
from utils import TokenizeMapWrapper, Evaluator, Seq2SeqFormatter

def main():
    config = TrainConfig()

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    pprint(config)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_ckpt, padding_side="left", use_cache=False)
    #tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    model_config = AutoConfig.from_pretrained(config.model_ckpt)
    model_config.use_sliding_window = True
    model_config.sliding_window = config.sliding_window
    model = AutoModelForCausalLM.from_pretrained(
        config.model_ckpt, 
        attn_implementation='flash_attention_2',
        quantization_config=bnb_config, 
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    ).to(device)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    train_ds = load_dataset(config.train_ds_path)['train'].shuffle(seed=42)
    test_ds = load_dataset(config.test_ds_path)['test'].shuffle(seed=42)
    
    prompt_formatter = Seq2SeqFormatter(
        feature=config.input_text_feat_name,
        target=config.label_text_feat_name,
        tokenizer=tokenizer,
    )
    
    num_samples = len(train_ds)

    compute_metrics = Evaluator(config.eval_methods, tokenizer)

    training_args = SFTConfig(
        max_seq_length=config.max_length,
        max_steps=int(floor(num_samples // config.batch_size)),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        lr_scheduler_type=config.scheduler_type,

        do_eval=True,
        eval_strategy='steps',
        greater_is_better=False,
        eval_steps=config.eval_steps,

        logging_first_step=0,
        logging_steps=1,
        report_to='wandb',

        output_dir='./out',
        save_strategy='best',
        metric_for_best_model='eval_loss',
    )

    trainer = SFTTrainer(
        model,
        training_args,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        formatting_func=prompt_formatter,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(f'models/{config.output_model_name}')

if __name__ == '__main__':
    main()
