from pprint import pprint
from typing import List, Literal
from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    base_model_ckpt: str = 'Qwen/Qwen2.5-1.5B-Instruct'
    model_ckpt: str = 'Qwen/Qwen2.5-1.5B-Instruct'

    sliding_window: int = 1024
    
    train_ds_path: str = '../data/train'
    test_ds_path: str = '../data/test'

    input_text_feat_name: str = 'original_text'
    label_text_feat_name: str = 'translated_text'
    num_train_samples: int = 35_715
    max_length: int = 4096
    
    train_method: Literal['full', 'LoRA', 'QLoRA'] = 'QLoRA'
    # LoRA / QLoRA Configs(only used when `train_method in ['LoRA', 'QLoRA']`)
    target_modules: List[str] = field(default_factory=lambda: ['q_proj', 'v_proj'])
    rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    eval_steps: int = 2500

    num_epochs: int = 1
    train_batch_size: int = 6
    eval_batch_size: int = 2
    learning_rate: float = 3e-4
    scheduler_type: str = 'linear'
    warmup_ratio: float = 0.01

    logging_steps: int = 1
    logging_dir: str = 'logs'

    eval_methods: List = field(default_factory=lambda: [
        'rouge'
    ])

    output_model_name: str = 'silloc_translator'

if __name__ == '__main__':
    train_config = TrainConfig()
    pprint(train_config)
