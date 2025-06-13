from pprint import pprint
from typing import List, Literal
from dataclasses import dataclass, field

@dataclass
class EvalConfig:
    base_model_ckpt: str = 'Qwen/Qwen2.5-1.5B'
    eval_tokenizer_ckpt: str = 'naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B'
    model_ckpt: str = '/workspace/ancient-document-LLM/train/models/silloc_translator'

    sliding_window: int = 1024
    
    test_ds_path: str = '../data/test'

    input_text_feat_name: str = 'original_text'
    label_text_feat_name: str = 'translated_text'
    num_train_samples: int = 35_715
    max_length: int = 4096
    max_new_tokens: int = 2048 + 512
    
    train_method: Literal['full', 'LoRA', 'QLoRA'] = 'QLoRA'
    # LoRA / QLoRA Configs(only used when `train_method in ['LoRA', 'QLoRA']`)
    target_modules: List[str] = field(default_factory=lambda: ['q_proj', 'v_proj'])
    rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    eval_batch_size: int = 2

    eval_methods: List = field(default_factory=lambda: [
        'bleu', 'rouge'
    ])

    eval_result_path: str = '../results/result.parquet'

if __name__ == '__main__':
    train_config = EvalConfig()
    pprint(train_config)
