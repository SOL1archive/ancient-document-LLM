from typing import List
from pprint import pprint
import re
from icecream import ic
import numpy as np
import evaluate
from torchmetrics.text import ROUGEScore, BLEUScore

class Seq2SeqFormatter:
    def __init__(
            self, 
            feature, 
            target, 
            tokenizer,
            system_instruction=None, 
        ):
        if system_instruction is not None:
            self.system_instruction = system_instruction
        else:
            self.system_instruction = None
        self.instruction_col_name = feature
        self.response_col_name = target
        self.tokenizer = tokenizer

    def __call__(self, example):
        prompt = [
            {'role': 'user', 'content': "Ancient Text: " + example[self.instruction_col_name]},
            {'role': 'assistant', 'content': " Modern Korean: " + example[self.response_col_name]}
        ]
        if self.system_instruction is not None:
            prompt.append({'role': 'system', 'content': self.system_instruction})
        return self.tokenizer.apply_chat_template(prompt, tokenize=False)

class TokenizeMapWrapper:
    def __init__(self, tokenizer, feature, option=None):
        if option is None:
            option = {
                'max_length': 8192,
                'truncation': True,
                'padding': 'longest',
            }
        
        self.feature = feature
        self.tokenizer = tokenizer

    def __call__(self, row):
        return self.tokenizer(row[self.feature], **self.option)

    def __repr__(self):
        return f'{self.__class__.__name__}(tokenizer={self.tokenizer})'

class Seq2SeqTokenizeMapWrapper(TokenizeMapWrapper):
    def __init__(self, tokenizer, feature, target, option=None):
        super().__init__(tokenizer, feature, option)
        self.target = target
        self.option = option

    def seq2seq_tokenize(self, row):
        form_embeddings = self.tokenizer(row[self.feature], **self.option)
        pprint(form_embeddings)
        correct_form_embeddings = self.tokenizer(row[self.target], **self.option)
        pprint(correct_form_embeddings)

        return {
            'input_ids': form_embeddings['input_ids'],
            'attention_mask': form_embeddings['attention_mask'],
            'labels': correct_form_embeddings['input_ids'],
        }

    def __call__(self, row):
        return self.seq2seq_tokenize(row)

class SummaryTokenizeMapWrapper(TokenizeMapWrapper):
    def __init__(self, tokenizer, feature, max_token=4096, option=None):
        if option is None:
            option = {
                'max_length': max_token,
                'truncation': True,
            }

        self.max_token = option['max_new_tokens']
        self.option = option
        self.feature = feature
        self.tokenizer = tokenizer

    def __call__(self, row):
        total_text = row[self.feature]
        if len(re.findall('\nSummary: \n', total_text)) == 1:
            text, summary = total_text.split('Summary: \n')
            summary = '\nSummary: \n' + summary
        else:
            print('warning: more than two summary exists')
            text_split = total_text.split('Summary: \n')
            text = text_split[0]
            summary = '\nSummary: \n'.join(text_split[1:])
        
        tokenized_text = self.tokenizer(text, **self.option)
        tokenized_summary = self.tokenizer(summary, **self.option)
        tokenized_total_text = dict()
        if len(tokenized_text['input_ids']) + len(tokenized_summary['input_ids']) <= self.max_token:
            for key in tokenized_text:
                tokenized_total_text[key] = tokenized_text[key] + tokenized_summary[key]
                if len(tokenized_total_text[key]) < self.max_token:
                    tokenized_total_text[key] = (tokenized_total_text[key] 
                                                 + [1] * (self.max_token - len(tokenized_total_text[key]))
                    )
        else:
            for key in tokenized_text:
                tokenized_total_text[key] = (tokenized_text[key][:- len(tokenized_summary['input_ids'])] 
                                             + tokenized_summary[key]
                )

        return tokenized_total_text

class Evaluator:
    def __init__(self, eval_methods: List[str], tokenizer):
        self.tokenizer = tokenizer
        self.eval_methods = {
            method: evaluate.load(method) for method in eval_methods
        }
        if 'rouge' in self.eval_methods:
            self.eval_methods['rouge'] = ROUGEScore(tokenizer=lambda text: self.tokenizer(text))
        #if 'bleu' in self.eval_methods:
        #    self.eval_methods['bleu'] = BLEUScore(n_gram=2)
    
    def __call__(self, preds, refs):
        results = dict()
        for name, metric in self.eval_methods.items():
            if name == 'rouge':
                rouge_scores = metric(
                    preds=[preds], 
                    target=[refs], 
                )
                results['rouge1_fmeasure'] = float(rouge_scores['rouge1_fmeasure'])
                results['rouge2_fmeasure'] = float(rouge_scores['rouge2_fmeasure'])
                results['rougeL_fmeasure'] = float(rouge_scores['rougeL_fmeasure'])
                results['rougeLsum_fmeasure'] = float(rouge_scores['rougeLsum_fmeasure'])
            elif name == 'bleu':
                bleu_scores = metric.compute(
                    predictions=[preds], 
                    references=[refs], 
                    max_order=2,
                )
                results['bleu'] = bleu_scores['bleu']
                results['reference_length'] = bleu_scores['reference_length']
                results['translation_length'] = bleu_scores['translation_length']

                #bleu_scores = metric(
                #    preds=[preds], 
                #    target=[refs], 
                #)
                #bleu_scores = float(bleu_scores)
                #results['bleu'] = bleu_scores
            else:
                results[name] = metric.compute(
                    predictions=[preds], 
                    references=[refs], 
                    tokenizer=self.tokenizer,
                )
        return results
