from trl import SFTConfig, SFTTrainer
import utilities
import process_dataset
from logging import Logger
import os
import math
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from typing import Union, Any

from transformers import AutoModelForCausalLM, T5ForConditionalGeneration, BartForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import TrainerCallback, Trainer, TrainingArguments, ProgressCallback, TrainerControl, TrainerState, EarlyStoppingCallback
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.data.data_collator import DataCollatorMixin
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from dataclasses import dataclass

from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import get_peft_model, PromptTuningConfig, PromptTuningInit, PrefixTuningConfig, TaskType, LoraConfig

@dataclass
class CustomDataCollator(DataCollatorForLanguageModeling):
    # def __call__(self, features, return_tensors=None):
    #     batch = super().__call__(features, return_tensors)
    #     # Now, add your custom field to the batch
    #     # Assuming 'weight' is a simple numerical value in each feature
    #     if "weight" in features[0]: # Check if the first feature has the 'weight' field
    #         batch_weights = [f["weight"] for f in features]
    #         batch["weight"] = torch.tensor(batch_weights, dtype=torch.float32).to(batch["input_ids"].device)
            
    #     return batch
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        if "weight" in examples[0]: # Check if the first feature has the 'weight' field
            batch_weights = [f["weight"] for f in examples]
            batch["weight"] = torch.tensor(batch_weights, dtype=torch.float32).to(batch["input_ids"].device)
        return batch



class MySeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #     return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Computes the loss for a Seq2SeqTrainer with instance-specific weights.

        Args:
            model (`torch.nn.Module`): The model to compute loss for.
            inputs (`dict`): The input batch, expected to contain 'input_ids',
                            'labels', 'attention_mask', and 'weight'.
            return_outputs (`bool`): Whether to return model outputs along with the loss.

        Returns:
            `torch.Tensor`: The computed loss.
            `tuple`: (loss, outputs) if `return_outputs` is True.
        """
        # Get the instance weights from the inputs
        if "weight" not in inputs:
            raise ValueError(
                "The 'weight' column is missing from the dataset inputs. "
                "Please ensure your dataset has a 'weight' column and it's passed correctly."
            )
        
        # Pop the weights from inputs so they are not passed to the model directly
        weights = inputs.pop("weight").float() # Ensure weights are float type
        
        # Forward pass through the model
        outputs = model(**inputs)

        # Get logits and labels
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        if labels is None:
            if outputs.loss is not None:
                print("Warning: Labels not found in inputs, but model returned a loss. Returning model's default loss.")
                return (outputs.loss, outputs) if return_outputs else outputs.loss
            else:
                raise ValueError(
                    "Labels are missing from the inputs and the model did not return a loss. "
                    "Labels are required for computing the loss in Seq2SeqTrainer."
                )

        # Determine the ignore_index for CrossEntropyLoss.
        # Transformers' Seq2Seq models typically use -100 as the ignore_index for padding tokens in labels.
        ignore_index = -100 # Default ignore_index for CrossEntropyLoss in Hugging Face models

        # Initialize CrossEntropyLoss with reduction='none' to get per-token loss
        loss_fct = CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

        # Flatten logits to (batch_size * sequence_length, vocab_size)
        logits_flat = logits.view(-1, logits.size(-1))
        
        # Flatten labels to (batch_size * sequence_length)
        labels_flat = labels.view(-1)

        # Calculate per-token loss
        # The loss tensor will have shape (batch_size * sequence_length)
        per_token_loss = loss_fct(logits_flat, labels_flat)

        # Reshape the per-token loss back to (batch_size, sequence_length)
        batch_size = labels.size(0)
        sequence_length = labels.size(1)
        per_sequence_loss_unmasked = per_token_loss.view(batch_size, sequence_length)

        # Create a mask for valid (non-padding) tokens in the labels
        # This is important because CrossEntropyLoss with ignore_index=-100 already handles it,
        # but we need a mask to correctly sum and average for weighting purposes.
        # The mask will be 1 for valid tokens and 0 for ignored tokens.
        valid_token_mask = (labels != ignore_index).float()

        # Apply the valid token mask to zero out loss from padding tokens
        # Resulting shape: (batch_size, sequence_length)
        masked_per_sequence_loss = per_sequence_loss_unmasked * valid_token_mask

        # Sum the loss for each sequence (ignoring padding due to previous masking)
        # Resulting shape: (batch_size,)
        sequence_loss_sum = masked_per_sequence_loss.sum(dim=1)

        # Count the number of non-padding tokens per sequence
        # This will be used for averaging the loss per sequence before weighting
        num_non_padding_tokens = valid_token_mask.sum(dim=1).float()
        
        # Avoid division by zero for empty sequences (though unlikely in practice)
        num_non_padding_tokens[num_non_padding_tokens == 0] = 1e-9 
        
        # Calculate the average loss per sequence
        # Resulting shape: (batch_size,)
        average_sequence_loss = sequence_loss_sum / num_non_padding_tokens

        # Apply the instance weights.
        # Ensure weights tensor has the same shape as average_sequence_loss (batch_size,)
        # If weights are (batch_size, 1), squeeze them: weights.squeeze(-1)
        if weights.ndim > 1:
            weights = weights.squeeze(-1) # Handle case where weight might be (batch_size, 1)

        weighted_average_sequence_loss = average_sequence_loss * weights

        # Calculate the final batch loss by taking the mean of the weighted sequence losses
        final_loss = weighted_average_sequence_loss.mean()

        return (final_loss, outputs) if return_outputs else final_loss
    
class MySFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #     return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Computes the loss with instance-specific weights.

        Args:
            model (`torch.nn.Module`): The model to compute loss for.
            inputs (`dict`): The input batch, expected to contain 'input_ids',
                            'labels', 'attention_mask', and 'weight'.
            return_outputs (`bool`): Whether to return model outputs along with the loss.

        Returns:
            `torch.Tensor`: The computed loss.
            `tuple`: (loss, outputs) if `return_outputs` is True.
        """
        if "weight" not in inputs:
            raise ValueError(
                "The 'weight' column is missing from the dataset inputs. "
                "Please ensure your dataset has a 'weight' column and it's passed correctly."
            )
        
        # Pop the weights from inputs so they are not passed to the model directly
        weights = inputs.pop("weight").float() # Ensure weights are float type
        # Forward pass through the model
        outputs = model(**inputs)
        # Get logits and labels
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        # If labels are not provided, the model might not return a loss,
        # or it's a generation task without direct loss calculation.
        if labels is None:
            # If labels are not present, fall back to default behavior or raise error
            # For SFTTrainer, labels should always be present for loss calculation
            print(f"Warning: Labels not found in inputs. Returning model's default loss (if any).")
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        # Shift so that tokens < n predict n
        # This is standard for causal language modeling loss calculation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Get the attention mask for shifting
        # This is crucial to ignore padding tokens in loss calculation
        shift_attention_mask = inputs.get("attention_mask")[..., 1:].contiguous()

        # Flatten the tensors for CrossEntropyLoss
        loss_fct = CrossEntropyLoss(reduction='none') # We need per-token loss
        
        # Calculate per-token loss
        # The loss tensor will have shape (batch_size * (sequence_length - 1))
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Apply the attention mask to zero out loss from padding tokens
        # Reshape attention mask to match the flattened loss tensor
        loss = loss * shift_attention_mask.view(-1)

        # Reshape the loss back to (batch_size, sequence_length - 1)
        # This allows us to sum loss per sequence
        batch_size = labels.size(0)
        sequence_length_minus_1 = labels.size(1) - 1
        loss = loss.view(batch_size, sequence_length_minus_1)

        # Sum the loss for each sequence (ignoring padding due to previous masking)
        # Resulting shape: (batch_size,)
        sequence_loss_sum = loss.sum(dim=1)

        # Count the number of non-padding tokens per sequence
        # This will be used for averaging the loss per sequence before weighting
        num_non_padding_tokens = shift_attention_mask.sum(dim=1).float()
        
        # Avoid division by zero for empty sequences (though unlikely with SFT)
        num_non_padding_tokens[num_non_padding_tokens == 0] = 1e-9 
        
        # Calculate the average loss per sequence
        # Resulting shape: (batch_size,)
        average_sequence_loss = sequence_loss_sum / num_non_padding_tokens

        # Apply the instance weights.
        # Ensure weights tensor has the same shape as average_sequence_loss (batch_size,)
        # If weights are (batch_size, 1), squeeze them: weights.squeeze(-1)
        if weights.ndim > 1:
            weights = weights.squeeze(-1) # Handle case where weight might be (batch_size, 1)

        weighted_average_sequence_loss = average_sequence_loss * weights

        # Calculate the final batch loss by taking the mean of the weighted sequence losses
        final_loss = weighted_average_sequence_loss.mean()

        return (final_loss, outputs) if return_outputs else final_loss

    
class ModelTraining:
    """training encoder decoder model"""
    def __init__(self, the_journal:utilities.Journaling) -> None:
        """"""
        self.the_journal = the_journal
        self.output_dir = os.path.join(self.the_journal.main_config['location']['results'],
                                       self.the_journal.selected_dir)
        self.prefix_tag = f"{self.the_journal.selected_dir} >>> "
        if os.path.exists(os.path.join(self.output_dir,'tag.completed')):
            self.the_journal.main_log.info(f"{self.prefix_tag} As folder exist, we choose an ABORT.")
            raise Exception("Intentionally aborted.")
        self.the_journal.main_log.info(f"{self.prefix_tag} START...")
        """get dataset"""
        the_dataset_class = getattr(process_dataset, self.the_journal.main_config['dataset2class'][self.the_journal.dataset_name])
        self.the_dataset = the_dataset_class(the_config=self.the_journal.main_config).build()
        self.train_dataset = utilities.chunking_dataset(dataset=self.the_dataset['train'], 
                                                        chunk_size=self.the_journal.train_size)
        self.eval_dataset = utilities.chunking_dataset(dataset=self.the_dataset['validation'], 
                                                        chunk_size=self.the_journal.train_size)
        self.test_dataset = utilities.chunking_dataset(self.the_dataset['test'],
                                                       chunk_size=self.the_journal.test_size)
        """callback"""
        self.basic_callbacks = [EarlyStoppingCallback(early_stopping_patience=self.the_journal.main_config['model']['patience'])]
        
        
    def get_total_steps(self) -> int:
        """"""
        total_steps = len(self.train_dataset)/(self.the_journal.main_config['model']['gradient_accumulation_steps']*self.the_journal.batch_size)
        total_steps = int(math.ceil(total_steps))
        self.the_journal.main_log.info(f"{self.prefix_tag} Total_steps is {total_steps}, dataset size {len(self.train_dataset)}.")
        return total_steps
    
    def text_mapper_for_encoder_decoder(self, example):
            """mapped"""
            model_inputs = self.tokenizer(example['prompt'], max_length=int(self.the_journal.main_config['preprocessing']['max_length']/2), 
                                          truncation=True, padding='max_length')
            labels = self.tokenizer(example["completion"], max_length=int(self.the_journal.main_config['preprocessing']['max_length']/2),
                                    truncation=True, padding='max_length')
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
    
    def set_peft_model(self):
        if any([i==self.the_journal.finetuning for i in ['lora', 'prefix', 'prompt', 'full']]):
            self.model = get_peft_model(self.model, peft_config=self.peft_config)
            trainable_params, all_param = self.model.get_nb_trainable_parameters()
            self.the_journal.main_log.info(f"{self.prefix_tag} trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}")
        elif any([i==self.the_journal.finetuning for i in ['approx']]):
            self.make_last_layer_trainable()
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            all_param =  sum(p.numel() for p in self.model.parameters())
            # trainable_params = sum(p.numel() for p in self.model.parameters())
            self.the_journal.main_log.info(f"{self.prefix_tag} [full-FT] trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}")
    
    def make_last_layer_trainable(self):
        """approx"""
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        if self.lm_type == 'decoder_only':
            # Unfreeze the final transformer block
            tr_layer = int(len(self.model.model.layers)/10) + 1
            last_blocks = self.model.model.layers[-tr_layer]
            for param in last_blocks.parameters():
                param.requires_grad = True
        elif self.lm_type == 'encoder_decoder':
            # Unfreeze last encoder block
            # tr_layer = int(len(self.model.encoder.block)/10) + 1
            for param in self.model.encoder.block[-1].parameters():
                param.requires_grad = True
            # Unfreeze last decoder block
            tr_layer = int(len(self.model.decoder.block)/10) + 1
            for param in self.model.decoder.block[-tr_layer].parameters():
                param.requires_grad = True
            # Optional: Unfreeze shared embedding layer
            # for param in self.model.shared.parameters():
            #     param.requires_grad = True
        elif self.lm_type == 'ssm':
            raise NotImplementedError
        else:
            raise NotImplementedError
        # Unfreeze lm_head
        for param in self.model.lm_head.parameters():
            param.requires_grad = True


    def run_training(self):
        """"""
        self.bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_use_double_quant=True,
                                            bnb_4bit_quant_type="nf4",  # "fp4" is default
                                            bnb_4bit_compute_dtype=torch.bfloat16)
        self.model_path = self.the_journal.main_config['model2path'][self.the_journal.model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True,
                                                       padding="max_length", truncation=True, 
                                                       max_length=self.the_journal.main_config['preprocessing']['max_length'],
                                                       cache_dir=self.the_journal.main_config['location']['models'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        """encoder decoder model"""
        if self.the_journal.model_name.startswith(('flant5', 't5', 'bart')): # or self.the_journal.model.count('bart'):
            self.lm_type = 'encoder_decoder'
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path,
                                                                    quantization_config=self.bnb_config,
                                                                    cache_dir=self.the_journal.main_config['location']['models'],
                                                                    device_map="auto",
                                                                    # load_in_4bit=True,
                                                                    torch_dtype=torch.bfloat16,
                                                                    )
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)
            if self.the_journal.finetuning == "lora":
                self.peft_config = LoraConfig(r=16, lora_alpha=16, target_modules=["q", "v"], lora_dropout=0.05, 
                                              bias="none", task_type=TaskType.SEQ_2_SEQ_LM)
            elif self.the_journal.finetuning == "prefix":
                self.peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, num_virtual_tokens=32)
            elif self.the_journal.finetuning == "prompt":
                self.peft_config = PromptTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, num_virtual_tokens=32,
                                                      prompt_tuning_init=PromptTuningInit.RANDOM,
                                                      tokenizer_name_or_path=self.model_path)
            elif self.the_journal.finetuning == "full":
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_path,
                                                                        cache_dir=self.the_journal.main_config['location']['models'],
                                                                        device_map="auto",
                                                                        torch_dtype=torch.bfloat16)
                self.peft_config = LoraConfig(r=32, lora_alpha=128, target_modules=["q", "v"], lora_dropout=0.05, 
                                              bias="none", task_type=TaskType.SEQ_2_SEQ_LM)
            elif self.the_journal.finetuning == "approx":
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_path,
                                                                        cache_dir=self.the_journal.main_config['location']['models'],
                                                                        device_map="auto",
                                                                        torch_dtype=torch.bfloat16)
            else:
                raise NotImplementedError
            self.set_peft_model()
            total_steps = self.get_total_steps()
            stamp_steps = int(math.ceil(total_steps*0.2))
            data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
            self.train_dataset = self.train_dataset.map(self.text_mapper_for_encoder_decoder)
            self.eval_dataset = self.eval_dataset.map(self.text_mapper_for_encoder_decoder)
            self.training_args = Seq2SeqTrainingArguments(output_dir=self.output_dir,
                                                    learning_rate=self.the_journal.main_config['model']['learning_rate'],
                                                    eval_steps=stamp_steps,
                                                    eval_strategy='steps',
                                                    logging_strategy='steps',
                                                    logging_steps=stamp_steps,
                                                    logging_first_step=True,
                                                    save_strategy='steps',
                                                    save_steps=stamp_steps,
                                                    save_total_limit=self.the_journal.main_config['model']['storing_count'],
                                                    max_steps=int(total_steps*self.the_journal.main_config['model']['num_epoch']),
                                                    per_device_train_batch_size=self.the_journal.batch_size,
                                                    per_device_eval_batch_size=self.the_journal.batch_size,
                                                    load_best_model_at_end=True,
                                                    max_grad_norm=self.the_journal.main_config['model']['max_grad_norm'],
                                                    metric_for_best_model="eval_loss",
                                                    greater_is_better=False,
                                                    bf16=True, # [bf16, fp16, tf32]
                                                    optim="paged_adamw_8bit",
                                                    resume_from_checkpoint=True,
                                                    push_to_hub=False)
            self.trainer = MySeq2SeqTrainer(model=self.model,
                                    args=self.training_args,
                                    train_dataset=self.train_dataset,
                                    eval_dataset=self.eval_dataset,
                                    tokenizer=self.tokenizer,
                                    data_collator=data_collator,
                                    callbacks=self.basic_callbacks)
        
        """for decoder only model"""
        if self.the_journal.model_name.count('bloom') or self.the_journal.model_name.count('opt') or self.the_journal.model_name.count('pythia') or self.the_journal.model_name.count('qwen'):
            self.lm_type = 'decoder_only'
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                              quantization_config=self.bnb_config,
                                                              cache_dir=self.the_journal.main_config['location']['models'],
                                                              device_map="auto",
                                                              # load_in_4bit=True,
                                                              torch_dtype=torch.bfloat16,
                                                              )
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)
            if self.the_journal.finetuning == "lora":
                self.peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                              #  target_modules= ['query_key_value'],#["q_proj", "v_proj"],
                                              r=16, lora_alpha=16, lora_dropout=0.05)
            elif self.the_journal.finetuning == "prefix":
                self.peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=32)
            elif self.the_journal.finetuning == "prompt":
                self.peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, 
                                                      num_virtual_tokens=32,
                                                      prompt_tuning_init=PromptTuningInit.RANDOM,
                                                      tokenizer_name_or_path=self.model_path)
            elif self.the_journal.finetuning == "full":
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                                cache_dir=self.the_journal.main_config['location']['models'],
                                                                device_map="auto",
                                                                torch_dtype=torch.bfloat16)
                self.peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                              #  target_modules= ['query_key_value'],#["q_proj", "v_proj"],
                                              r=32, lora_alpha=128, lora_dropout=0.05)
            elif self.the_journal.finetuning == "approx":
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                                cache_dir=self.the_journal.main_config['location']['models'],
                                                                device_map="auto",
                                                                torch_dtype=torch.bfloat16)
            else:
                raise NotImplementedError
            self.set_peft_model()
            total_steps = self.get_total_steps()
            stamp_steps = int(math.ceil(total_steps*0.2))
            self.training_args = SFTConfig(output_dir=self.output_dir,
                                            learning_rate=self.the_journal.main_config['model']['learning_rate'],
                                            completion_only_loss=True,
                                            eval_steps=stamp_steps,
                                            eval_strategy='steps',
                                            logging_strategy='steps',
                                            logging_steps=stamp_steps,
                                            logging_first_step=True,
                                            save_strategy='steps',
                                            save_steps=stamp_steps,
                                            save_total_limit=self.the_journal.main_config['model']['storing_count'],
                                            max_steps=int(total_steps*self.the_journal.main_config['model']['num_epoch']),
                                            per_device_train_batch_size=self.the_journal.batch_size,
                                            per_device_eval_batch_size=self.the_journal.batch_size,
                                            load_best_model_at_end=True,
                                            max_grad_norm=self.the_journal.main_config['model']['max_grad_norm'],
                                            metric_for_best_model="eval_loss",
                                            greater_is_better=False,
                                            dataset_text_field= 'text',
                                            bf16=True, # [bf16, fp16, tf32]
                                            optim="paged_adamw_8bit",
                                            resume_from_checkpoint=True,
                                            push_to_hub=False,
                                            # remove_unused_columns=False,
                                            )
            collator = CustomDataCollator(self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token))
            self.trainer = SFTTrainer(model=self.model,
                                args=self.training_args,
                                # data_collator=collator,            # explore later ()
                                processing_class=self.tokenizer,
                                train_dataset=self.train_dataset,
                                eval_dataset=self.eval_dataset,
                                compute_loss_func=None,        # explore later
                                compute_metrics=None,          # explore later (used at evaluation)
                                callbacks=self.basic_callbacks,  # explore later (for early stopping)
                                preprocess_logits_for_metrics=None, # exlore later
                                # ONLY ON SFT
                                peft_config=None,
                                formatting_func=None,)
        
        """state space model"""
        if self.the_journal.model_name.startswith(('mamba')):
            self.lm_type = 'ssm'
            from transformers import MambaForCausalLM
            self.model = MambaForCausalLM.from_pretrained(self.model_path,
                                                            # quantization_config=self.bnb_config,
                                                            cache_dir=self.the_journal.main_config['location']['models'],
                                                            torch_dtype=torch.bfloat16,
                                                            # load_in_8bit=True,
                                                            device_map="auto")
            # self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)
            if self.the_journal.finetuning == "lora":
                self.peft_config =  LoraConfig(r=16,
                                               lora_alpha=16,
                                               target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
                                               task_type=TaskType.CAUSAL_LM,
                                               bias="none",
                                               lora_dropout=0.05)
            elif self.the_journal.finetuning == "prefix":
                self.peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=32, num_attention_heads=6)
            elif self.the_journal.finetuning == "prompt":
                raise NotImplementedError
            elif self.the_journal.finetuning == "full":
                self.peft_config =  LoraConfig(r=32,
                                               lora_alpha=128,
                                               target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
                                               task_type=TaskType.CAUSAL_LM,
                                               bias="none",
                                               lora_dropout=0.05)
            elif self.the_journal.finetuning == "approx":
                raise NotImplementedError
            else:
                raise NotImplementedError
            self.set_peft_model()
            total_steps = self.get_total_steps()
            stamp_steps = int(math.ceil(total_steps*0.2))
            self.training_args = SFTConfig(output_dir=self.output_dir,
                                            learning_rate=self.the_journal.main_config['model']['learning_rate'],
                                            completion_only_loss=True,
                                            eval_steps=stamp_steps,
                                            eval_strategy='steps',
                                            logging_strategy='steps',
                                            logging_steps=stamp_steps,
                                            logging_first_step=True,
                                            save_strategy='steps',
                                            save_steps=stamp_steps,
                                            save_total_limit=self.the_journal.main_config['model']['storing_count'],
                                            max_steps=int(total_steps*self.the_journal.main_config['model']['num_epoch']),
                                            per_device_train_batch_size=self.the_journal.batch_size,
                                            per_device_eval_batch_size=self.the_journal.batch_size,
                                            load_best_model_at_end=True,
                                            max_grad_norm=self.the_journal.main_config['model']['max_grad_norm'],
                                            metric_for_best_model="eval_loss",
                                            greater_is_better=False,
                                            # dataset_text_field= 'text',
                                            bf16=True, # [bf16, fp16, tf32]
                                            optim="paged_adamw_8bit",
                                            resume_from_checkpoint=True,
                                            push_to_hub=False)
            self.trainer = MySFTTrainer(model=self.model,
                                args=self.training_args,
                                data_collator=None,            # explore later ()
                                processing_class=self.tokenizer,
                                train_dataset=self.train_dataset,
                                eval_dataset=self.eval_dataset,
                                compute_loss_func=None,        # explore later
                                compute_metrics=None,          # explore later (used at evaluation)
                                callbacks=self.basic_callbacks,  # explore later (for early stopping)
                                preprocess_logits_for_metrics=None, # exlore later
                                # ONLY ON SFT
                                peft_config=self.peft_config,
                                formatting_func=None,)
        
        self.trainer.train()
        self.the_journal.main_log.info(f"{self.prefix_tag} END...")
        end_pointer = open(os.path.join(self.output_dir,'tag.completed'), 'w')
        end_pointer.write(f"{self.the_journal.selected_dir} completed.")
        end_pointer.close()


    
    
   

        

        


if __name__ == "__main__":
    """"""
    pass
    # def eval_mapper(example):
    #     model_inputs = tokenizer(example['prompt'], max_length=256, truncation=True, padding='max_length')
    #     labels = tokenizer(example['completion'], max_length=256, truncation=True, padding='max_length')
    #     model_inputs['labels'] = labels['input_ids']
    #     return model_inputs
    
    # x_dataset = eval_dataset.map(eval_mapper)

    # trainer.evaluate(eval_dataset=x_dataset)

    # print(eval_dataset.column_names)
    # print(trainer.train_dataset.column_names)

    # import math
    # eval_results = trainer.evaluate(eval_dataset=eval_dataset)
    # eval_results = trainer.evaluate()
    # print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")







