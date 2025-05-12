from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import warnings
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import DataCollatorForLanguageModeling

default_system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information."

def initializer(model_name_or_path, model_kwargs, padding_side="right"):
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.generation_config.do_sample = True # to solve a bug in checkpoints saving...

    # the tokenizer modification is model-specific
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, add_eos_token=False, add_bos_token=False)
    # by default, tokenizer should not add eos token, as it is already added in our string formatting
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
    tokenizer.padding_side = padding_side

    return model, tokenizer


class MistralStringConverter:
    # OpenAI chat format to Mistral string format
    
    @staticmethod
    def string_formatter(example):
        """
        Convert OpenAI-style chat format to Mistral-7b-instruct-2.0 string format.
        
        Mistral-7b-instruct-2.0 format:
        - System message is part of user message with a <s> prefix
        - User message format: <s>[INST] {system_prompt} {user_content} [/INST]
        - Assistant message format: {assistant_content} </s>
        - For multi-turn: <s>[INST] {user_content} [/INST] {assistant_content} </s><s>[INST] {user_content} [/INST]
        """
        
        if 'messages' not in example:
            raise ValueError("No messages in the example")
            
        messages = example['messages']
        if len(messages) == 0: 
            raise ValueError("No messages in the example")
        
        # Check if there's a system message
        pt = 0
        if messages[0]['role'] == 'system':
            system_prompt = messages[0]['content']
            pt = 1
        else:
            system_prompt = default_system_prompt
        
        # Start building the conversation
        conversation = ""
        
        # Validate we have at least one user message after system
        if pt >= len(messages):
            raise ValueError("The message should be user-assistant alternation")
        
        # Process the messages
        while pt < len(messages):
            # Expect user message
            if messages[pt]['role'] != 'user':
                raise ValueError("The message should be user-assistant alternation")
            
            user_content = messages[pt]['content']
            pt += 1
            
            # For the first user message, include the system prompt
            if pt == 1 or (pt == 2 and messages[0]['role'] == 'system'):
                conversation += f"<s>[INST] {system_prompt}\n\n{user_content} [/INST]"
            else:
                conversation += f"<s>[INST] {user_content} [/INST]"
            
            # Check if there's an assistant response
            if pt < len(messages):
                if messages[pt]['role'] != 'assistant':
                    raise ValueError("The message should be user-assistant alternation")
                
                assistant_content = messages[pt]['content']
                conversation += f" {assistant_content} </s>"
                pt += 1
            else:
                # No assistant response yet, model should generate one
                pass
        
        return {'text': conversation}
    
    @staticmethod
    def string_formatter_completion_only(example):
        """
        Convert OpenAI-style chat format to Mistral format for completion only.
        
        This formatter expects the last message to be an assistant message
        and formats it for completion (without the end-of-turn token).
        """
        
        if 'messages' not in example:
            raise ValueError("No messages in the example")
            
        messages = example['messages']
        if len(messages) == 0: 
            raise ValueError("No messages in the example")
        
        # Validate that last message is from assistant
        if messages[-1]['role'] != 'assistant':
            raise ValueError("Completion only mode should end with a message from assistant")
        
        # Check if there's a system message
        pt = 0
        if messages[0]['role'] == 'system':
            system_prompt = messages[0]['content']
            pt = 1
        else:
            system_prompt = default_system_prompt
        
        # Start building the conversation
        conversation = ""
        
        # Process messages up to the last one
        while pt < len(messages) - 1:
            # Expect user message
            if messages[pt]['role'] != 'user':
                raise ValueError("The message should be user-assistant alternation")
            
            user_content = messages[pt]['content']
            pt += 1
            
            # For the first user message, include the system prompt
            if pt == 1 or (pt == 2 and messages[0]['role'] == 'system'):
                conversation += f"<s>[INST] {system_prompt}\n\n{user_content} [/INST]"
            else:
                conversation += f"<s>[INST] {user_content} [/INST]"
            
            # Check if there's an assistant response (except for the last one)
            if pt < len(messages) - 1:
                if messages[pt]['role'] != 'assistant':
                    raise ValueError("The message should be user-assistant alternation")
                
                assistant_content = messages[pt]['content']
                conversation += f" {assistant_content} </s>"
                pt += 1
        
        # Add the final assistant message without the closing tag for completion
        conversation += f" {messages[-1]['content']}"
        
        return {'text': conversation}
    
    @staticmethod
    def conversion_to_mistral_style_string(dataset):
        redundant_columns = list(dataset.features.keys())
        dataset = dataset.map(MistralStringConverter.string_formatter, remove_columns=redundant_columns)
        return dataset


class CustomDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensures that the loss is only
    calculated on the completion made by the assistant.
    """

    def __init__(
        self,
        response_template = None,
        instruction_template: Optional[Union[str, List[int]]] = None,
        num_shift_tokens: Optional[int] = 0,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        # For Mistral, we use [/INST] as the response template
        self.response_template = [" [/INST]"]
        # Convert to token IDs
        self.response_token_ids = [self.tokenizer.encode(template, add_special_tokens=False) for template in self.response_template]

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index
        self.num_shift_tokens = num_shift_tokens

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            
            for i in range(len(examples)):
                response_token_ids_start_idx = None
                
                for template_idx, template in enumerate(self.response_token_ids):
                    if response_token_ids_start_idx is not None:
                        break

                    for idx in np.where(batch["labels"][i] == template[0])[0]:
                        if (
                            idx + len(template) <= len(batch["labels"][i]) and
                            template
                            == batch["labels"][i][idx : idx + len(template)].tolist()
                        ):
                            response_token_ids_start_idx = idx

                    if response_token_ids_start_idx is None:
                        continue          
                    else:
                        response_token_ids_end_idx = response_token_ids_start_idx + len(template) + self.num_shift_tokens

                        # Make pytorch loss function ignore all tokens up through the end of the response key
                        batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"the raw tokens for this sequence are {batch['input_ids'][i]}"
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

        else:
            raise ValueError("Instruction template is not None, which is not supported in this version of the data collator")

        return batch


class AugmentedSafetyDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator for safety training with harmful and refusal pairs.
    """

    def __init__(
        self,
        response_template = None,
        instruction_template: Optional[Union[str, List[int]]] = None,
        num_shift_tokens: Optional[int] = 0,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        # For Mistral, we use [/INST] as the response template
        self.response_template = [" [/INST]"]
        # Convert to token IDs
        self.response_token_ids = [self.tokenizer.encode(template, add_special_tokens=False) for template in self.response_template]

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index
        self.num_shift_tokens = num_shift_tokens

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        
        harmful_input_ids = [torch.tensor(example['harmful_input_ids'], dtype=torch.long) for example in examples]
        refusal_input_ids = [torch.tensor(example['refusal_input_ids'], dtype=torch.long) for example in examples]
        harmful_attention_mask = [torch.tensor(example['harmful_attention_mask'], dtype=torch.long) for example in examples]
        refusal_attention_mask = [torch.tensor(example['refusal_attention_mask'], dtype=torch.long) for example in examples]

        max_length = max(max(seq.size(0) for seq in harmful_input_ids), max(seq.size(0) for seq in refusal_input_ids))

        # Pad sequences
        harmful_input_ids = torch.stack([F.pad(input_id, (0, max_length - input_id.size(0)), "constant", self.tokenizer.pad_token_id) for input_id in harmful_input_ids])
        refusal_input_ids = torch.stack([F.pad(input_id, (0, max_length - input_id.size(0)), "constant", self.tokenizer.pad_token_id) for input_id in refusal_input_ids])
        harmful_attention_mask = torch.stack([F.pad(mask, (0, max_length - mask.size(0)), "constant", 0) for mask in harmful_attention_mask])
        refusal_attention_mask = torch.stack([F.pad(mask, (0, max_length - mask.size(0)), "constant", 0) for mask in refusal_attention_mask])

        batch = {
            'harmful_input_ids': harmful_input_ids,
            'harmful_attention_mask': harmful_attention_mask,
            'refusal_input_ids': refusal_input_ids,
            'refusal_attention_mask': refusal_attention_mask
        }

        labels = batch["harmful_input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
        batch['harmful_labels'] = labels

        labels = batch["refusal_input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
        batch['refusal_labels'] = labels

        if self.instruction_template is None:
            
            for partition in ['harmful', 'refusal']:

                for i in range(len(examples)):

                    response_token_ids_start_idx = None

                    for template_idx, template in enumerate(self.response_token_ids):
                        if response_token_ids_start_idx is not None:
                            break

                        for idx in np.where(batch[f"{partition}_labels"][i] == template[0])[0]:
                            if (
                                idx + len(template) <= len(batch[f"{partition}_labels"][i]) and
                                template
                                == batch[f"{partition}_labels"][i][idx : idx + len(template)].tolist()
                            ):
                                response_token_ids_start_idx = idx

                        if response_token_ids_start_idx is None:
                            continue          
                        else:
                            response_token_ids_end_idx = response_token_ids_start_idx + len(template) + self.num_shift_tokens

                            # Make pytorch loss function ignore all tokens up through the end of the response key
                            batch[f"{partition}_labels"][i, :response_token_ids_end_idx] = self.ignore_index

                    if response_token_ids_start_idx is None:
                        warnings.warn(
                            f"Could not find response key `{self.response_template}` in the "
                            f'following instance: {self.tokenizer.decode(batch[f"{partition}_labels"][i])} '
                            f"This instance will be ignored in loss calculation. "
                            f"Note, if this happens often, consider increasing the `max_seq_length`."
                        )
                        batch[f"{partition}_labels"][i, :] = self.ignore_index

        else:
            raise ValueError("Instruction template is not None, which is not supported in this version of the data collator")

        return batch