# Copyright 2023-2025 SGLang Team
# Copyright Amazon.com, Inc. or its affiliates.
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import Any, Callable
from metatool.rl import reward
import torch

from verl.protocol import DataProto

RawRewardFn = Callable[..., Any]


def _select_rm_score_fn(data_source):
    """
    Select the corresponding reward scoring function based on data source
    
    Args:
        data_source (str): Data source identifier
        
    Returns:
        function: Corresponding scoring computation function
        
    Raises:
        NotImplementedError: Raised when data source is not supported
    """
    if "rlla" in data_source:
        return reward.compute_score
    else:
        raise NotImplementedError


class AbstractRewardManager():
    """
    Reward manager class
    
    Responsible for computing and managing various reward scores during PPO training, including:
    - Total reward score
    - Format score
    - Correctness score
    - Length score
    """

    def __init__(self, tokenizer, num_examine) -> None:
        """
        Initialize reward manager
        
        Args:
            tokenizer: Tokenizer for decoding token sequences
            num_examine (int): Number of decoded response batches to print to console
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # Number of decoded response batches to print to console

    def __call__(self, data: DataProto, step: int):
        """
        Main method for computing reward scores
        
        We will gradually extend this function based on available datasets
        
        Args:
            data (DataProto): Protocol object containing batch data
            step (int): Current training step
            
        Returns:
            tuple: Tuple containing four tensors:
                - reward_tensor: Total reward score
                - format_tensor: Format score
                - correctness_tensor: Correctness score
                - length_tensor: Length score
        """

        # If rm scores already exist, return directly; otherwise compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        # Initialize various reward tensors
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        format_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        correctness_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        length_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # Record already printed data sources to avoid duplicate printing
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # Get single data item (DataProtoItem)

            # Get prompt IDs and length
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            # Calculate valid prompt length and IDs
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # Get response IDs and valid length
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode complete sequence (prompt + response)
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            # Get ground truth answer
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # Select corresponding reward scoring function
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            # Compute various scores
            score, fomrat_score, correctness_score, length_score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, step=step)
            
            # Store scores in corresponding tensors
            reward_tensor[i, valid_response_length - 1] = score
            format_tensor[i, valid_response_length - 1] = fomrat_score
            correctness_tensor[i, valid_response_length - 1] = correctness_score
            length_tensor[i, valid_response_length - 1] = length_score

            # Manage data source printing logic
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            # Control printing quantity to avoid excessive output
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor, format_tensor, correctness_tensor, length_tensor
