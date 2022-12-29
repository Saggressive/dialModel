# Copyright 2021 Condenser Author All rights reserved.
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

import os
import warnings

import torch
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from transformers import BertModel, BertConfig, AutoModel, AutoModelForMaskedLM, AutoConfig, PretrainedConfig, \
    RobertaModel
from transformers.models.bert.modeling_bert import BertPooler, BertOnlyMLMHead, BertPreTrainingHeads, BertLayer
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPooling, MaskedLMOutput
from transformers.models.roberta.modeling_roberta import RobertaLayer
from transformers.activations import ACT2FN

from arguments import DataTrainingArguments, ModelArguments, CoCondenserPreTrainingArguments
from transformers import TrainingArguments
import logging

logger = logging.getLogger(__name__)

class BertIntermediate(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(2304, 3072)
        self.intermediate_act_fn = ACT2FN['gelu']
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(3072, 768)
        self.LayerNorm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.intermediate = BertIntermediate()
        self.out = BertOutput()
    def forward(self, hidden_states: torch.Tensor):
        intermediate_output = self.intermediate(hidden_states)
        last_ouput = self.out(intermediate_output)
        return last_ouput

class CondenserForPretraining(nn.Module):
    def __init__(
        self,
        bert: BertModel,
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        train_args: TrainingArguments
    ):
        super(CondenserForPretraining, self).__init__()
        self.lm = bert
        self.mlp = MLP()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.relative_position_embeddings = nn.Embedding(512, 768)
        self.model_args = model_args
        self.train_args = train_args
        self.data_args = data_args

    def forward(self, model_input, labels, left, right, pos_ids):

        attention_mask = self.lm.get_extended_attention_mask(
            model_input['attention_mask'],
            model_input['attention_mask'].shape,
            model_input['attention_mask'].device
        )

        lm_out: MaskedLMOutput = self.lm(
            **model_input,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )

        sbo_loss = torch.tensor(0,dtype=torch.float)
        if self.model_args.use_sbo:
            hidden_states=lm_out.hidden_states[-1]
            batch_size,sql_len,_=hidden_states.size()
            sbo_labels = labels[labels!=-100]
            mask_hidden_states = hidden_states[labels!=-100]
            left_hidden_states = torch.gather(hidden_states, dim=1, \
                index=left.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2]))
            left_hidden_states = left_hidden_states[labels!=-100]
            right_hidden_states = torch.gather(hidden_states, dim=1, \
                index=right.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2]))
            right_hidden_states = right_hidden_states[labels!=-100]
            assert mask_hidden_states.shape==left_hidden_states.shape
            assert mask_hidden_states.shape==right_hidden_states.shape
            # position_ids = self.lm.bert.embeddings.position_ids[:,:sql_len]
            position_embedding = self.relative_position_embeddings(pos_ids)
            # position_embedding = position_embedding.repeat(batch_size,1,1)
            position_embedding = position_embedding[labels!=-100]
            assert mask_hidden_states.shape==position_embedding.shape
            sbo_hidden_states = torch.cat([left_hidden_states,right_hidden_states,position_embedding],dim=1)
            sbo_hidden_states = self.mlp(sbo_hidden_states)
            sbo_loss=self.mlm_loss(sbo_hidden_states, sbo_labels)
            loss = lm_out.loss + sbo_loss
        else:
            loss = lm_out.loss
        return loss , sbo_loss , lm_out.loss


    def mlm_loss(self, hiddens, labels):
        pred_scores = self.lm.cls(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            labels.view(-1)
        )
        return masked_lm_loss


    @classmethod
    def from_pretrained(
            cls, model_args: ModelArguments, data_args: DataTrainingArguments, train_args: TrainingArguments,
            *args, **kwargs
    ):
        hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)
        model = cls(hf_model, model_args, data_args, train_args)
        path = args[0]
        if os.path.exists(os.path.join(path, 'model.pt')):
            logger.info('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = model.load_state_dict(model_dict, strict=False)
        return model

    @classmethod
    def from_config(
            cls,
            config: PretrainedConfig,
            model_args: ModelArguments,
            data_args: DataTrainingArguments,
            train_args: TrainingArguments,
    ):
        hf_model = AutoModelForMaskedLM.from_config(config)
        hf_model.cuda()
        model = cls(hf_model, model_args, data_args, train_args)

        return model

    def save_pretrained(self, output_dir: str):
        self.lm.save_pretrained(output_dir)
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('lm')]
        warnings.warn(f'omiting {len(hf_weight_keys)} transformer weights')
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))
