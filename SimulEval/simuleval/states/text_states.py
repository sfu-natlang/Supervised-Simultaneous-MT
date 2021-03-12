# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import codecs
import torch
from torch import Tensor
from typing import Dict, List, Optional
from . states import ListEntry, BiSideEntries, BaseStates
from mosestokenizer import *
from subword_nmt.apply_bpe import BPE

from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS


class TextStates(BaseStates):

    def __init__(self, args, client, instance_id, agent):
        super(TextStates, self).__init__(args, client, instance_id, agent)
        self.en_tokenize = MosesTokenizer('en')
        bpe_code = codecs.open(
            '/Users/alinejad/Desktop/SFU/Research/Speech-to-text-transation/Supervised-fairseq/data-bin/wmt14_en_de/bpe_code',
            encoding='utf-8'
        )
        self.bpe = BPE(bpe_code)
        self.add_eos = True  # Adds eos at the end of each partial source segment before translating.
        self.partial_translations = []

        self.num_reads = 0
        self.num_writes = 0

        max_len = 400
        self.encoder_embed_dim = self.agent.nmt_model[0].decoder.embed_dim
        self.decoder_embed_dim = self.agent.nmt_model[0].decoder.output_embed_dim
        self.feat_len = 1 + self.encoder_embed_dim + self.decoder_embed_dim

        self.input_features = torch.zeros(self.feat_len, dtype=torch.float)
        self.incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.agent.policy_generator.agent_models.models_size)
            ],
        )
        self.prior_inputs = (torch.zeros(1, max_len, self.feat_len).float())
        self.prior_outputs = (torch.zeros(1, max_len).long())


    def update_source(self, num_segment=1):
        super(TextStates, self).update_source(num_segment)
        self.num_reads += 1
        step = self.num_reads - 1 + self.num_writes

        sample = self.prepare_sample()
        self.partial_translations.append(self.agent.generate_translation(sample)[0])

        local_num_write = self.num_writes

        if not self.finish_hypo() and self.num_writes >= len(self.partial_translations[-1]['decoder_out']):
            local_num_write = len(self.partial_translations[-1]['decoder_out']) - 1

        self.input_features[:self.encoder_embed_dim] = self.partial_translations[-1]['encoder_out'][self.num_reads-1]
        self.input_features[self.encoder_embed_dim:self.feat_len-1] = self.partial_translations[-1]['decoder_out'][local_num_write]
        self.input_features[-1] = torch.mean(self.partial_translations[-1]['attention'][:, local_num_write])

        self.prior_inputs[0, step, :] = self.input_features
        self.prior_outputs[0, step] = self.agent.read_index
        

    def update_target(self, unit):
        super(TextStates, self).update_target(unit)
        self.num_writes += 1
        step = self.num_reads - 1 + self.num_writes

        if not self.finish_hypo():
            self.input_features[:self.encoder_embed_dim] = self.partial_translations[-1]['encoder_out'][self.num_reads-1]
            self.input_features[self.encoder_embed_dim:self.feat_len-1] = self.partial_translations[-1]['decoder_out'][self.num_writes]
            self.input_features[-1] = torch.mean(self.partial_translations[-1]['attention'][:, self.num_writes])

            self.prior_inputs[0, step, :] = self.input_features
            self.prior_outputs[0, step] = self.agent.write_index


    def prepare_sample(self):
        source_indexes = []
        #preprocessed = self.en_tokenize(" ".join(self.source.value))
        for token in self.source.value:
            source_indexes.append(
                self.agent.src_dict.index(token)
            )

        # Add EOS token at the end of each sequence
        if self.add_eos and source_indexes[-1] != self.agent.src_dict.eos_index:
            source_indexes.append(self.agent.src_dict.eos_index)

        sample = {
            'id': torch.tensor([1]),
            'nsentences': 1,
            'ntokens': len(source_indexes),
            'net_input': {
                'src_tokens': torch.unsqueeze(torch.tensor(source_indexes), dim=0),
                'src_lengths': torch.tensor([len(source_indexes)])
            }
        }
        return sample


    def segment_to_units(self, segment):
        units = []
        for token in self.en_tokenize(segment):
            units += self.bpe.process_line(token).split()
        return units


    def units_to_segment(self, unit_queue):
        if unit_queue.value[-1].endswith('@@'):
            return None
        string_len = len(unit_queue)
        return "".join([unit_queue.pop() for _ in range(string_len)]).replace("@@", "")
