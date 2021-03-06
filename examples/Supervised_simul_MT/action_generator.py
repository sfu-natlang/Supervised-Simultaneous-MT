# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import search, utils
from fairseq.models import FairseqIncrementalDecoder
from torch import Tensor
from examples.Supervised_simul_MT.data import agent_utils
from examples.Supervised_simul_MT.models.agent_lstm_5 import AgentLSTMModel5
from examples.Supervised_simul_MT.models.agent_lstm_4 import AgentLSTMModel4
from examples.Supervised_simul_MT.models.agent_lstm_3 import AgentLSTMModel3
from examples.Supervised_simul_MT.models.agent_lstm_2 import AgentLSTMModel2
from examples.Supervised_simul_MT.models.agent_lstm import AgentLSTMModel
from examples.Supervised_simul_MT.models.agent_lstm_0 import AgentLSTMModel0
from examples.Supervised_simul_MT.models import waitk_transformer


class ActionGenerator(nn.Module):
    def __init__(
            self,
            models,
            tgt_dict,
            src_dict,
            beam_size=1,
            max_len_a=0,
            max_len_b=200,
            min_len=1,
            normalize_scores=True,
            len_penalty=1.0,
            unk_penalty=0.0,
            temperature=1.0,
            match_source_len=False,
            no_repeat_ngram_size=0,
            search_strategy=None,
            eos=None,
            symbols_to_strip_from_output=None,
            lm_model=None,
            lm_weight=1.0,
            has_target=True,
            agent_arch=None,
            teacher_forcing_rate=1.,
            distort_action_sequence=False,
    ):
        """Generates translations of a given source sentence.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        super().__init__()
        if isinstance(models, EnsembleModel):
            self.model = models
        else:
            self.model = EnsembleModel(models)
        self.tgt_dict = tgt_dict
        self.src_dict = src_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len

        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size

        # If True then _generate() returns Encoder and Decoder outputs as well.
        self.simple_models = ['agent_lstm', 'agent_lstm_0', 'agent_lstm_big', 'agent_lstm_0_big', None]
        self.all_features = False if agent_arch in self.simple_models else True
        self.has_target = has_target
        self.teacher_forcing_rate = teacher_forcing_rate
        self.distort_action_sequence = distort_action_sequence

        assert temperature > 0, "--temperature must be greater than 0"

        self.search = (
            search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        )
        # We only need to set src_lengths in LengthConstrainedBeamSearch.
        # As a module attribute, setting it would break in multithread
        # settings when the model is shared.
        self.should_set_src_lengths = (
                hasattr(self.search, "needs_src_lengths") and self.search.needs_src_lengths
        )

        self.model.eval()

        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.lm_model.eval()

    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def forward(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            prefix_tokens: Optional[Tensor] = None,
            bos_token: Optional[int] = None,
    ):
        """Generate a batch of translations.

        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    # TODO(myleott): unused, deprecate after pytorch-translate migration
    def generate_batched_itr(self, data_itr, beam_size=None, cuda=False, timer=None):
        """Iterate over a batched dataset and yield individual translations.
        Args:
            cuda (bool, optional): use GPU for generation
            timer (StopwatchMeter, optional): time generations
        """
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if "net_input" not in s:
                continue
            input = s["net_input"]
            # model.forward normally channels prev_output_tokens into the decoder
            # separately, but SequenceGenerator directly calls model.encoder
            encoder_input = {
                k: v for k, v in input.items() if k != "prev_output_tokens"
            }
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(encoder_input)
            if timer is not None:
                timer.stop(sum(len(h[0]["tokens"]) for h in hypos))
            for i, id in enumerate(s["id"].data):
                # remove padding
                src = utils.strip_pad(input["src_tokens"].data[i, :], self.pad)
                ref = (
                    utils.strip_pad(s["target"].data[i, :], self.pad)
                    if s["target"] is not None
                    else None
                )
                yield id, src, ref, hypos[i]

    def extract_incremental_samples(self, sample):
        incremental_sample = {
            k: v for k, v in sample.items() if k != "net_input"
        }
        new_net_input = []
        sample_net_input = sample['net_input']

        # If the first element of a sample is pad, replace it with unk. (We'll get errors when generating translations)
        first = sample_net_input['src_tokens'][:, 0]
        if self.pad in first:
            for batch_index in range(sample_net_input['src_tokens'].shape[0]):
                if sample_net_input['src_tokens'][batch_index, 0] == self.pad:
                    sample_net_input['src_tokens'][batch_index, 0] = self.unk

        max_len = sample_net_input['src_tokens'].shape[1]
        for i in range(max_len):
            num_undone = (sample_net_input['src_lengths'] > i).sum()
            source_tokens = sample_net_input['src_tokens'][:num_undone, : i+1]

            # Add EOS to unfinished sentences
            last_elements = torch.tensor([self.eos if element[-1] not in [self.eos, self.pad] else self.pad
                                          for element in source_tokens], device=source_tokens.device)
            source_tokens = torch.cat((source_tokens, torch.unsqueeze(last_elements, 1)), 1)
            new_net_input.append(
                {'src_tokens': source_tokens,
                 'src_lengths': torch.min(sample_net_input['src_lengths'][:num_undone],
                                          torch.tensor([i + 2] * num_undone, device=source_tokens.device)),
                 'full_src_lengths': sample_net_input['src_lengths'][:num_undone]
                     }  # full_src_lengths will be used in _generate_train() to determine the length of target sentences.
            )
        incremental_sample['net_input'] = new_net_input
        return incremental_sample

    def prepare_post_process(self, src_token):
        src_str = None
        src_tokens = utils.strip_pad(
            src_token, self.tgt_dict.pad()
        )
        if self.src_dict is not None:
            src_str = self.src_dict.string(src_tokens)

        return src_str

    def extend_sample(self, translations, sample):
        net_input = sample['net_input']
        action_seq = sample['action_seq']
        extended_translation = []
        for index, translation in enumerate(translations):
            subsets_generated = [subset[0] for subset in translation]
            extended_sample = translation[-1][0]
            extended_sample['src_tokens'] = net_input['src_tokens'][index]
            extended_sample['action_seq'] = action_seq[index]
            extended_sample['subsets'] = subsets_generated

            # Generating translations based on action sequences.
            generateed_translation = []
            i, j = 0, 0
            read, write = 4, 5
            for action in action_seq[index]:
                if action == read:
                    j = j + 1
                else:
                    generateed_translation.append(
                        subsets_generated[j-1]['tokens'][i] if i < len(subsets_generated[j-1]['tokens'])
                        else subsets_generated[j-1]['tokens'][-2]  # The last element is the eos token. We take the one before.
                    )
                    i = i + 1

            extended_sample['tokens'] = generateed_translation
            extended_translation.append(extended_sample)
        return extended_translation

    def generate_action_sequence(self, translations, net_input=None, post_process=True):
        extended_translations = []
        src_tokens = net_input['src_tokens']
        src_lengths = net_input['src_lengths']
        for index, translation in enumerate(translations):
            extended_translations.append(
                self._generate_action_sequence(translation, src_tokens[index],
                src_lengths[index], post_process)
            )
        return extended_translations

    def _generate_action_sequence(self, sample, src_token, src_length, post_process):
        i, j = 0, 0
        action_seq = [0]
        trg_Gen = sample[-1][0]['tokens']

        trg_length = len(trg_Gen)

        # Generate action based on the first beam.
        subsets_generated = [subset[0] for subset in sample]

        if post_process:
            src_str = self.prepare_post_process(src_token)
            for index, sub_gen in enumerate(subsets_generated):
                _, hypo_str, _ = utils.post_process_prediction(
                    hypo_tokens=sub_gen["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=sub_gen["alignment"],
                    align_dict=None,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=None,
                    extra_symbols_to_ignore=None,
                )
                hypo_str = hypo_str.strip().split(' ') + ['<EOS>']
                subsets_generated[index]["hypo_str"] = hypo_str
            trg_Gen = hypo_str
            trg_length = len(trg_Gen)

        while i < trg_length:
            subset = subsets_generated[j]['hypo_str'] if post_process else subsets_generated[j]['tokens']
            if i < len(subset) and subset[i] == trg_Gen[i]:
                action_seq.append(1)
                i = i + 1
            elif j + 1 < src_length:
                if j + 3 == src_length:     # The last two translations are the same
                    action_seq.append(0)
                    j = j + 1
                action_seq.append(0)
                j = j + 1
        # # Debug
        # if j + 1 < src_length:
        #     # print("WARNING --> Some zeros added")
        #     while j + 1 < src_length:
        #         action_seq.append(0)
        #         j = j + 1

        # assert (src_length + trg_length == len(action_seq)), [src_length, trg_length, len(action_seq)]
        extended_sample = sample[-1][0]
        extended_sample['src_tokens'] = src_token
        extended_sample['action_seq'] = action_seq
        extended_sample['subsets'] = subsets_generated
        return extended_sample

    def _generate_waitk_action_sequence(self, sample, src_token, src_length, post_process, k=5):
        i, j = 0, 0
        action_seq = [0]
        trg_Gen = sample[-1][0]['tokens']

        trg_length = len(trg_Gen)

        # Generate action based on the first beam.
        subsets_generated = [subset[0] for subset in sample]

        if post_process:
            src_str = self.prepare_post_process(src_token)
            for index, sub_gen in enumerate(subsets_generated):
                _, hypo_str, _ = utils.post_process_prediction(
                    hypo_tokens=sub_gen["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=sub_gen["alignment"],
                    align_dict=None,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=None,
                    extra_symbols_to_ignore=None,
                )
                hypo_str = hypo_str.strip().split(' ') + ['<EOS>']
                subsets_generated[index]["hypo_str"] = hypo_str
            trg_Gen = hypo_str
            trg_length = len(trg_Gen)

        while i < trg_length and j + 1 < src_length:
            if j + 1 < k:  # +1 for the extra read added at the beginning.
                action_seq.append(0)
                j = j + 1
            else:
                action = 1 - action_seq[-1]  # Either 0 or 1
                action_seq.append(action)
                j = j + action
                i = i + (1-action)

        # Debug
        if j + 1 < src_length:
            # print("WARNING --> Some zeros added")
            while j + 1 < src_length:
                action_seq.append(0)
                j = j + 1
        if i < trg_length:
            while i < trg_length:
                action_seq.append(1)
                i = i + 1

        assert (src_length + trg_length == len(action_seq)), [src_length, trg_length, len(action_seq)]
        extended_sample = sample[-1][0]
        extended_sample['src_tokens'] = src_token
        extended_sample['action_seq'] = action_seq
        extended_sample['subsets'] = subsets_generated
        return extended_sample

    def fix_oracle_action_sequence(self, sample, dist_index):
        new_action_sequence = sample['action_seq'][:dist_index]
        new_sample = sample.copy()
        eos_index = 2
        i = sum(new_action_sequence)
        j = len(new_action_sequence) - i - 1

        trg_Gen = sample['tokens']
        trg_length = len(trg_Gen)
        subsets = sample['subsets']
        src_length = len(subsets)

        if sample['action_seq'][dist_index] == 0:
            # The distortion is not valid and we will return None
            if i >= len(subsets[j]['tokens']):
                return None
            elif subsets[j]['tokens'][i] == eos_index:
                return None

        distorted_action = 1 - sample['action_seq'][dist_index]
        new_action_sequence.append(distorted_action)   # Add distorted action
        new_action_sequence.append(-1)   # To mark where distortion happens. Will be used during training.
        i = i + distorted_action
        j = j + (1-distorted_action)

        while i < trg_length:
            subset = subsets[j]['tokens']
            if i < len(subset) and subset[i] == trg_Gen[i]:
                new_action_sequence.append(1)
                i = i + 1
            elif j + 1 < src_length:
                if j + 3 == src_length:     # The last two translations are the same
                    new_action_sequence.append(0)
                    j = j + 1
                new_action_sequence.append(0)
                j = j + 1
        # Debug
        # if j + 1 < src_length:
        #     # print("WARNING --> Some zeros added")
        #     while j + 1 < src_length:
        #         new_action_sequence.append(0)
        #         j = j + 1

        # assert (src_length + trg_length == len(new_action_sequence) - 1), \
        #     [src_length, trg_length, len(new_action_sequence)-1]
        new_sample['action_seq'] = new_action_sequence
        return new_sample

    def distort_oracle_action_sequence(self, samples):
        distort_rtow = 1
        distort_wtor = 1
        distorted_samples = []
        for index, sample in enumerate(samples):
            action_seq = torch.tensor(sample['action_seq'])
            read_idxs = (action_seq == 0).nonzero().squeeze()
            write_idxs = (action_seq == 1).nonzero().squeeze()

            read_idxs = read_idxs[read_idxs != 0]  # we don't want to distort the first read

            num_distort_rtow = min(distort_rtow, len(read_idxs))
            num_distort_wtor = min(distort_wtor, len(write_idxs))

            if read_idxs.dim() == 0 or write_idxs.dim() == 0 or len(read_idxs) <= 3 or len(write_idxs) <= 2:
                # We don't want to distort short sequences.
                continue

            if num_distort_rtow != 0:
                distortion_rtow_idxs = read_idxs[
                    torch.randperm(len(read_idxs))[:num_distort_rtow]
                ]

                for distortion_idx in distortion_rtow_idxs:

                    if write_idxs[-1] < distortion_idx:
                        # We don't want to choose a read action after the last write action.
                        continue

                    distorted = self.fix_oracle_action_sequence(sample, distortion_idx)
                    if distorted is not None:
                        distorted['id_index'] = index     # We use it to restore id number.
                        distorted_samples.append(distorted)

            if num_distort_wtor != 0:
                distortion_wtor_idxs = write_idxs[
                    torch.randperm(len(write_idxs))[:num_distort_wtor]
                ]
                for distortion_idx in distortion_wtor_idxs:
                    if read_idxs[-2] < distortion_idx:
                        # If we have already seen the whole src, then we don't want to read anymore.
                        continue
                    distorted = self.fix_oracle_action_sequence(sample, distortion_idx)
                    if distorted is not None:
                        distorted['id_index'] = index  # We use it to restore id number.
                        distorted_samples.append(distorted)

        return distorted_samples

    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        incremental_samples = self.extract_incremental_samples(sample)
        for i in range(sample['net_input']['src_tokens'].shape[1]):
            subsample = {
                k: v for k, v in sample.items() if k != "net_input"
            }
            subsample['net_input'] = incremental_samples['net_input'][i]

            partial_translation = self._generate_train(subsample) \
                if self.has_target and random.random() <= self.teacher_forcing_rate \
                else self._generate(subsample, **kwargs)

            if i == 0:
                translations = [[partial] for partial in partial_translation]
            else:
                for index, trans in enumerate(partial_translation):
                    translations[index].append(trans)

        if sample['action_seq'] is not None:
            return self.extend_sample(translations, sample)
        if self.distort_action_sequence:
            samples_with_oracle = self.generate_action_sequence(translations, sample['net_input'], post_process=False)
            distorted_samples = self.distort_oracle_action_sequence(samples_with_oracle)
            return distorted_samples
        return self.generate_action_sequence(translations, sample['net_input'], post_process=False)

    def _generate_train(
            self,
            sample: Dict[str, Dict[str, Tensor]]
    ):
        net_input = sample["net_input"]
        target = sample["target"]

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = net_input["src_tokens"].size()[:2]

        # compute the encoder output for each beam
        encoder_outs = self.model.forward_encoder(net_input)
        decoder_out = []

        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        previous_tokens = (
            torch.zeros(bsz, target.size(1)).to(net_input["src_tokens"]).long().fill_(self.pad)
        )  # +2 for eos and pad
        previous_tokens[:, 0] = self.eos
        previous_tokens[:, 1:] = target[:bsz, :-1]

        lprobs, avg_attn_scores = self.model.forward_decoder(
            previous_tokens,
            encoder_outs,
            self.temperature,
            has_incremental=False
        )
        if avg_attn_scores is not None:
            # bsz x tgt_len x src_len --> bsz x src_len x tgt_len
            avg_attn_scores = avg_attn_scores.permute(0, 2, 1)

        if self.all_features:
            decoder_out = self.model.forward_decoder_features_only(
                previous_tokens,
                encoder_outs,
                has_incremental=False
            )

        # Take care of NaNs
        lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

        lprobs[:, :, self.pad] = -math.inf  # never select pad
        lprobs[:, :, self.unk] -= self.unk_penalty  # apply unk penalty

        top_predictions = torch.topk(lprobs, k=1)
        indices = torch.squeeze(top_predictions[1])

        if bsz == 1 and indices.size(0) != bsz:
            indices = indices.unsqueeze(0)

        finalized_sents = self.refine_hypos(
            indices,
            avg_attn_scores,
            encoder_outs,
            decoder_out,
        )
        return finalized_sents

    def _generate(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            prefix_tokens: Optional[Tensor] = None,
            constraints: Optional[Tensor] = None,
            bos_token: Optional[int] = None,
    ):
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception("expected src_tokens or source in net input")

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimenions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        max_len: int = -1
        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                self.model.max_decoder_positions() - 1,
            )
        assert (
                self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"

        # encoder_input = {
        #     k: v for k, v in net_input.items() if k != "prev_output_tokens"
        # }
        #        net_output = self.model(net_input)

        # compute the encoder output for each beam
        encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        orig_encoder_out = encoder_outs
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        # ensure encoder_outs is a List.
        assert encoder_outs is not None

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
                .to(src_tokens)
                .long()
                .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn: Optional[Tensor] = None
        decoder_out: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        finished = [
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens).to(src_tokens.device)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )

            lprobs, avg_attn_scores = self.model.forward_decoder(
                tokens[:, : step + 1],
                encoder_outs,
                incremental_states,
                self.temperature,
            )

            if self.all_features:
                decoder_out_scores = self.model.forward_decoder_features_only(
                    tokens[:, : step + 1],
                    encoder_outs,
                    incremental_states,
                )

            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, : step + 1])
                probs = self.lm_model.get_normalized_probs(
                    lm_out, log_probs=True, sample=None
                )
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                    prefix_tokens is not None
                    and step < prefix_tokens.size(1)
                    and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            if self.all_features:
                if decoder_out is None:
                    decoder_out = torch.empty(
                        bsz * beam_size, max_len + 2, decoder_out_scores.size(2)
                    ).to(scores)
                decoder_out[:, step + 1, :] = decoder_out_scores[:, 0, :]

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)

            if self.should_set_src_lengths:
                self.search.set_src_lengths(src_lengths)

            if self.no_repeat_ngram_size > 0:
                lprobs = self._no_repeat_ngram(tokens, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    encoder_outs,
                    decoder_out,
                    src_lengths,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                if decoder_out is not None:
                    decoder_out = decoder_out.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, decoder_out.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(
                tokens[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            if decoder_out is not None:
                decoder_out[:, :step + 2, :] = torch.index_select(
                    decoder_out[:, : step + 2, ], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized

    def _prefix_tokens(
            self, step: int, lprobs, scores, tokens, prefix_tokens, beam_size: int
    ):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        lprobs[prefix_mask] = torch.tensor(-math.inf).to(lprobs)
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
            -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
        )
        # if prefix includes eos, then we should make sure tokens and
        # scores are the same across all beams
        eos_mask = prefix_toks.eq(self.eos)
        if eos_mask.any():
            # validate that the first beam matches the prefix
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[
                         :, 0, 1: step + 1
                         ]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()

            # copy tokens, scores and lprobs from the first beam to all beams
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    def replicate_first_beam(self, tensor, mask, beam_size: int):
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))

    def refine_hypos(
            self,
            tokens,
            attn: Optional[Tensor],
            encoder_out: Optional[List],
            decoder_out: Optional[List],
    ):
        """
        Same as finalize_hypos() but will be used during training of Agent
        """

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(len(tokens))],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        # For every finished beam item
        for i in range(len(tokens)):
            # Remove all tokens generated after the first EOS token.
            current_token = tokens[i]
            if current_token.dim() == 0:
                current_token = torch.unsqueeze(current_token, dim=0)
            eos_idx = torch.nonzero(current_token==self.eos)
            if len(eos_idx) > 0:
                final_tokens = current_token[:int(eos_idx[0])+1]
            else:
                final_tokens = torch.cat(
                    (current_token, torch.tensor([self.eos]).to(tokens))
                ).to(tokens)

            if attn is not None:
                # remove padding tokens from attn scores
                hypo_attn = attn[i]
            else:
                hypo_attn = torch.empty(0)

            if self.all_features:
                hypo_encoder_out = encoder_out[0]['encoder_out'][:, i, :] \
                    if torch.is_tensor(encoder_out[0]['encoder_out']) \
                    else encoder_out[0]['encoder_out'][0][:, i, :]
                hypo_decoder_out = decoder_out[i]
            else:
                hypo_encoder_out = torch.empty(0)
                hypo_decoder_out = torch.empty(0)

            finalized[i].append(
                {
                    "tokens": final_tokens,
                    "score": torch.tensor(0.0).type(torch.float),
                    "attention": hypo_attn, # src_len x tgt_len
                    "encoder_out": hypo_encoder_out,
                    "decoder_out": hypo_decoder_out,
                    "alignment": torch.empty(0),
                    "positional_scores": torch.tensor(0.0).type(torch.float),
                }
            # score, alignment, positional_scores are there to keep output format
            )
        return finalized

    def finalize_hypos(
            self,
            step: int,
            bbsz_idx,
            eos_scores,
            tokens,
            scores,
            finalized: List[List[Dict[str, Tensor]]],
            finished: List[bool],
            beam_size: int,
            attn: Optional[Tensor],
            encoder_out:Optional[Tensor],
            decoder_out:Optional[Tensor],
            src_lengths,
            max_len: int,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone = tokens.index_select(0, bbsz_idx)[
                       :, 1: step + 2
                       ]  # skip the first index, which is EOS

        tokens_clone[:, step] = self.eos
        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, :, 1: step + 2]
            if attn is not None
            else None
        )
        decoder_out_clone = (
            decoder_out.index_select(0, bbsz_idx)[:, 1: step + 2, :]
            if decoder_out is not None
            else None
        )

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + 1) ** self.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)

        # The keys here are of the form "{sent}_{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # set() is not supported in script export
        sents_seen: Dict[str, Optional[Tensor]] = {}

        # For every finished beam item
        for i in range(bbsz_idx.size()[0]):
            idx = bbsz_idx[i]
            score = eos_scores[i]
            # sentence index in the current (possibly reduced) batch
            unfin_idx = idx // beam_size
            # sentence index in the original (unreduced) batch
            sent = unfin_idx + cum_unfin[unfin_idx]
            # Cannot create dict for key type '(int, int)' in torchscript.
            # The workaround is to cast int to string
            seen = str(sent.item()) + "_" + str(unfin_idx.item())
            if seen not in sents_seen:
                sents_seen[seen] = None

            if self.match_source_len and step > src_lengths[unfin_idx]:
                score = torch.tensor(-math.inf).to(score)

            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)

                if self.all_features:
                    hypo_encoder_out = encoder_out[0]['encoder_out'][0][:, i, :]
                    hypo_decoder_out = decoder_out_clone[i]
                else:
                    hypo_encoder_out = torch.empty(0)
                    hypo_decoder_out = torch.empty(0)


                finalized[sent].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": score,
                        "attention": hypo_attn, # src_len x tgt_len
                        "encoder_out": hypo_encoder_out,
                        "decoder_out": hypo_decoder_out,
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                    }
                )

        newly_finished: List[int] = []

        for seen in sents_seen.keys():
            # check termination conditions for this sentence
            sent: int = int(float(seen.split("_")[0]))
            unfin_idx: int = int(float(seen.split("_")[1]))

            if not finished[sent] and self.is_finished(
                    step, unfin_idx, max_len, len(finalized[sent]), beam_size
            ):
                finished[sent] = True
                newly_finished.append(unfin_idx)

        return newly_finished

    def is_finished(
            self,
            step: int,
            unfin_idx: int,
            max_len: int,
            finalized_sent_len: int,
            beam_size: int,
    ):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False

    def calculate_banned_tokens(
            self,
            tokens,
            step: int,
            gen_ngrams: List[Dict[str, List[int]]],
            no_repeat_ngram_size: int,
            bbsz_idx: int,
    ):
        tokens_list: List[int] = tokens[
                                 bbsz_idx, step + 2 - no_repeat_ngram_size: step + 1
                                 ].tolist()
        # before decoding the next token, prevent decoding of ngrams that have already appeared
        ngram_index = ",".join([str(x) for x in tokens_list])
        return gen_ngrams[bbsz_idx].get(ngram_index, torch.jit.annotate(List[int], []))

    def transpose_list(self, l: List[List[int]]):
        # GeneratorExp aren't supported in TS so ignoring the lint
        min_len = min([len(x) for x in l])  # noqa
        l2 = [[row[i] for row in l] for i in range(min_len)]
        return l2

    def _no_repeat_ngram(self, tokens, lprobs, bsz: int, beam_size: int, step: int):
        # for each beam and batch sentence, generate a list of previous ngrams
        gen_ngrams: List[Dict[str, List[int]]] = [
            torch.jit.annotate(Dict[str, List[int]], {})
            for bbsz_idx in range(bsz * beam_size)
        ]
        cpu_tokens = tokens.cpu()
        for bbsz_idx in range(bsz * beam_size):
            gen_tokens: List[int] = cpu_tokens[bbsz_idx].tolist()
            for ngram in self.transpose_list(
                    [gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]
            ):
                key = ",".join([str(x) for x in ngram[:-1]])
                gen_ngrams[bbsz_idx][key] = gen_ngrams[bbsz_idx].get(
                    key, torch.jit.annotate(List[int], [])
                ) + [ngram[-1]]

        if step + 2 - self.no_repeat_ngram_size >= 0:
            # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            banned_tokens = [
                self.calculate_banned_tokens(
                    tokens, step, gen_ngrams, self.no_repeat_ngram_size, bbsz_idx
                )
                for bbsz_idx in range(bsz * beam_size)
            ]
        else:
            banned_tokens = [
                torch.jit.annotate(List[int], []) for bbsz_idx in range(bsz * beam_size)
            ]
        for bbsz_idx in range(bsz * beam_size):
            lprobs[bbsz_idx][
                torch.tensor(banned_tokens[bbsz_idx]).long()
            ] = torch.tensor(-math.inf).to(lprobs)
        return lprobs


class SimultaneousGenerator(nn.Module):
    def __init__(
            self,
            nmt_models,
            agent_models,
            task=None,
            beam_size=1,
            max_len_a=0,
            max_len_b=200,
            min_len=1,
            normalize_scores=True,
            len_penalty=1.0,
            unk_penalty=0.0,
            temperature=1.0,
            match_source_len=False,
            no_repeat_ngram_size=0,
            search_strategy=None,
            eos=None,
            symbols_to_strip_from_output=None,
            lm_model=None,
            lm_weight=1.0
    ):
        """Generates translations of a given source sentence.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models,
                currently support fairseq.models.TransformerModel for scripting
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        super().__init__()
        self.task = task
        if isinstance(nmt_models, EnsembleModel):
            self.nmt_models = nmt_models
        else:
            self.nmt_models = EnsembleModel(nmt_models)

        if isinstance(agent_models, EnsembleModel):
            self.agent_models = agent_models
        else:
            self.agent_models = EnsembleModel(agent_models)

        for model in self.agent_models.models:
            if isinstance(model, AgentLSTMModel5):
                self.task.args.arch = "agent_lstm_5"
            elif isinstance(model, AgentLSTMModel4):
                self.task.args.arch = "agent_lstm_4"
            elif isinstance(model, AgentLSTMModel3):
                self.task.args.arch = "agent_lstm_3"
            elif isinstance(model, AgentLSTMModel2):
                self.task.args.arch = "agent_lstm_2"
            elif isinstance(model, AgentLSTMModel):
                self.task.args.arch = "agent_lstm"
            elif isinstance(model, AgentLSTMModel0):
                self.task.args.arch = "agent_lstm_0"
            else:
                raise NotImplementedError(
                    "The Agent is coming from an unknown model"
                )
        self.tgt_dict = self.task.tgt_dict
        self.src_dict = self.task.src_dict
        self.agt_dict = self.task.agent_dictionary

        self.pad = self.agt_dict.pad()
        self.unk = self.agt_dict.unk()
        self.eos = self.agt_dict.eos() if eos is None else eos
        self.bos = self.agt_dict.bos()
        self.read = self.agt_dict.indices['0']
        self.write = self.agt_dict.indices['1']

        self.symbols_to_strip_from_output = (
            symbols_to_strip_from_output.union({self.eos})
            if symbols_to_strip_from_output is not None
            else {self.eos}
        )
        self.vocab_size = len(self.agt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len

        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size

        # If True then _generate() returns Encoder and Decoder outputs as well.
        self.all_features = False

        self.has_target = self.task.has_target

        assert temperature > 0, "--temperature must be greater than 0"

        self.search = (
            search.BeamSearch(self.agt_dict) if search_strategy is None else search_strategy
        )
        # We only need to set src_lengths in LengthConstrainedBeamSearch.
        # As a module attribute, setting it would break in multithread
        # settings when the model is shared.
        self.should_set_src_lengths = (
                hasattr(self.search, "needs_src_lengths") and self.search.needs_src_lengths
        )

        self.nmt_models.eval()
        # self.agent_models.eval()

        self.lm_model = lm_model
        self.lm_weight = lm_weight
        if self.lm_model is not None:
            self.lm_model.eval()

    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def forward(
            self,
            sample: Dict[str, Dict[str, Tensor]],
            prefix_tokens: Optional[Tensor] = None,
            bos_token: Optional[int] = None,
    ):
        """Generate a batch of translations.

        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        """Generate translations. Match the api of other fairseq generators.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            constraints (torch.LongTensor, optional): force decoder to include
                the list of constraints
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """

        if isinstance(self.agent_models.models[0], AgentLSTMModel):
            agent_arch = "agent_lstm"
        elif isinstance(self.agent_models.models[0], AgentLSTMModel0):
            agent_arch = "agent_lstm_0"
        else:
            agent_arch = None
        action_generator = ActionGenerator(self.nmt_models, self.tgt_dict, self.src_dict,
                                           beam_size=self.beam_size, max_len_a=self.max_len_a, max_len_b=self.max_len_b,
                                           min_len=self.min_len, normalize_scores=self.normalize_scores,
                                           len_penalty=self.len_penalty, unk_penalty=self.unk_penalty,
                                           temperature=self.temperature, match_source_len=self.match_source_len,
                                           no_repeat_ngram_size=self.no_repeat_ngram_size, search_strategy=None,
                                           has_target=self.has_target, agent_arch=agent_arch)
        hypos = action_generator.generate(models, sample)
#        input_hypos = agent_utils.prepare_simultaneous_input(hypos, sample, self.task)

        final_hypos = self._generate(hypos, **kwargs) if not isinstance(self.agent_models.models[0], AgentLSTMModel5) \
            else self._generate_agent_5(hypos, **kwargs)

        return final_hypos

    def _generate(
            self,
            sample,
            prefix_tokens: Optional[Tensor] = None,
            constraints: Optional[Tensor] = None,
            bos_token: Optional[int] = None,
    ):
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.agent_models.models_size)
            ],
        )

        subsets = [hypo['subsets'] for hypo in sample]

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimenions (i.e. audio features)
        bsz = len(subsets)
        beam_size = self.beam_size
        device = subsets[0][0]['tokens'].device

        if isinstance(self.agent_models.models[0], AgentLSTMModel) or \
           isinstance(self.agent_models.models[0], AgentLSTMModel0):
            feat_len = 2
            all_features = False
        else:
            feat_len = 1 + subsets[0][0]['encoder_out'].shape[1] + subsets[0][0]['decoder_out'].shape[1]
            all_features = True

        num_reads = 1

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # max action len
        max_len = min(
            int(2*self.max_len_b),
            # exclude the EOS marker
            self.agent_models.max_decoder_positions() - 1,
            )
        assert (
                self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"

        # initialize buffers
        scores = (
            torch.zeros(bsz, max_len + 1).to(device).float()
        )  # +1 for eos; pad is never chosen for scoring
        prior_outputs = (
            torch.zeros(bsz, max_len + 2).to(device).long().fill_(self.pad)
        )  # +2 for eos and pad

        # Our agent do not generate the first action, since it's always READ.
        # So we pass READ as initial action.
        prior_outputs[:, 0] = self.read
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz).to(device).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        finished = [
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining

        prior_inputs = (torch.zeros(bsz, max_len + 2, feat_len).to(device).float()) if all_features \
            else (torch.zeros(bsz, max_len + 2, feat_len).to(device).long()) # +2 for eos and pad

        # The tokens we send at each step. Will be used to determine when to stop.
        prior_src_tokens = (
            torch.zeros(bsz, max_len + 2).to(device).int()
        )  # +2 for eos and pad
        prior_trg_tokens = (
            torch.zeros(bsz, max_len + 2).to(device).int()
        )  # +2 for eos and pad

        reorder_state: Optional[Tensor] = None
        batch_idxs = torch.arange(bsz, device=device)

        # Will store translated tokens selected by the Agent.
        selected_tokens = (
            torch.zeros(bsz, max_len + 2).to(device).int()
        )

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                self.agent_models.reorder_incremental_state(incremental_states, reorder_state)
            new_subsets = [subsets[index] for index in batch_idxs]
            current_input = agent_utils.infer_input_features(new_subsets, prior_outputs[:, :step+1], all_features)
            prior_inputs[:, step, :] = current_input['input_features']
            prior_src_tokens[:, step] = current_input['src_tokens']
            prior_trg_tokens[:, step] = current_input['trg_tokens']

            lprobs, avg_attn_scores = self.agent_models.forward_agent_decoder(
                prior_inputs[:, :step+1, :],
                prior_outputs[:, :step+1],
                incremental_states,
                self.temperature,
            )

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] = -math.inf  # never select unk
            lprobs[:, self.bos] = -math.inf  # never select bos
            lprobs[:, self.eos] = -math.inf  # never select eos, except when we say so

            lprobs[:, self.read] = torch.where(
                prior_src_tokens[:, step].eq(self.eos),
                torch.tensor(-math.inf, dtype=torch.float32, device=device),
                lprobs[:, self.read].float()
            )
            lprobs[:, self.write] = torch.where(
                prior_trg_tokens[:, step].eq(self.pad),
                torch.tensor(-math.inf, dtype=torch.float32, device=device),
                lprobs[:, self.write].float()
            )
            if step > 0:
                lprobs[:, self.write] = torch.where(
                    prior_trg_tokens[:, step-1].eq(self.eos),
                    torch.tensor(-math.inf, dtype=torch.float32, device=device),
                    lprobs[:, self.write].float()
                )
                lprobs[:, self.eos] = torch.where(
                    prior_src_tokens[:, step].eq(self.eos) &
                    prior_trg_tokens[:, step].eq(self.eos) &
                    prior_trg_tokens[:, step-1].eq(self.eos),
                    torch.tensor(math.inf, dtype=torch.float32, device=device),
                    lprobs[:, self.eos].float()
                )
                if not all_features:
                    lprobs[:, self.eos] = torch.where(
                        prior_src_tokens[:, step].eq(self.eos) &
                        prior_trg_tokens[:, step].eq(self.pad),
                        torch.tensor(math.inf, dtype=torch.float32, device=device),
                        lprobs[:, self.eos].float()
                    )

            # handle max length constraint
            if step >= max_len:
                lprobs[:, self.eos] = math.inf
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                    prefix_tokens is not None
                    and step < prefix_tokens.size(1)
                    and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)

            if self.no_repeat_ngram_size > 0:
                lprobs = self._no_repeat_ngram(prior_outputs, lprobs, bsz, beam_size, step)

            top_predictions = torch.topk(lprobs, k=1)
            indices = torch.squeeze(top_predictions[1])

            if indices.dim()==0:
                indices = torch.unsqueeze(indices, dim=0)

            eos_mask = indices.eq(self.eos)
            eos_idx = torch.squeeze((eos_mask == True).nonzero(), dim=1)
            active_idx = torch.squeeze((eos_mask == False).nonzero(), dim=1)

            finalized_sents: List[int] = []
            if eos_idx.numel() > 0:
                finalized_sents = self.finalize_hypos(
                    step,
                    eos_idx,
                    batch_idxs,
                    prior_outputs,
                    selected_tokens,
                    scores,
                    finalized,
                    finished,
                    attn,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.index_select(
                    batch_idxs, dim=0, index=active_idx
                )

                indices = indices[active_idx]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[active_idx]
                cands_to_ignore = cands_to_ignore[active_idx]

                scores = scores[active_idx]
                prior_outputs = prior_outputs[active_idx]
                prior_inputs = prior_inputs[active_idx]
                prior_src_tokens = prior_src_tokens[active_idx]
                prior_trg_tokens = prior_trg_tokens[active_idx]
                if attn is not None:
                    attn = attn.view(bsz, -1)[active_idx].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz

            # Select the next token for each of them
            prior_outputs[:, step + 1] = indices

            write_mask = indices.eq(self.write)
            write_idx = torch.squeeze((write_mask == True).nonzero(), dim=1)
            if write_idx.numel() > 0:
                num_writes = torch.sum(torch.eq(prior_outputs, 5), dim=1)
                selected_tokens[batch_idxs, num_writes-1] = torch.where(
                    write_mask,
                    prior_trg_tokens[:, step],
                    selected_tokens[batch_idxs, num_writes-1]
                )

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized

    def _generate_agent_5(
            self,
            sample,
            prefix_tokens: Optional[Tensor] = None,
            constraints: Optional[Tensor] = None,
            bos_token: Optional[int] = None,
    ):
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.agent_models.models_size)
            ],
        )

        subsets = [hypo['subsets'] for hypo in sample]

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimenions (i.e. audio features)
        bsz = len(subsets)
        beam_size = self.beam_size
        device = subsets[0][0]['tokens'].device

        if isinstance(self.agent_models.models[0], AgentLSTMModel) or \
           isinstance(self.agent_models.models[0], AgentLSTMModel0) or \
           isinstance(self.agent_models.models[0], AgentLSTMModel5):
            feat_len = 2
            all_features = False
        else:
            feat_len = 1 + subsets[0][0]['encoder_out'].shape[1] + subsets[0][0]['decoder_out'].shape[1]
            all_features = True

        num_prev_tokens = 3

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # max action len
        max_len = min(
            int(2*self.max_len_b),
            # exclude the EOS marker
            self.agent_models.max_decoder_positions() - 1,
            )
        assert (
                self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"

        # initialize buffers
        scores = (
            torch.zeros(bsz, max_len + 1).to(device).float()
        )  # +1 for eos; pad is never chosen for scoring
        prior_outputs = (
            torch.zeros(bsz, max_len + 2, num_prev_tokens).to(device).long().fill_(self.pad)
        )  # +2 for eos and pad

        # Our agent do not generate the first action, since it's always READ.
        # So we pass READ as initial action.
        prior_outputs[:, 0, -1] = self.read
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz).to(device).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        finished = [
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining

        prior_inputs = (torch.zeros(bsz, max_len + 2, feat_len).to(device).float()) if all_features \
            else (torch.zeros(bsz, max_len + 2, feat_len).to(device).long()) # +2 for eos and pad

        # The tokens we send at each step. Will be used to determine when to stop.
        prior_src_tokens = (
            torch.zeros(bsz, max_len + 2).to(device).int()
        )  # +2 for eos and pad
        prior_trg_tokens = (
            torch.zeros(bsz, max_len + 2).to(device).int()
        )  # +2 for eos and pad

        reorder_state: Optional[Tensor] = None
        batch_idxs = torch.arange(bsz, device=device)

        # Will store translated tokens selected by the Agent.
        selected_tokens = (
            torch.zeros(bsz, max_len + 2).to(device).int()
        )

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                self.agent_models.reorder_incremental_state(incremental_states, reorder_state)
            new_subsets = [subsets[index] for index in batch_idxs]
            current_input = agent_utils.infer_input_features(new_subsets, prior_outputs[:, :step+1, -1], all_features)
            prior_inputs[:, step, :] = current_input['input_features']
            prior_src_tokens[:, step] = current_input['src_tokens']
            prior_trg_tokens[:, step] = current_input['trg_tokens']

            lprobs, avg_attn_scores = self.agent_models.forward_agent_decoder(
                prior_inputs[:, :step+1, :],
                prior_outputs[:, :step+1],
                incremental_states,
                self.temperature,
            )

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] = -math.inf  # never select unk
            lprobs[:, self.bos] = -math.inf  # never select bos
            lprobs[:, self.eos] = -math.inf  # never select eos, except when we say so

            lprobs[:, self.read] = torch.where(
                prior_src_tokens[:, step].eq(self.eos),
                torch.tensor(-math.inf, dtype=torch.float32, device=device),
                lprobs[:, self.read].float()
            )
            lprobs[:, self.write] = torch.where(
                prior_trg_tokens[:, step].eq(self.pad),
                torch.tensor(-math.inf, dtype=torch.float32, device=device),
                lprobs[:, self.write].float()
            )
            if step > 0:
                lprobs[:, self.write] = torch.where(
                    prior_trg_tokens[:, step-1].eq(self.eos),
                    torch.tensor(-math.inf, dtype=torch.float32, device=device),
                    lprobs[:, self.write].float()
                )
                # lprobs[:, self.eos] = torch.where(
                #     prior_src_tokens[:, step].eq(self.eos) &
                #     prior_trg_tokens[:, step].eq(self.eos) &
                #     prior_trg_tokens[:, step-1].eq(self.eos),
                #     torch.tensor(math.inf, dtype=torch.float32, device=device),
                #     lprobs[:, self.eos].float()
                # )
                if not all_features:
                    lprobs[:, self.eos] = torch.where(
                        prior_trg_tokens[:, step-1].eq(self.eos) &
                        prior_outputs[:, step-1, -1].eq(self.write),
                        torch.tensor(math.inf, dtype=torch.float32, device=device),
                        lprobs[:, self.eos].float()
                    )

            # handle max length constraint
            if step >= max_len:
                lprobs[:, self.eos] = math.inf
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                    prefix_tokens is not None
                    and step < prefix_tokens.size(1)
                    and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)

            if self.no_repeat_ngram_size > 0:
                lprobs = self._no_repeat_ngram(prior_outputs, lprobs, bsz, beam_size, step)

            top_predictions = torch.topk(lprobs, k=1)
            indices = torch.squeeze(top_predictions[1])

            if indices.dim()==0:
                indices = torch.unsqueeze(indices, dim=0)

            eos_mask = indices.eq(self.eos)
            eos_idx = torch.squeeze((eos_mask == True).nonzero(), dim=1)
            active_idx = torch.squeeze((eos_mask == False).nonzero(), dim=1)

            finalized_sents: List[int] = []
            if eos_idx.numel() > 0:
                finalized_sents = self.finalize_hypos(
                    step,
                    eos_idx,
                    batch_idxs,
                    prior_outputs[:, :, -1],
                    selected_tokens,
                    scores,
                    finalized,
                    finished,
                    attn,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.index_select(
                    batch_idxs, dim=0, index=active_idx
                )

                indices = indices[active_idx]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[active_idx]
                cands_to_ignore = cands_to_ignore[active_idx]

                scores = scores[active_idx]
                prior_outputs = prior_outputs[active_idx]
                prior_inputs = prior_inputs[active_idx]
                prior_src_tokens = prior_src_tokens[active_idx]
                prior_trg_tokens = prior_trg_tokens[active_idx]
                if attn is not None:
                    attn = attn.view(bsz, -1)[active_idx].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz

            # Select the next token for each of them
            prior_outputs[:, step + 1, :-1] = prior_outputs[:, step, 1:]
            prior_outputs[:, step + 1, -1] = indices

            write_mask = indices.eq(self.write)
            write_idx = torch.squeeze((write_mask == True).nonzero(), dim=1)
            if write_idx.numel() > 0:
                num_writes = torch.sum(torch.eq(prior_outputs[:, :, -1], 5), dim=1)
                selected_tokens[batch_idxs, num_writes-1] = torch.where(
                    write_mask,
                    prior_trg_tokens[:, step],
                    selected_tokens[batch_idxs, num_writes-1]
                )

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized

    def _generate_simuleval(self, prior_input, prior_output, incremental_state):
        if prior_input.dim() == 1:
            prior_input = prior_input.unsqueeze(0).unsqueeze(0)
        if prior_output.dim() == 1:
            prior_output = prior_output.unsqueeze(0)
        lprobs, avg_attn_scores = self.agent_models.forward_agent_decoder(
            prior_input,
            prior_output,
            incremental_state,
            self.temperature,
        )
        lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

        lprobs[:, self.pad] = -math.inf  # never select pad
        lprobs[:, self.unk] = -math.inf  # never select unk
        lprobs[:, self.bos] = -math.inf  # never select bos
        lprobs[:, self.eos] = -math.inf  # never select eos, except when we say so

        top_predictions = torch.topk(lprobs, k=1)
        index = torch.squeeze(top_predictions[1])
        return index, incremental_state

    def _generate_beam(
            self,
            sample,
            prefix_tokens: Optional[Tensor] = None,
            constraints: Optional[Tensor] = None,
            bos_token: Optional[int] = None,
    ):
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.agent_models.models_size)
            ],
        )

        subsets = [hypo['subsets'] for hypo in sample]

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimenions (i.e. audio features)
        bsz = len(subsets)
        beam_size = self.beam_size
        device = subsets[0][0]['tokens'].device
        feat_len = 1 + subsets[0][0]['encoder_out'].shape[1] + subsets[0][0]['decoder_out'].shape[1]

        num_reads = 1

        if constraints is not None and not self.search.supports_constraints:
            raise NotImplementedError(
                "Target-side constraints were provided, but search method doesn't support them"
            )

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)

        # max action len
        max_len = min(
            int(2*self.max_len_b),
            # exclude the EOS marker
            self.agent_models.max_decoder_positions() - 1,
            )
        assert (
                self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(device).long()
        subsets = self.agent_models.reorder_list(subsets, new_order)

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(device).float()
        )  # +1 for eos; pad is never chosen for scoring
        prior_outputs = (
            torch.zeros(bsz * beam_size, max_len + 2)
                .to(device)
                .long()
                .fill_(self.pad)
        )  # +2 for eos and pad

        # Our agent do not generate the first action, since it's always READ.
        # So we pass READ as initial action.
        prior_outputs[:, 0] = self.read
        attn: Optional[Tensor] = None

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(device).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        finished = [
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(prior_outputs).to(device)
        cand_offsets = torch.arange(0, cand_size).type_as(prior_outputs).to(device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(prior_outputs)

        prior_inputs = (
            torch.zeros(bsz * beam_size, max_len + 2, feat_len)
                .to(device)
                .float()
        )  # +2 for eos and pad

        # The tokens we send at each step. Will be used to determine when to stop.
        prior_src_tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
                .to(device)
                .int()
        )  # +2 for eos and pad
        prior_trg_tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
                .to(device)
                .int()
        )  # +2 for eos and pad
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.agent_models.reorder_incremental_state(incremental_states, reorder_state)
                subsets = self.agent_models.reorder_list(subsets, reorder_state)

            current_input = agent_utils.infer_input_features(subsets, prior_outputs[:, :step+1])
            prior_inputs[:, step, :] = current_input['input_features']
            prior_src_tokens[:, step] = current_input['src_tokens']
            prior_trg_tokens[:, step] = current_input['trg_tokens']

            lprobs, avg_attn_scores = self.agent_models.forward_agent_decoder(
                prior_inputs[:, :step+1, :],
                prior_outputs[:, :step+1],
                incremental_states,
                self.temperature,
            )

            if self.lm_model is not None:
                lm_out = self.lm_model(tokens[:, : step + 1])
                probs = self.lm_model.get_normalized_probs(
                    lm_out, log_probs=True, sample=None
                )
                probs = probs[:, -1, :] * self.lm_weight
                lprobs += probs

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] = -math.inf  # never select unk
            lprobs[:, self.bos] = -math.inf  # never select bos
            lprobs[:, self.eos] = -math.inf  # never select eos, except we say so

            # If NMT reaches to the EOS in the src side, the agent should not select READ anymore.
            # lprobs[:, self.read] = torch.where(
            #     prior_src_tokens[:, step].eq(self.eos),
            #     torch.tensor(-math.inf, dtype=torch.float32),
            #     lprobs[:, self.read].float()
            # )
            #
            # # We have to chose WRITE for the first beam and EOS for the second beam.
            # lprobs[:, self.write] = torch.where(
            #     prior_src_tokens[:, step].eq(self.eos) & prior_trg_tokens[:, step-1].ne(self.eos),
            #     torch.tensor(2, dtype=torch.float32),
            #     lprobs[:, self.write].float()
            # )
            # lprobs[:, self.eos] = torch.where(
            #     prior_src_tokens[:, step].eq(self.eos) & prior_trg_tokens[:, step-1].ne(self.eos),
            #     torch.tensor(1, dtype=torch.float32),
            #     lprobs[:, self.eos].float()
            # )
            #
            # # Stopping criteria
            # if step > 0:
            #     lprobs[:, self.eos] = torch.where(
            #         prior_trg_tokens[:, step].eq(self.eos) &
            #         prior_src_tokens[:, step].eq(self.eos) &
            #         prior_trg_tokens[:, step-1].eq(self.eos),
            #         torch.tensor(1, dtype=torch.float32),
            #         lprobs[:, self.eos].float()
            #     )  # Wrong

            # Stopping criteria
            lprobs[:, self.eos] = torch.where(
                prior_src_tokens[:, step].eq(self.eos),
                torch.tensor(100, dtype=torch.float32),
                lprobs[:, self.eos].float()
            )
            if step > 0:
                lprobs[:, self.eos] = torch.where(
                    prior_trg_tokens[:, step-1].eq(self.eos),
                    torch.tensor(math.inf, dtype=torch.float32),
                    lprobs[:, self.eos].float()
                )

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if (
                    prefix_tokens is not None
                    and step < prefix_tokens.size(1)
                    and step < max_len
            ):
                lprobs, tokens, scores = self._prefix_tokens(
                    step, lprobs, scores, tokens, prefix_tokens, beam_size
                )
            # elif step < self.min_len:
            #     # minimum length constraint (does not apply if using prefix_tokens)
            #     lprobs[:, self.eos] = -math.inf

            # Record attention scores, only support avg_attn_scores is a Tensor
            if avg_attn_scores is not None:
                if attn is None:
                    attn = torch.empty(
                        bsz * beam_size, avg_attn_scores.size(1), max_len + 2
                    ).to(scores)
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)

            if self.no_repeat_ngram_size > 0:
                lprobs = self._no_repeat_ngram(prior_outputs, lprobs, bsz, beam_size, step)

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                prior_outputs[:, : step + 1],
                original_batch_idxs,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    prior_outputs,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    attn,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)
                feat_len = prior_inputs.shape[2]

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                prior_outputs = prior_outputs.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                prior_inputs = prior_inputs.view(bsz, -1, feat_len)[batch_idxs].view(new_bsz * beam_size, -1, feat_len)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1
                    )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
                )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            prior_outputs[:, : step + 1] = torch.index_select(
                prior_outputs[:, : step + 1], dim=0, index=active_bbsz_idx
            )
            prior_inputs[:, :] = torch.index_select(
                prior_inputs[:, :], dim=0, index=active_bbsz_idx
            )

            # Select the next token for each of them
            prior_outputs.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )

            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            if attn is not None:
                attn[:, :, : step + 2] = torch.index_select(
                    attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
                )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized

    def _prefix_tokens(
            self, step: int, lprobs, scores, tokens, prefix_tokens, beam_size: int
    ):
        """Handle prefix tokens"""
        prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
        prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
        prefix_mask = prefix_toks.ne(self.pad)
        lprobs[prefix_mask] = torch.tensor(-math.inf).to(lprobs)
        lprobs[prefix_mask] = lprobs[prefix_mask].scatter(
            -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
        )
        # if prefix includes eos, then we should make sure tokens and
        # scores are the same across all beams
        eos_mask = prefix_toks.eq(self.eos)
        if eos_mask.any():
            # validate that the first beam matches the prefix
            first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[
                         :, 0, 1 : step + 1
                         ]
            eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
            target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
            assert (first_beam == target_prefix).all()

            # copy tokens, scores and lprobs from the first beam to all beams
            tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim, beam_size)
            scores = self.replicate_first_beam(scores, eos_mask_batch_dim, beam_size)
            lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim, beam_size)
        return lprobs, tokens, scores

    def replicate_first_beam(self, tensor, mask, beam_size: int):
        tensor = tensor.view(-1, beam_size, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))

    def finalize_hypos(
            self,
            step: int,
            eos_idx,
            batch_idx,
            actions,
            selected_tokens,
            scores,
            finalized: List[List[Dict[str, Tensor]]],
            finished: List[bool],
            attn: Optional[Tensor],
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 0..{step + 2}
        num_writes = torch.sum(torch.eq(actions, 5), dim=1)
        actions_clone = actions.index_select(0, eos_idx)[
                       :, : step + 1
                       ]
        attn_clone = (
            attn.index_select(0, eos_idx)[:, :, : step + 1]
            if attn is not None
            else None
        )

        # compute scores per token position
        pos_scores = scores.index_select(0, eos_idx)[:, : step + 1]
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        newly_finished: List[int] = []

        # For every finished beam item
        for i, eosIndex in enumerate(eos_idx):
            idx = int(batch_idx[eosIndex])
            score = scores[i]
            # Cannot create dict for key type '(int, int)' in torchscript.
            # The workaround is to cast int to string

            tokens = selected_tokens[idx, : num_writes[eosIndex]]

            if self.match_source_len:
                score = torch.tensor(-math.inf).to(score)

            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if attn_clone is not None:
                # remove padding tokens from attn scores
                hypo_attn = attn_clone[i]
            else:
                hypo_attn = torch.empty(0)

            finalized[idx].append(
                {
                    "actions": actions_clone[i],
                    "tokens": tokens,
                    "score": score[-1],
                    "attention": hypo_attn,  # src_len x tgt_len
                    "alignment": torch.empty(0),
                    "positional_scores": pos_scores[i],
                }
            )
            finished[idx] = True
            newly_finished.append(idx)

        return newly_finished

    def is_finished(
            self,
            step: int,
            unfin_idx: int,
            max_len: int,
            finalized_sent_len: int,
            beam_size: int,
    ):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False

    def calculate_banned_tokens(
            self,
            tokens,
            step: int,
            gen_ngrams: List[Dict[str, List[int]]],
            no_repeat_ngram_size: int,
            bbsz_idx: int,
    ):
        tokens_list: List[int] = tokens[
                                 bbsz_idx, step + 2 - no_repeat_ngram_size : step + 1
                                 ].tolist()
        # before decoding the next token, prevent decoding of ngrams that have already appeared
        ngram_index = ",".join([str(x) for x in tokens_list])
        return gen_ngrams[bbsz_idx].get(ngram_index, torch.jit.annotate(List[int], []))

    def transpose_list(self, l: List[List[int]]):
        # GeneratorExp aren't supported in TS so ignoring the lint
        min_len = min([len(x) for x in l])  # noqa
        l2 = [[row[i] for row in l] for i in range(min_len)]
        return l2

    def _no_repeat_ngram(self, tokens, lprobs, bsz: int, beam_size: int, step: int):
        # for each beam and batch sentence, generate a list of previous ngrams
        gen_ngrams: List[Dict[str, List[int]]] = [
            torch.jit.annotate(Dict[str, List[int]], {})
            for bbsz_idx in range(bsz * beam_size)
        ]
        cpu_tokens = tokens.cpu()
        for bbsz_idx in range(bsz * beam_size):
            gen_tokens: List[int] = cpu_tokens[bbsz_idx].tolist()
            for ngram in self.transpose_list(
                    [gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]
            ):
                key = ",".join([str(x) for x in ngram[:-1]])
                gen_ngrams[bbsz_idx][key] = gen_ngrams[bbsz_idx].get(
                    key, torch.jit.annotate(List[int], [])
                ) + [ngram[-1]]

        if step + 2 - self.no_repeat_ngram_size >= 0:
            # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            banned_tokens = [
                self.calculate_banned_tokens(
                    tokens, step, gen_ngrams, self.no_repeat_ngram_size, bbsz_idx
                )
                for bbsz_idx in range(bsz * beam_size)
            ]
        else:
            banned_tokens = [
                torch.jit.annotate(List[int], []) for bbsz_idx in range(bsz * beam_size)
            ]
        for bbsz_idx in range(bsz * beam_size):
            lprobs[bbsz_idx][
                torch.tensor(banned_tokens[bbsz_idx]).long()
            ] = torch.tensor(-math.inf).to(lprobs)
        return lprobs


class EnsembleModel(nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models_size = len(models)
        # method '__len__' is not supported in ModuleList for torch script
        self.single_model = models[0]
        self.models = nn.ModuleList(models)

        self.has_incremental: bool = False
        if all(
                hasattr(m, "decoder") and isinstance(m.decoder, FairseqIncrementalDecoder)
                for m in models
        ):
            self.has_incremental = True

    def forward(self, net_input: Dict[str, Tensor]):
        return [model(**net_input) for model in self.models]

    def has_encoder(self):
        return hasattr(self.single_model, "encoder")

    def has_incremental_states(self):
        return self.has_incremental

    def max_decoder_positions(self):
        return min([m.max_decoder_positions() for m in self.models])

    @torch.jit.export
    def forward_encoder(self, net_input: Dict[str, Tensor]):
        if not self.has_encoder():
            return None
        return [model.encoder.forward_torchscript(net_input) for model in self.models]

    @torch.jit.export
    def forward_agent_decoder(
            self,
            tokens,
            prev_output_tokens,
            incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
            temperature: float = 1.0,
            has_incremental=True,
    ):
        log_probs = []
        self.has_incremental = has_incremental
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        for i, model in enumerate(self.models):
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.forward(
                    tokens,
                    prev_output_tokens,
                    incremental_state=incremental_states[i],
                )
                decoder_out_tuple = (
                    decoder_out[0][:, -1:, :].div_(temperature),
                    None if len(decoder_out) <= 1 else decoder_out[1],
                )
            else:
                decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)
                decoder_out_tuple = (
                    decoder_out[0][:, :, :].div_(temperature),
                    None if len(decoder_out) <= 1 else decoder_out[1],
                )

            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None and self.has_incremental_states():
                    attn = attn[:, -1, :]


            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )

            if self.has_incremental_states():
                probs = probs[:, -1, :]

            if self.models_size == 1:
                return probs, attn

            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )

        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn

    @torch.jit.export
    def forward_decoder(
            self,
            tokens,
            encoder_outs: List[Dict[str, List[Tensor]]],
            incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
            temperature: float = 1.0,
            has_incremental=True,
    ):
        log_probs = []
        self.has_incremental = has_incremental
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                    incremental_state=incremental_states[i],
                )
                decoder_out_tuple = (
                    decoder_out[0][:, -1:, :].div_(temperature),
                    None if len(decoder_out) <= 1 else decoder_out[1],
                )
            else:
                decoder_out = model.decoder.forward_train(tokens, encoder_out=encoder_out) \
                    if isinstance(model, waitk_transformer.WaitkTransformerModel) \
                    else model.decoder.forward(tokens, encoder_out=encoder_out)
                decoder_out_tuple = (
                    decoder_out[0][:, :, :].div_(temperature),
                    None if len(decoder_out) <= 1 else decoder_out[1],
                )

            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None and self.has_incremental_states():
                    attn = attn[:, -1, :]


            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )

            if self.has_incremental_states():
                probs = probs[:, -1, :]

            if self.models_size == 1:
                return probs, attn

            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )

        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn

    @torch.jit.export
    def forward_decoder_features_only(
            self,
            tokens,
            encoder_outs: List[Dict[str, List[Tensor]]],
            incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            has_incremental=True,
    ):
        self.has_incremental = has_incremental
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                    incremental_state=incremental_states[i],
                    features_only=True
                )
            else:
                decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out, features_only=True)

        return decoder_out[0]

    @torch.jit.export
    def reorder_encoder_out(self, encoder_outs: Optional[List[Dict[str, List[Tensor]]]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[Dict[str, List[Tensor]]] = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(
                model.encoder.reorder_encoder_out(encoder_outs[i], new_order)
            )
        return new_outs

    @torch.jit.export
    def reorder_tokens(self, encoder_outs: Optional[Tensor], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[Dict[str, List[Tensor]]] = []
        assert encoder_outs is not None
        new_outs.append(
            encoder_outs.index_select(0, new_order)
        )
        return new_outs

    def reorder_list(self, hypos, new_order):
        new_outs = [hypos[index] for index in new_order]
        return new_outs

    @torch.jit.export
    def reorder_incremental_state(
            self,
            incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
            new_order,
    ):
        if not self.has_incremental_states():
            return
        for i, model in enumerate(self.models):
            model.decoder.reorder_incremental_state_scripting(
                incremental_states[i], new_order
            )
