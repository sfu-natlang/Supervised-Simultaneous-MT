# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq import checkpoint_utils, utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from simuleval.agents import TextAgent
from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
from examples.Supervised_simul_MT.action_generator import ActionGenerator, SimultaneousGenerator

class SupervisedTextAgent(TextAgent):

    data_type = "text"

    def __init__(self, args):
        super().__init__(args)
        self.waitk = args.waitk
        self.nmt_path = args.nmt_path
        self.agt_path = args.agent_path
        self.data = args.data_path
        self.beam_size = args.nmt_beam
        self.use_cuda = False
        self.has_target = True

        args.task = 'Supervised_simultaneous_translation'

        utils.import_user_module(args)
        cfg = convert_namespace_to_omegaconf(args)
        cfg.task.data = self.data
        self.task = tasks.setup_task(cfg.task)

        self.nmt_model, _ = checkpoint_utils.load_model_ensemble(
            utils.split_paths(self.nmt_path)
        )
        self.agt_model, _ = checkpoint_utils.load_model_ensemble(
            utils.split_paths(self.agt_path)
        )

        self.tgt_dict = self.task.target_dictionary
        self.src_dict = self.task.source_dictionary

        self.read_index = self.task.agent_dictionary.indices['0']
        self.write_index = self.task.agent_dictionary.indices['1']
        self.eos_index = self.task.agent_dictionary.eos_index

        for model in self.nmt_model:
            if model is None:
                continue
            if self.use_cuda:
                model.cuda()
            # model.prepare_for_inference_(cfg)

        if self.use_cuda:
            self.agt_model.cuda()

        self.translation_generator = ActionGenerator(
            self.nmt_model, self.tgt_dict, self.src_dict,
            beam_size=self.beam_size,
            has_target=self.has_target,
            all_features=True
        )
        self.policy_generator = SimultaneousGenerator(
            self.nmt_model, self.agt_model, self.task
        )

    @staticmethod
    def add_args(parser):
        # Add additional command line arguments here
        parser.add_argument("--waitk", type=int, default=3)
        parser.add_argument("--nmt-beam", type=int, default=5)
        parser.add_argument('--data-path', metavar="DATAPATH", default=None,
                            help='Path to the dataset')
        parser.add_argument('--nmt-path', metavar="NMTPATH", default=None,
                            help='Path to the fully trained NMT model')
        parser.add_argument('--agent-path', metavar="AGTPATH", default=None,
                            help='Path to the fully trained agent model')
        parser.add_argument('--user-dir', metavar="USRPATH", default=None,
                            help='Path to the the agent model')

    def policy(self, states):
        step = states.num_reads - 1 + states.num_writes
        if step < 0:
            # The first action is always a read action
            return READ_ACTION

        action_index, states.incremental_states = self.policy_generator._generate_simuleval(
            states.prior_inputs[:, :step+1], states.prior_outputs[:, :step+1], states.incremental_states
        )
        return READ_ACTION if int(action_index) == self.read_index and not states.finish_read() else WRITE_ACTION

    def predict(self, states):
        if states.num_writes >= len(states.partial_translations[-1]['tokens']):
            if states.partial_translations[-1]['tokens'][-1].tolist() == self.eos_index:
                return DEFAULT_EOS
            return self.tgt_dict.string([states.partial_translations[-1]['tokens'][-1].tolist()])
        elif states.partial_translations[-1]['tokens'][states.num_writes].tolist() == self.eos_index:
            return DEFAULT_EOS

        return self.tgt_dict.string([states.partial_translations[-1]['tokens'][states.num_writes].tolist()])

    def generate_translation(self, source):
        return self.translation_generator._generate(source)[0]
