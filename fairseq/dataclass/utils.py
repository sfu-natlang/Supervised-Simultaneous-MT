# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import inspect
import logging
import os
import re
from argparse import ArgumentError, ArgumentParser, Namespace
from dataclasses import _MISSING_TYPE, MISSING
from enum import Enum
from typing import Any, Dict, List, Tuple, Type

from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import FairseqConfig
from hydra.core.global_hydra import GlobalHydra
from hydra.experimental import compose, initialize
from omegaconf import DictConfig, OmegaConf, open_dict

logger = logging.getLogger(__name__)


def eval_str_list(x, x_type=float):
    if x is None:
        return None
    if isinstance(x, str):
        if len(x) == 0:
            return []
        x = ast.literal_eval(x)
    try:
        return list(map(x_type, x))
    except TypeError:
        return [x_type(x)]


def interpret_dc_type(field_type):
    if isinstance(field_type, str):
        raise RuntimeError("field should be a type")

    if field_type == Any:
        return str

    typestring = str(field_type)
    if re.match(r"(typing.|^)Union\[(.*), NoneType\]$", typestring):
        return field_type.__args__[0]
    return field_type


def gen_parser_from_dataclass(
        parser: ArgumentParser,
        dataclass_instance: FairseqDataclass,
        delete_default: bool = False,
) -> None:
    """convert a dataclass instance to tailing parser arguments"""

    def argparse_name(name: str):
        if name == "data":
            # normally data is positional args
            return name
        if name == "_name":
            # private member, skip
            return None
        return "--" + name.replace("_", "-")

    def get_kwargs_from_dc(
            dataclass_instance: FairseqDataclass, k: str
    ) -> Dict[str, Any]:
        """k: dataclass attributes"""

        kwargs = {}

        field_type = dataclass_instance._get_type(k)
        inter_type = interpret_dc_type(field_type)

        field_default = dataclass_instance._get_default(k)

        if isinstance(inter_type, type) and issubclass(inter_type, Enum):
            field_choices = [t.value for t in list(inter_type)]
        else:
            field_choices = None

        field_help = dataclass_instance._get_help(k)
        field_const = dataclass_instance._get_argparse_const(k)

        if isinstance(field_default, str) and field_default.startswith("${"):
            kwargs["default"] = field_default
        else:
            if field_default is MISSING:
                kwargs["required"] = True
            if field_choices is not None:
                kwargs["choices"] = field_choices
            if (
                    isinstance(inter_type, type)
                    and (issubclass(inter_type, List) or issubclass(inter_type, Tuple))
            ) or ("List" in str(inter_type) or "Tuple" in str(inter_type)):
                if "int" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, int)
                elif "float" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, float)
                elif "str" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, str)
                else:
                    raise NotImplementedError(
                        "parsing of type " + str(inter_type) + " is not implemented"
                    )
                if field_default is not MISSING:
                    kwargs["default"] = (
                        ",".join(map(str, field_default))
                        if field_default is not None
                        else None
                    )
            elif (
                    isinstance(inter_type, type) and issubclass(inter_type, Enum)
            ) or "Enum" in str(inter_type):
                kwargs["type"] = str
                if field_default is not MISSING:
                    if isinstance(field_default, Enum):
                        kwargs["default"] = field_default.value
                    else:
                        kwargs["default"] = field_default
            elif inter_type is bool:
                kwargs["action"] = (
                    "store_false" if field_default is True else "store_true"
                )
                kwargs["default"] = field_default
            else:
                kwargs["type"] = inter_type
                if field_default is not MISSING:
                    kwargs["default"] = field_default

        kwargs["help"] = field_help
        if field_const is not None:
            kwargs["const"] = field_const
            kwargs["nargs"] = "?"

        return kwargs

    for k in dataclass_instance._get_all_attributes():
        field_name = argparse_name(dataclass_instance._get_name(k))
        field_type = dataclass_instance._get_type(k)
        if field_name is None:
            continue
        elif inspect.isclass(field_type) and issubclass(field_type, FairseqDataclass):
            gen_parser_from_dataclass(parser, field_type(), delete_default)
            continue

        kwargs = get_kwargs_from_dc(dataclass_instance, k)

        field_args = [field_name]
        alias = dataclass_instance._get_argparse_alias(k)
        if alias is not None:
            field_args.append(alias)

        if "default" in kwargs:
            if isinstance(kwargs["default"], str) and kwargs["default"].startswith(
                    "${"
            ):
                if kwargs["help"] is None:
                    # this is a field with a name that will be added elsewhere
                    continue
                else:
                    del kwargs["default"]
            if delete_default:
                del kwargs["default"]
        try:
            parser.add_argument(*field_args, **kwargs)
        except ArgumentError:
            pass


def _set_legacy_defaults(args, cls):
    """Helper to set default arguments based on *add_args*."""
    if not hasattr(cls, "add_args"):
        return

    import argparse

    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS, allow_abbrev=False
    )
    cls.add_args(parser)
    # copied from argparse.py:
    defaults = argparse.Namespace()
    for action in parser._actions:
        if action.dest is not argparse.SUPPRESS:
            if not hasattr(defaults, action.dest):
                if action.default is not argparse.SUPPRESS:
                    setattr(defaults, action.dest, action.default)
    for key, default_value in vars(defaults).items():
        if not hasattr(args, key):
            setattr(args, key, default_value)


def _override_attr(
        sub_node: str, data_class: Type[FairseqDataclass], args: Namespace
) -> List[str]:
    overrides = []

    if not inspect.isclass(data_class) or not issubclass(data_class, FairseqDataclass):
        return overrides

    def get_default(f):
        if not isinstance(f.default_factory, _MISSING_TYPE):
            return f.default_factory()
        return f.default

    for k, v in data_class.__dataclass_fields__.items():
        if k.startswith("_"):
            # private member, skip
            continue

        val = get_default(v) if not hasattr(args, k) else getattr(args, k)

        field_type = interpret_dc_type(v.type)
        if (
                isinstance(val, str)
                and not val.startswith("${")  # not interpolation
                and field_type != str
                and (
                not inspect.isclass(field_type) or not issubclass(field_type, Enum)
        )  # not choices enum
        ):
            # upgrade old models that stored complex parameters as string
            val = ast.literal_eval(val)

        if isinstance(val, tuple):
            val = list(val)

        v_type = getattr(v.type, "__origin__", None)
        if (
                (v_type is List or v_type is list)
                # skip interpolation
                and not (isinstance(val, str) and val.startswith("${"))
        ):
            # if type is int but val is float, then we will crash later - try to convert here
            t_args = v.type.__args__
            if len(t_args) == 1:
                val = list(map(t_args[0], val))
        elif val is not None and (field_type is int or field_type is bool or field_type is float):
            try:
                val = field_type(val)
            except:
                pass  # ignore errors here, they are often from interpolation args

        if val is None:
            overrides.append("{}.{}=null".format(sub_node, k))
        elif val == "":
            overrides.append("{}.{}=''".format(sub_node, k))
        elif isinstance(val, str):
            val = val.replace("'", r"\'")
            overrides.append("{}.{}='{}'".format(sub_node, k, val))
        elif isinstance(val, FairseqDataclass):
            overrides += _override_attr(f"{sub_node}.{k}", type(val), args)
        elif isinstance(val, Namespace):
            sub_overrides, _ = override_module_args(val)
            for so in sub_overrides:
                overrides.append(f"{sub_node}.{k}.{so}")
        else:
            overrides.append("{}.{}={}".format(sub_node, k, val))

    return overrides


def migrate_registry(
        name, value, registry, args, overrides, deletes, use_name_as_val=False
):
    if value in registry:
        overrides.append("{}={}".format(name, value))
        overrides.append("{}._name={}".format(name, value))
        overrides.extend(_override_attr(name, registry[value], args))
    elif use_name_as_val and value is not None:
        overrides.append("{}={}".format(name, value))
    else:
        deletes.append(name)


def override_module_args(args: Namespace) -> Tuple[List[str], List[str]]:
    """use the field in args to overrides those in cfg"""
    overrides = []
    deletes = []

    for k in FairseqConfig.__dataclass_fields__.keys():
        overrides.extend(
            _override_attr(k, FairseqConfig.__dataclass_fields__[k].type, args)
        )

    if args is not None:
        if hasattr(args, "task"):
            from fairseq.tasks import TASK_DATACLASS_REGISTRY

            migrate_registry(
                "task", args.task, TASK_DATACLASS_REGISTRY, args, overrides, deletes
            )
        else:
            deletes.append("task")

        # these options will be set to "None" if they have not yet been migrated
        # so we can populate them with the entire flat args
        CORE_REGISTRIES = {"criterion", "optimizer", "lr_scheduler"}

        from fairseq.registry import REGISTRIES

        for k, v in REGISTRIES.items():
            if hasattr(args, k):
                migrate_registry(
                    k,
                    getattr(args, k),
                    v["dataclass_registry"],
                    args,
                    overrides,
                    deletes,
                    use_name_as_val=k not in CORE_REGISTRIES,
                )
            else:
                deletes.append(k)

        no_dc = True
        if hasattr(args, "arch"):
            from fairseq.models import ARCH_MODEL_REGISTRY, ARCH_MODEL_NAME_REGISTRY

            if args.arch in ARCH_MODEL_REGISTRY:
                m_cls = ARCH_MODEL_REGISTRY[args.arch]
                dc = getattr(m_cls, "__dataclass", None)
                if dc is not None:
                    m_name = ARCH_MODEL_NAME_REGISTRY[args.arch]
                    overrides.append("model={}".format(m_name))
                    overrides.append("model._name={}".format(args.arch))
                    # override model params with those exist in args
                    overrides.extend(_override_attr("model", dc, args))
                    no_dc = False
        if no_dc:
            deletes.append("model")

    return overrides, deletes


'''
def extend_override_module(overrides):
    extension = {
    nmt_task._name:'Supervised_simultaneous_translation',
    all_gather_list_size=16384,
    azureml_logging=False,
    batch_size=None,
    batch_size_valid=None,
    beam=2,
    best_checkpoint_metric='loss',
    bf16=False,
    bpe=None,
    broadcast_buffers=False,
    bucket_cap_mb=25,
    checkpoint_shard_count=1,
    checkpoint_suffix='',
    constraints=None,
    cpu=False,
    criterion='cross_entropy',
    curriculum=0,
    data='/Users/alinejad/Desktop/SFU/Research/Speech-to-text-transation/Supervised-fairseq/data-bin/wmt14_en_de/bin-de_en-encoded',
    data_buffer_size=10,
    dataset_impl=None,
    ddp_backend='c10d',
    decoding_format=None,
    disable_validation=False,
    distributed_backend='nccl',
    distributed_init_method=None,
    distributed_no_spawn=False,
    distributed_port=-1, distributed_rank=0, distributed_world_size=1, distributed_wrapper='DDP', diverse_beam_groups=-1, diverse_beam_strength=0.5, diversity_rate=-1.0, empty_cache_freq=0, eos=2, eval_bleu=False, eval_bleu_args=None, eval_bleu_detok='space', eval_bleu_detok_args=None, eval_bleu_print_samples=False, eval_bleu_remove_bpe=None, eval_tokenized_bleu=False, fast_stat_sync=False, find_unused_parameters=False, finetune_from_model=None, fix_batches_to_gpus=False, fixed_validation_seed=None, force_anneal=None, fp16=False, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, gen_subset='valid', iter_decode_eos_penalty=0.0, iter_decode_force_max_iter=False, iter_decode_max_iter=10, iter_decode_with_beam=1, iter_decode_with_external_reranker=False, keep_best_checkpoints=-1, keep_interval_updates=-1, keep_last_epochs=-1, left_pad_source='False', left_pad_target='False', lenpen=1, lm_path=None, lm_weight=0.0, load_alignments=False, load_checkpoint_on_all_dp_ranks=False, localsgd_frequency=3, log_format=None, log_interval=100, lr_scheduler='fixed', lr_shrink=0.1, match_source_len=False, max_len_a=0, max_len_b=200, max_source_positions=1024, max_target_positions=1024, max_tokens=8000, max_tokens_valid=8000, maximize_best_checkpoint_metric=False, memory_efficient_bf16=False, memory_efficient_fp16=False, min_len=1, min_loss_scale=0.0001, model_overrides='{}', model_parallel_size=1, nbest=1, no_beamable_mm=False, no_early_stop=False, no_epoch_checkpoints=False, no_last_checkpoints=False, no_progress_bar=False, no_repeat_ngram_size=0, no_save=False, no_save_optimizer_state=False, no_seed_provided=False, nprocs_per_node=1, num_batch_buckets=0, num_shards=1, num_workers=1, optimizer=None, optimizer_overrides='{}', pad=1, path='/Users/alinejad/Desktop/SFU/Research/Speech-to-text-transation/Supervised-fairseq/nmt_trans_wmt14_deen_med/checkpoint_best.pt', patience=-1, pipeline_balance=None, pipeline_checkpoint='never', pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None, pipeline_model_parallel=False, post_process=None, prefix_size=0, print_alignment=None, print_step=False, profile=False, quantization_config_path=None, quiet=False, replace_unk=None, required_batch_size_multiple=8, required_seq_len_multiple=1, reset_dataloader=False, reset_logging=True, reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False, restore_file='checkpoint_last.pt', results_path=None, retain_dropout=False, retain_dropout_modules=None, retain_iter_history=False, sacrebleu=False, sampling=False, sampling_topk=-1, sampling_topp=-1.0, save_dir='checkpoints', save_interval=1, save_interval_updates=0, score_reference=False, scoring='bleu', seed=1, shard_id=0, skip_invalid_size_inputs_valid_test=True, slowmo_algorithm='LocalSGD', slowmo_momentum=None, source_lang='de', target_lang='en', task='Supervised_simultaneous_translation', temperature=1.0, tensorboard_logdir=None, threshold_loss_scale=None, tokenizer=None, tpu=False, train_subset='train', truncate_source=False, unk=3, unkpen=0, unnormalized=False, upsample_primary=1, user_dir='../examples/Supervised_simul_MT', valid_subset='valid', validate_after_updates=0, validate_interval=1, validate_interval_updates=0, wandb_project=None, warmup_updates=0, zero_sharding='none')
    }
'''


def convert_namespace_to_omegaconf(args: Namespace) -> DictConfig:
    """Convert a flat argparse.Namespace to a structured DictConfig."""

    # Here we are using field values provided in args to override counterparts inside config object
    overrides, deletes = override_module_args(args)

    # configs will be in fairseq/config after installation
    config_path = os.path.join("..", "config")

    GlobalHydra.instance().clear()
    with initialize(config_path=config_path):
        try:
            composed_cfg = compose("config", overrides=overrides, strict=False)
        except:
            logger.error("Error when composing. Overrides: " + str(overrides))
            raise
        for k in deletes:
            composed_cfg[k] = None

    cfg = OmegaConf.create(
        OmegaConf.to_container(composed_cfg, resolve=True, enum_to_str=True)
    )

    # hack to be able to set Namespace in dict config. this should be removed when we update to newer
    # omegaconf version that supports object flags, or when we migrate all existing models
    from omegaconf import _utils

    old_primitive = _utils.is_primitive_type
    _utils.is_primitive_type = lambda _: True

    if cfg.task is None and getattr(args, "task", None):
        cfg.task = Namespace(**vars(args))
        from fairseq.tasks import TASK_REGISTRY

        _set_legacy_defaults(cfg.task, TASK_REGISTRY[args.task])
        cfg.task._name = args.task
    if cfg.model is None and getattr(args, "arch", None):
        cfg.model = Namespace(**vars(args))
        from fairseq.models import ARCH_MODEL_REGISTRY

        _set_legacy_defaults(cfg.model, ARCH_MODEL_REGISTRY[args.arch])
        cfg.model._name = args.arch
    if cfg.optimizer is None and getattr(args, "optimizer", None):
        cfg.optimizer = Namespace(**vars(args))
        from fairseq.optim import OPTIMIZER_REGISTRY

        _set_legacy_defaults(cfg.optimizer, OPTIMIZER_REGISTRY[args.optimizer])
        cfg.optimizer._name = args.optimizer
    if cfg.lr_scheduler is None and getattr(args, "lr_scheduler", None):
        cfg.lr_scheduler = Namespace(**vars(args))
        from fairseq.optim.lr_scheduler import LR_SCHEDULER_REGISTRY

        _set_legacy_defaults(cfg.lr_scheduler, LR_SCHEDULER_REGISTRY[args.lr_scheduler])
        cfg.lr_scheduler._name = args.lr_scheduler
    if cfg.criterion is None and getattr(args, "criterion", None):
        cfg.criterion = Namespace(**vars(args))
        from fairseq.criterions import CRITERION_REGISTRY

        _set_legacy_defaults(cfg.criterion, CRITERION_REGISTRY[args.criterion])
        cfg.criterion._name = args.criterion

    _utils.is_primitive_type = old_primitive
    OmegaConf.set_struct(cfg, True)
    return cfg


def populate_dataclass(
        dataclass: FairseqDataclass,
        args: Namespace,
) -> FairseqDataclass:
    for k in dataclass.__dataclass_fields__.keys():
        if k.startswith("_"):
            # private member, skip
            continue
        if hasattr(args, k):
            setattr(dataclass, k, getattr(args, k))

    return dataclass


def overwrite_args_by_name(cfg: DictConfig, overrides: Dict[str, any]):
    # this will be deprecated when we get rid of argparse and model_overrides logic

    from fairseq.registry import REGISTRIES

    with open_dict(cfg):
        for k in cfg.keys():
            # "k in cfg" will return false if its a "mandatory value (e.g. ???)"
            if k in cfg and isinstance(cfg[k], DictConfig):
                overwrite_args_by_name(cfg[k], overrides)
            elif k in cfg and isinstance(cfg[k], Namespace):
                for override_key, val in overrides.items():
                    setattr(cfg[k], override_key, val)
            elif k in overrides:
                if (
                        k in REGISTRIES
                        and overrides[k] in REGISTRIES[k]["dataclass_registry"]
                ):
                    cfg[k] = DictConfig(
                        REGISTRIES[k]["dataclass_registry"][overrides[k]]
                    )
                    overwrite_args_by_name(cfg[k], overrides)
                    cfg[k]._name = overrides[k]
                else:
                    cfg[k] = overrides[k]


def merge_with_parent(dc: FairseqDataclass, cfg: FairseqDataclass):
    merged_cfg = OmegaConf.merge(dc, cfg)
    merged_cfg.__dict__["_parent"] = cfg.__dict__["_parent"]
    OmegaConf.set_struct(merged_cfg, True)
    return merged_cfg
