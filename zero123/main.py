import argparse
import datetime
import glob
import os
import sys

import lovely_tensors as lt
import torch

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from parser_helpers import get_parser, nondefault_trainer_args
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from utils import rank_zero_print


MULTINODE_HACKS = False


if __name__ == "__main__":

    lt.monkey_patch()

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    # try:

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())

    # default to ddp
    trainer_config["accelerator"] = "ddp"
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        gpuinfo = trainer_config["gpus"]
        rank_zero_print(f"Running on GPUs {gpuinfo}")
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    model = instantiate_from_config(config.model)
    model.cpu()

    if opt.finetune_from != "":
        rank_zero_print(f"Attempting to load state from {opt.finetune_from}")
        old_state = torch.load(opt.finetune_from, map_location="cpu")

        if "state_dict" in old_state:
            rank_zero_print(f"Found nested key 'state_dict' in checkpoint, loading this instead")
            old_state = old_state["state_dict"]

        # Check if we need to port weights from 4ch input to 8ch
        in_filters_load = old_state["model.diffusion_model.input_blocks.0.0.weight"]
        new_state = model.state_dict()
        in_filters_current = new_state["model.diffusion_model.input_blocks.0.0.weight"]
        in_shape = in_filters_current.shape
        if in_shape != in_filters_load.shape:
            input_keys = [
                "model.diffusion_model.input_blocks.0.0.weight",
                "model_ema.diffusion_modelinput_blocks00weight",
            ]

            for input_key in input_keys:
                if input_key not in old_state or input_key not in new_state:
                    continue
                input_weight = new_state[input_key]
                if input_weight.size() != old_state[input_key].size():
                    print(f"Manual init: {input_key}")
                    input_weight.zero_()
                    input_weight[:, :4, :, :].copy_(old_state[input_key])
                    old_state[input_key] = torch.nn.parameter.Parameter(input_weight)

        m, u = model.load_state_dict(old_state, strict=False)

        if len(m) > 0:
            rank_zero_print("missing keys:")
            rank_zero_print(m)
        if len(u) > 0:
            rank_zero_print("unexpected keys:")
            rank_zero_print(u)

    # trainer and callbacks
    trainer_kwargs = dict()

    # default logger configs
    default_logger_cfgs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "name": nowname,
                "save_dir": logdir,
                "offline": opt.debug,
                "id": nowname,
            },
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.TestTubeLogger",
            "params": {
                "name": "testtube",
                "save_dir": logdir,
            },
        },
    }
    default_logger_cfg = default_logger_cfgs["testtube"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:06}-{step:09}",
            "verbose": True,
            "save_last": True,
            "save_top_k": -1,
        },
    }

    # the val/loss_simple_ema is not in the metric dict so just save all
    # if hasattr(model, "monitor"):
    #     rank_zero_print(f"Monitoring {model.monitor} as checkpoint metric.")
    #     default_modelckpt_cfg["params"]["monitor"] = model.monitor
    #     default_modelckpt_cfg["params"]["save_top_k"] = -1

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)

    # if version.parse(pl.__version__) < version.parse("1.4.0"):
    #     trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "custom_callbacks.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
                "debug": opt.debug,
            },
        },
    }
    # if version.parse(pl.__version__) >= version.parse("1.4.0"):
    default_callbacks_cfg.update({"checkpoint_callback": modelckpt_cfg})

    if "callbacks" in lightning_config:
        ln_callbacks_cfg = lightning_config.callbacks
    else:
        ln_callbacks_cfg = OmegaConf.create()

    if (
        "metrics_over_trainsteps_checkpoint" in lightning_config
        and lightning_config.metrics_over_trainsteps_checkpoint
    ):
        rank_zero_print(
            "Caution: Saving checkpoints every n train steps without deleting. This might require some free space."
        )
        default_metrics_over_trainsteps_ckpt_dict = {
            "metrics_over_trainsteps_checkpoint": {
                "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                "params": {
                    "dirpath": os.path.join(ckptdir, "trainstep_checkpoints"),
                    "filename": "{epoch:06}-{step:09}",
                    "verbose": True,
                    "save_top_k": -1,
                    "every_n_train_steps": 300,
                    "save_weights_only": True,
                },
            }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, ln_callbacks_cfg)
    if "ignore_keys_callback" in callbacks_cfg and hasattr(trainer_opt, "resume_from_checkpoint"):
        callbacks_cfg.ignore_keys_callback.params["ckpt_path"] = trainer_opt.resume_from_checkpoint
    elif "ignore_keys_callback" in callbacks_cfg:
        del callbacks_cfg["ignore_keys_callback"]

    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    if not "plugins" in trainer_kwargs:
        trainer_kwargs["plugins"] = list()
    if not lightning_config.get("find_unused_parameters", True):
        from pytorch_lightning.plugins import DDPPlugin

        trainer_kwargs["plugins"].append(DDPPlugin(find_unused_parameters=False))

    if MULTINODE_HACKS:
        # disable resume from hpc ckpts
        # NOTE below only works in later versions
        # from pytorch_lightning.plugins.environments import SLURMEnvironment
        # trainer_kwargs["plugins"].append(SLURMEnvironment(auto_requeue=False))
        # hence we monkey patch things
        from pytorch_lightning.trainer.connectors.checkpoint_connector import (
            CheckpointConnector,
        )

        setattr(CheckpointConnector, "hpc_resume_path", None)

    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir

    # data
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    rank_zero_print("#### Data ####")
    try:
        for k in data.datasets:
            rank_zero_print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
    except:
        rank_zero_print("datasets not yet initialized.")
    rank_zero_print("##############")

    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    if not cpu:
        ngpu = len(lightning_config.trainer.gpus.strip(",").split(","))
    else:
        ngpu = 1
    if "accumulate_grad_batches" in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    rank_zero_print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        rank_zero_print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr
            )
        )
    else:
        model.learning_rate = base_lr
        rank_zero_print("++++ NOT USING LR SCALING ++++")
        rank_zero_print(f"Setting learning rate to {model.learning_rate:.2e}")

    # # allow checkpointing via USR1
    # def melk(*args, **kwargs):
    #     # run all checkpoint hooks
    #     if trainer.global_rank == 0:
    #         rank_zero_print("Summoning checkpoint.")
    #         ckpt_path = os.path.join(ckptdir, "last.ckpt")
    #         trainer.save_checkpoint(ckpt_path)

    # def divein(*args, **kwargs):
    #     if trainer.global_rank == 0:
    #         import pudb

    #         pudb.set_trace()

    # import signal

    # signal.signal(signal.SIGUSR1, melk)
    # signal.signal(signal.SIGUSR2, divein)

    # run
    if opt.train:
        trainer.fit(model, data)
        # try:
        #     trainer.fit(model, data)
        # except Exception:
        #     if not opt.debug:
        #         melk()
        #     raise
    if not opt.no_test and not trainer.interrupted:
        trainer.test(model, data)

    # except RuntimeError as err:
    #     if MULTINODE_HACKS:
    #         import datetime
    #         import os
    #         import socket

    #         import requests

    #         device = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    #         hostname = socket.gethostname()
    #         ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    #         resp = requests.get("http://169.254.169.254/latest/meta-data/instance-id")
    #         rank_zero_print(
    #             f"ERROR at {ts} on {hostname}/{resp.text} (CUDA_VISIBLE_DEVICES={device}): {type(err).__name__}: {err}",
    #             flush=True,
    #         )
    #     raise err

    # except Exception:
    #     if opt.debug and trainer.global_rank == 0:
    #         try:
    #             import pudb as debugger
    #         except ImportError:
    #             import pdb as debugger
    #         debugger.post_mortem()
    #     raise

    # finally:
    #     # move newly created debug project to debug_runs
    #     if opt.debug and not opt.resume and trainer.global_rank == 0:
    #         dst, name = os.path.split(logdir)
    #         dst = os.path.join(dst, "debug_runs", name)
    #         os.makedirs(os.path.split(dst)[0], exist_ok=True)
    #         os.rename(logdir, dst)
    #     if trainer.global_rank == 0:
    #         rank_zero_print(trainer.profiler.summary())
