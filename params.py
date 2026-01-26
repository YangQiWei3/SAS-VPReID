import os
import time
import math
import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn


# ---------------------------
# Utils: Params, GPU memory
# ---------------------------

def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def bytes_to_mib(x: int) -> float:
    return x / (1024 ** 2)

def cuda_mem_stats(device="cuda"):
    assert torch.cuda.is_available()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)
    return {
        "allocated_MiB": bytes_to_mib(allocated),
        "reserved_MiB": bytes_to_mib(reserved),
        "max_allocated_MiB": bytes_to_mib(max_allocated),
        "max_reserved_MiB": bytes_to_mib(max_reserved),
    }


# ---------------------------
# FLOPs calculators (fallback)
# ---------------------------

def try_profile_flops(model, inputs):
    """
    Returns: (flops, params) where flops is number of floating point operations (approx).
    This is *per forward pass* for given input shape.
    It tries: thop -> fvcore -> ptflops.
    """
    model.eval()

    # 1) thop
    try:
        from thop import profile  # type: ignore
        with torch.no_grad():
            flops, params = profile(model, inputs=inputs, verbose=False)
        return flops, params
    except Exception:
        pass

    # 2) fvcore
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count  # type: ignore
        with torch.no_grad():
            flops = FlopCountAnalysis(model, inputs).total()
        params = sum(parameter_count(model).values())
        return flops, params
    except Exception:
        pass

    # 3) ptflops
    try:
        from ptflops import get_model_complexity_info  # type: ignore

        # ptflops expects input_res without batch dim, and a callable model
        # We only support single-tensor input for this fallback.
        if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("ptflops fallback supports only single tensor input.")
        x = inputs[0]
        input_res = tuple(x.shape[1:])

        def _forward_dummy(m):
            # ptflops calls model(*inputs) internally; it uses input_res
            return m

        macs, params = get_model_complexity_info(
            _forward_dummy(model),
            input_res,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        # ptflops returns MACs often; FLOPs ≈ 2*MACs for many conv/linear (not always).
        flops = 2 * macs
        return flops, params
    except Exception:
        pass

    return None, None


# ---------------------------
# Benchmark: latency & throughput
# ---------------------------

def benchmark_inference(
    model: nn.Module,
    inputs,
    device="cuda",
    iters=200,
    warmup=50,
    amp_dtype="fp16",  # fp32/fp16/bf16
    use_compile=False,
):
    assert isinstance(inputs, (list, tuple)) and len(inputs) >= 1
    model.eval()

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but device=cuda requested.")

    model = model.to(device)
    inputs = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs)

    # AMP context
    if device.startswith("cuda"):
        if amp_dtype == "fp16":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        elif amp_dtype == "bf16":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        elif amp_dtype == "fp32":
            autocast_ctx = nullcontext()
        else:
            raise ValueError("amp_dtype must be one of: fp32, fp16, bf16")
    else:
        autocast_ctx = nullcontext()

    # torch.compile (optional)
    if use_compile:
        # Note: compile can change performance characteristics and may take time on first run.
        model = torch.compile(model)

    # Reset GPU memory peaks
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            with autocast_ctx:
                _ = model(*inputs)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # Timed runs
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            with autocast_ctx:
                _ = model(*inputs)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_s = t1 - t0
    avg_ms = (total_s / iters) * 1000.0
    throughput = iters / total_s  # iterations per second

    mem = None
    if device.startswith("cuda"):
        mem = cuda_mem_stats(device)

    return {
        "avg_latency_ms": avg_ms,
        "throughput_iter_per_s": throughput,
        "total_time_s": total_s,
        "mem": mem,
    }


# ---------------------------
# Example: plug your model here
# ---------------------------

def build_dummy_model():
    # Replace with your model, e.g.:
    # from mypkg import MyModel
    # model = MyModel(...)
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, 1000),
    )

def build_dummy_inputs(batch_size=1, device="cuda"):
    # Replace with your real input shapes & types.
    x = torch.randn(batch_size, 3, 8, 256, 128, device=device)
    return (x,)


import os

if os.getenv("PYCHARM_IP", "0") != "0":
    import pydevd_pycharm

    pydevd_pycharm.settrace(host=os.getenv("PYCHARM_IP"), port=12321, stdout_to_server=True, stderr_to_server=True)
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import os.path as osp
import sys
import datetime

import scipy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import argparse
from config import cfg

from utils.logger import setup_logger
# from datasets.make_dataloader_clipreid import make_dataloader
from datasets.make_video_dataloader import make_dataloader, make_CLIMB_dataloader
from model.make_model_clipreid import make_model
from loss.make_loss import make_loss
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
# from processor.processor_clipreid_stage1 import do_train_stage1
from processor.processor_clipreid_stage2 import do_train_stage2


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():

    #############################################
    # --> 加载参数和初始化
    #############################################
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file", default="configs/vit_clipreid.yml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--amp", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no_flops", action="store_true", help="skip FLOPs profiling")
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("TFCLIP", output_dir, cfg.MODEL.NAME, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    #############################################
    # --> 数据加载
    #############################################
    # train_loader_stage2, train_loader_stage1, query_loader, gallery_loader, num_query, num_classes, camera_num, view_num = make_CLIMB_dataloader(cfg)

    model = make_model(cfg, num_class=215, camera_num=38, view_num=None).to(args.device)
    inputs = build_dummy_inputs(batch_size=args.batch, device=args.device)

    # Params
    total_params, trainable_params = count_params(model)

    # FLOPs (per forward)
    flops = None
    flops_params = None
    if not args.no_flops:
        # Ensure model and inputs are on the same device for FLOPs profiling
        model_device = next(model.parameters()).device
        inputs = tuple(x.to(model_device) if isinstance(x, torch.Tensor) else x for x in inputs)
        prof_flops, prof_params = try_profile_flops(model, inputs)
        flops = prof_flops
        flops_params = prof_params  # profiler's idea of params (may include buffers)

    # Time + memory
    bench = benchmark_inference(
        model=model,
        inputs=inputs,
        device=args.device,
        iters=args.iters,
        warmup=args.warmup,
        amp_dtype=args.amp,
        use_compile=args.compile,
    )

    # Report
    print("==== Benchmark Result ====")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch}")
    print(f"AMP: {args.amp} | torch.compile: {args.compile}")
    print("")
    print(f"Params (total):     {total_params:,}")
    print(f"Params (trainable): {trainable_params:,}")

    if flops is not None:
        print(f"FLOPs / forward:    {flops:,.0f}")
        if flops_params is not None:
            print(f"Params (profiler):  {int(flops_params):,}")

    print("")
    print(f"Avg latency:        {bench['avg_latency_ms']:.3f} ms / iter")
    print(f"Throughput:         {bench['throughput_iter_per_s']:.2f} iter/s")
    if bench["mem"] is not None:
        m = bench["mem"]
        print("")
        print("CUDA memory (MiB):")
        print(f"  allocated:        {m['allocated_MiB']:.2f}")
        print(f"  reserved:         {m['reserved_MiB']:.2f}")
        print(f"  max_allocated:    {m['max_allocated_MiB']:.2f}")
        print(f"  max_reserved:     {m['max_reserved_MiB']:.2f}")

    print("==========================")


if __name__ == "__main__":
    main()
