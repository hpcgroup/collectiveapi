#!/usr/bin/env python3

import argparse
import json
import logging
import sys
import traceback

from logging import FileHandler
from mscclang2chakra_converter import MSCCL2ChakraConverter

def get_logger(log_filename: str) -> logging.Logger:
    formatter = logging.Formatter(
        "%(levelname)s [%(asctime)s] %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p")

    file_handler = FileHandler(log_filename, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execution Trace Converter")
    parser.add_argument(
        "--input_type",
        type=str,
        default=None,
        required=True,
        help="Input execution trace type")
    parser.add_argument(
        "--input_filename",
        type=str,
        default=None,
        required=True,
        help="Input execution trace filename")
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        required=True,
        help="Output Chakra execution trace filename")
    parser.add_argument(
        "--log_filename",
        type=str,
        default="debug.log",
        help="Log filename")
    parser.add_argument(
        "--coll_size",
        type=int,
        default=1048576,
        help="Collective size in Bytes")
    parser.add_argument(
        "--ll_size_factor",
        type=float,
        default=2.0,
        help="Effective Chakra comm_size multiplier for MSCCL LL protocol")
    parser.add_argument(
        "--simple_latency_factor",
        type=float,
        default=1.5,
        help="Link-latency multiplier for MSCCL Simple protocol")
    parser.add_argument(
        "--gpus_per_node",
        type=int,
        default=4,
        help="Number of GPUs per node for intra/inter-node latency classification")
    parser.add_argument(
        "--intra_latency_ns",
        type=float,
        default=4000.0,
        help="Intra-node link latency in ns")
    parser.add_argument(
        "--inter_latency_ns",
        type=float,
        default=7200.0,
        help="Inter-node link latency in ns")
    parser.add_argument(
        "--protocol_override",
        type=str,
        choices=["Simple", "LL", "LL128"],
        default=None,
        help="Override the MSCCL XML root proto field for simulation")
    parser.add_argument(
        "--msccl_stripe_calls",
        type=int,
        choices=[0, 1],
        default=0,
        help="1: a send that stands for several MSCCL stripe calls pays the per-message cost "
             "once per call (see msccl_engine_model.py). 0: unchanged traces")
    parser.add_argument(
        "--msccl_conn_tags",
        type=int,
        choices=[0, 1],
        default=0,
        help="1: tag = channel + 64 * connIndex, the MSCCL runtime's connection slot "
             "(see msccl_engine_model.py). 0: tag = channel")
    parser.add_argument(
        "--msccl_stripe_runtime",
        type=str,
        choices=["v2b", "v3"],
        default="v2b",
        help="Stripe rules of the target MSCCL runtime (see msccl_engine_model.py, part 3). "
             "v2b: unchanged traces. v3: runtime_v3's per-launch stripe count and, with stage5 stripe_passes, "
             "its per-stripe dependency flags")
    parser.add_argument(
        "--stage3_json",
        type=str,
        default="",
        help="Stage 3 measured terms as a JSON object of converter keyword arguments "
             "(net_hop_ns, nvl_hop_ns, call_cost_ns, ll_net_size_factor, ll_conn_GBps, "
             "net_link_GBps, proxy_chan_ns, proxy_chan_scope, stage4). Empty: none (Stage 1 traces)")
    args = parser.parse_args()

    logger = get_logger(args.log_filename)
    logger.debug(" ".join(sys.argv))

    try:
        if args.input_type == "msccl":
            converter = MSCCL2ChakraConverter(
                args.input_filename,
                args.output_filename,
                args.coll_size,
                args.ll_size_factor,
                args.simple_latency_factor,
                args.gpus_per_node,
                args.intra_latency_ns,
                args.inter_latency_ns,
                args.protocol_override,
                logger,
                msccl_stripe_calls=bool(args.msccl_stripe_calls),
                msccl_conn_tags=bool(args.msccl_conn_tags),
                msccl_stripe_runtime=args.msccl_stripe_runtime,
                **(json.loads(args.stage3_json) if args.stage3_json else {}))
            converter.convert()
        else:
            logger.error(f"{args.input_type} unsupported")
            sys.exit(1)
    except Exception as e:
        traceback.print_exc()
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
