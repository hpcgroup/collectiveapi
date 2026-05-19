#!/usr/bin/env python3

import argparse
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
                logger)
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
