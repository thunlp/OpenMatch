import os
import sys
import time
import logging
import argparse

logger = logging.getLogger()

# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
def add_default_args(parser):
    
    ## ************************
    # Modes
    ## ************************
    modes = parser.add_argument_group("Modes")
    modes.add_argument(
        "--no_cuda", 
        action="store_true", 
        default=False,
        help="Train model on GPUs.",
    )
    modes.add_argument(
        "--local_rank", 
        default=-1, 
        type=int, 
        help="Set local_rank=0 for distributed training on multiple gpus.",
    )
    modes.add_argument(
        "--data_workers", 
        default=0, 
        type=int, 
        help="Number of subprocesses for data loading",
    )
    modes.add_argument(
        "--seed", 
        default=42, 
        type=int, 
        help="Random seed for initialization: 42",
    )
    
    ## ************************
    # File
    ## ************************
    files = parser.add_argument_group("Files")
    files.add_argument("--target_dataset_dir", 
                       required=True,
                       type=str, 
                       help="Target dataset path",
                      )
    
    ## ************************
    # Generator
    ## ************************
    generator = parser.add_argument_group("Generator")
    
    generator.add_argument(
        "--generator_mode", 
        choices=["contrastqg", "qg"],
        required=True,
        type=str, 
        help="Select contrastqg or qg mode",
    )
    
    generator.add_argument(
        "--pretrain_generator_type", 
        choices=["t5-small", "t5-base"],
        default="t5-small",
        type=str,
        help="Select pretrain generator type.",
    )
    generator.add_argument(
        "--per_gpu_gen_batch_size", 
        default=64, 
        type=int,
        help="Batch size per GPU/CPU for test."
    )
    generator.add_argument(
        "--generator_load_dir", 
        type=str, 
        required=True
    )
    generator.add_argument(
        "--reverse_genseq", 
        action='store_true', 
        default=False
    )
    generator.add_argument(
        "--max_input_length", 
        type=int, 
        default=512
    )
    generator.add_argument(
        "--max_gen_len", 
        type=int, 
        default=32, 
        help="Maximum length of output sequence"
    )
    generator.add_argument(
        "--min_gen_length", 
        type=int, 
        default=20
    )
    generator.add_argument(
        "--temperature", 
        type=float, 
        default=1.0, 
        help="temperature of 1 implies greedy sampling. \
        The value used to module the next token probabilities. Must be strictly positive"
    )
    generator.add_argument(
        "--top_p", 
        type=float, 
        default=0.9, 
        help="The cumulative probability of parameter highest probability \
        vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1."
    )
    generator.add_argument(
        "--retry_times", 
        type=int, 
        default=3
    )
    
    
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
def init_args_config(args):
    
    # logging file
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO) # logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s: [ %(message)s ]", "%m/%d/%Y %I:%M:%S %p")
    
    console = logging.StreamHandler() 
    console.setFormatter(fmt) 
    logger.addHandler(console) 
    logger.info("COMMAND: %s" % " ".join(sys.argv))