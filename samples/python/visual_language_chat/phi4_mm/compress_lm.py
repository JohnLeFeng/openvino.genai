import gc
import sys
import nncf
import shutil
import logging
import argparse

from pathlib import Path

import openvino as ov


logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout) 
log = logging.getLogger()


def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    # fmt: off
    args.add_argument('-h', '--help', action = 'help',
                      help='Show this help message and exit.')
    args.add_argument('-m','--model',type = str, default = "Phi-4-multimodal-instruct-ov", required = False,
                      help='Optional. Model name and default is "Phi-4-multimodal-instruct-ov".')
    args.add_argument('-gs','--group_size',type = int, default = 64, required = False,
                      help='Optional. Group size and default is 64.')
    args.add_argument('-r','--ratio',type = float, default = 1.0, required = False,
                      help='Optional. Compression ratio and default is 1.0.')
    return parser.parse_args()


def compress_lm_weights(model_dir, group_size, ratio):
    compression_configuration = {"mode": nncf.CompressWeightsMode.INT4_SYM, "group_size": group_size, "ratio": ratio, "all_layers": True}
    ov_model_path = model_dir / "openvino_language_model.xml"
    ov_int4_model_path = model_dir / "openvino_language_model_int4.xml"
    ov_model = ov.Core().read_model(ov_model_path)
    ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration)
    ov.save_model(ov_compressed_model, ov_int4_model_path)
    del ov_compressed_model
    del ov_model
    gc.collect()
    ov_model_path.unlink()
    ov_model_path.with_suffix(".bin").unlink()
    shutil.move(ov_int4_model_path, ov_model_path)
    shutil.move(ov_int4_model_path.with_suffix(".bin"), ov_model_path.with_suffix(".bin"))


def main():
    args = parse_args()

    model_dir = Path(args.model)
    group_size = args.group_size
    ratio = args.ratio
    
    log.info(f"Compressing language model in {model_dir} with group size {group_size} and ratio {ratio}")
    compress_lm_weights(model_dir, group_size, ratio)
    log.info("Language model compression is done.")


if __name__ == "__main__":
    sys.exit(main())