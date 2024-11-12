from transformers import HfArgumentParser
import os

from integration.args import Arguments
from integration.evaluate import evaluate


def main() -> None:
    args = HfArgumentParser(Arguments).parse_args()
    # if args.cache_location is not None:
    #     os.environ['HF_HOME'] = args.cache_location
    #     os.environ['HF_HUB_CACHE'] = args.cache_location
    #     os.environ['HUGGINGFACE_HUB_CACHE'] = args.cache_location
    evaluate(args)


if __name__ == "__main__":
    main()
