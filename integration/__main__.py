from transformers import HfArgumentParser

from integration.args import Arguments
from integration.evaluate import evaluate


def main() -> None:
    args = HfArgumentParser(Arguments).parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
