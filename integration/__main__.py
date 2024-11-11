from transformers import HfArgumentParser

from integration.abstract import Object
from integration.args import Arguments


def main() -> None:
    class Main(Object):
        pass

    args = HfArgumentParser(Arguments).parse_args()
    # do things with args
    Main.info(f"Ran module with args: {vars(args)}")


if __name__ == "__main__":
    main()
