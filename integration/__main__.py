from transformers import HfArgumentParser

from integration.args import Arguments
from integration.evaluate import evaluate
from integration.downstream import evaluate_downstream
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    args = HfArgumentParser(Arguments).parse_args()
    if args.do_vipr_and_sizes:
        evaluate(args)
    if args.do_vqa2_zeroshot or args.do_nlvr_zeroshot:
        evaluate_downstream(args)


if __name__ == "__main__":
    main()
