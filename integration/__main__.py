from dotenv import load_dotenv
from transformers import HfArgumentParser

from integration.args import Arguments
from integration.evaluate import evaluate
from integration.downstream import evaluate_downstream


def main() -> None:
    load_dotenv()
    args = HfArgumentParser(Arguments).parse_args()
    if args.do_vipr_and_sizes:
        evaluate(args)
    if any([
        args.do_vqa2_zeroshot,
        args.do_vqa2_finetune,
        args.do_nlvr_zeroshot,
        args.do_nlvr_finetune,
    ]):
        evaluate_downstream(args)


if __name__ == "__main__":
    main()
