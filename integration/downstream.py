import pandas as pd
from integration.tasks.vqa import zeroshot_vqa2, finetune_vqa2
from integration.tasks.nlvr import zeroshot_nlvr, finetune_nlvr
from integration.abstract import Object


class DownstreamEval(Object):
    pass

NLVR_TRAIN_LIMIT = 100 # x6 examples, use 5k when fr

VQA2_TRAIN_LIMIT = 100 # use 30k when fr
VQA2_EVAL_LIMIT = 100 # use 30k when fr


def evaluate_downstream(args):
    if args.do_nlvr_zeroshot:
        rows = []
        for model_name in args.model_names:
            try:
                precision, consistency = zeroshot_nlvr(model_name, save_intermediate=args.save_intermediate)
            except Exception as e:  # pylint: disable=broad-exception-caught
                DownstreamEval.error(f"Failed to evaluate {model_name}: {e}")
                precision = -1.0
                consistency = -1.0
            finally:
                rows.append(
                    {
                        "model": model_name,
                        "NLVR_zeroshot_precision": precision,  # pylint: disable=used-before-assignment
                        "NLVR_zeroshot_consistency": consistency,  # pylint: disable=used-before-assignment
                    }
                )
        df = pd.DataFrame(rows)
        df.to_csv(
            f"{'_'.join(m.split('/')[1] for m in df['model'].unique())}_NLVR_zeroshot.csv",
            index=False,
        )

    if args.do_nlvr_finetune:
        rows = []
        for model_name in args.model_names:
            try:
                precision, consistency = finetune_nlvr(model_name, save_intermediate=args.save_intermediate, train_limit=NLVR_TRAIN_LIMIT)
            except Exception as e:  # pylint: disable=broad-exception-caught
                DownstreamEval.error(f"Failed to evaluate {model_name}: {e}")
                precision = -1.0
                consistency = -1.0
            finally:
                rows.append(
                    {
                        "model": model_name,
                        "NLVR_finetune_precision": precision,  # pylint: disable=used-before-assignment
                        "NLVR_finetune_consistency": consistency,  # pylint: disable=used-before-assignment
                    }
                )
        df = pd.DataFrame(rows)
        df.to_csv(
            f"{'_'.join(m.split('/')[1] for m in df['model'].unique())}_NLVR_finetune.csv",
            index=False,
        )

    if args.do_vqa2_zeroshot:
        rows = []
        for model_name in args.model_names:
            try:
                exact_acc, partial_acc = zeroshot_vqa2(
                    model_name, save_intermediate=args.save_intermediate, limit=VQA2_EVAL_LIMIT
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                DownstreamEval.error(f"Failed to evaluate {model_name}: {e}")
                exact_acc = -1.0
                partial_acc = -1.0
            finally:
                rows.append(
                    {
                        "model": model_name,
                        "VQA2_zeroshot_exact_acc": exact_acc,  # pylint: disable=used-before-assignment
                        "VQA2_zeroshot_partial_acc": partial_acc,  # pylint: disable=used-before-assignment
                    }
                )
        df = pd.DataFrame(rows)
        df.to_csv(
            f"{'_'.join(m.split('/')[1] for m in df['model'].unique())}_VQA2_zeroshot.csv",
            index=False,
        )

    if args.do_vqa2_finetune:
        rows = []
        for model_name in args.model_names:
            try:
                exact_acc, partial_acc = finetune_vqa2(
                    model_name, save_intermediate=args.save_intermediate, train_limit=VQA2_TRAIN_LIMIT, eval_limit=VQA2_EVAL_LIMIT
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                DownstreamEval.error(f"Failed to evaluate {model_name}: {e}")
                exact_acc = -1.0
                partial_acc = -1.0
            finally:
                rows.append(
                    {
                        "model": model_name,
                        "VQA2_finetune_exact_acc": exact_acc,  # pylint: disable=used-before-assignment
                        "VQA2_finetune_partial_acc": partial_acc,  # pylint: disable=used-before-assignment
                    }
                )
        df = pd.DataFrame(rows)
        df.to_csv(
            f"{'_'.join(m.split('/')[1] for m in df['model'].unique())}_VQA2_finetune.csv",
            index=False,
        )
