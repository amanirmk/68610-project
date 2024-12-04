import pandas as pd
from integration.tasks.vqa import zeroshot_vqa2
from integration.tasks.nlvr import zeroshot_nlvr
from integration.abstract import Object


class DownstreamEval(Object):
    pass


def evaluate_downstream(args):
    if args.do_nlvr_zeroshot:
        rows = []
        for model_name in args.model_names:
            try:
                precision, consistency = zeroshot_nlvr(model_name, args.device)
            except Exception as e:
                DownstreamEval.error(f"Failed to evaluate {model_name}: {e}")
                precision = -1.0
                consistency = -1.0
            finally:
                rows.append(
                    {
                        "model": model_name,
                        "NLVR_precision": precision,
                        "NLVR_consistency": consistency,
                    }
                )
        df = pd.DataFrame(rows)
        df.to_csv(
            f"{'_'.join(m.split('/')[1] for m in df['model'].unique())}_NLVR.csv",
            index=False,
        )

    if args.do_vqa2_zeroshot:
        rows = []
        for model_name in args.model_names:
            try:
                exact_acc, partial_acc = zeroshot_vqa2(
                    model_name, args.device, limit=100_000
                )
            except Exception as e:
                DownstreamEval.error(f"Failed to evaluate {model_name}: {e}")
                exact_acc = -1.0
                partial_acc = -1.0
            finally:
                rows.append(
                    {
                        "model": model_name,
                        "VQA2_exact_acc": exact_acc,
                        "VQA2_partial_acc": partial_acc,
                    }
                )
        df = pd.DataFrame(rows)
        df.to_csv(
            f"{'_'.join(m.split('/')[1] for m in df['model'].unique())}_VQA2.csv",
            index=False,
        )
