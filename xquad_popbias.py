import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_model_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    if "score_matrix" not in data.files:
        raise KeyError("Missing field in npz file: score_matrix")

    score_matrix = data["score_matrix"]

    user_map = None
    item_map = None

    if "user_map" in data.files:
        user_map = data["user_map"].item()
    if "item_map" in data.files:
        item_map = data["item_map"].item()

    meta = {}
    for key in ["best_epoch", "best_val_metric", "metric_name", "exp_id"]:
        if key in data.files:
            value = data[key]
            if isinstance(value, np.ndarray) and value.shape == (1,):
                meta[key] = value[0]
            else:
                meta[key] = value

    return score_matrix, user_map, item_map, meta


def load_long_tail_items(path):
    arr = np.load(path, allow_pickle=True)
    return np.asarray(arr, dtype=np.int64)


def load_user_history(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return list(obj)
    return list(obj)


def minmax_normalize_vector(x, eps=1e-12):
    x = x.astype(np.float64, copy=True)
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < eps:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def safe_candidate_selection(user_scores, candidate_size):
    candidate_size = min(candidate_size, len(user_scores))
    candidate_idx = np.argpartition(user_scores, -candidate_size)[-candidate_size:]
    candidate_idx = candidate_idx[np.argsort(user_scores[candidate_idx])[::-1]]
    return candidate_idx


def build_category_masks(num_items, long_tail_items):
    long_tail_mask = np.zeros(num_items, dtype=np.int64)
    long_tail_mask[long_tail_items] = 1
    short_head_mask = 1 - long_tail_mask
    return short_head_mask, long_tail_mask


def compute_user_category_prior(user_history, short_head_mask, long_tail_mask, default_prior=(0.5, 0.5)):
    """
    Estimate P(d|u) from the user profile.

    Category 0: short head
    Category 1: long tail
    """
    num_users = len(user_history)
    priors = np.zeros((num_users, 2), dtype=np.float64)

    for u, items in enumerate(user_history):
        items = np.asarray(items, dtype=np.int64)

        if len(items) == 0:
            priors[u] = np.array(default_prior, dtype=np.float64)
            continue

        sh_count = short_head_mask[items].sum()
        lt_count = long_tail_mask[items].sum()
        total = sh_count + lt_count

        if total == 0:
            priors[u] = np.array(default_prior, dtype=np.float64)
        else:
            priors[u, 0] = sh_count / total
            priors[u, 1] = lt_count / total

    return priors


def p_v_given_c(item_id, category_idx, short_head_mask, long_tail_mask):
    if category_idx == 0:
        return float(short_head_mask[item_id])
    return float(long_tail_mask[item_id])


def binary_uncovered_prob(selected_items, category_idx, short_head_mask, long_tail_mask):
    """
    Binary xQuAD:
    A category is uncovered only if no selected item belongs to it.
    """
    if len(selected_items) == 0:
        return 1.0

    if category_idx == 0:
        already_covered = np.any(short_head_mask[selected_items] == 1)
    else:
        already_covered = np.any(long_tail_mask[selected_items] == 1)

    return 0.0 if already_covered else 1.0


def smooth_uncovered_prob(selected_items, category_idx, short_head_mask, long_tail_mask):
    """
    Smooth xQuAD:
    The uncovered probability decreases with the fraction of selected items
    already belonging to the category.
    """
    if len(selected_items) == 0:
        return 1.0

    if category_idx == 0:
        covered_ratio = np.mean(short_head_mask[selected_items])
    else:
        covered_ratio = np.mean(long_tail_mask[selected_items])

    return 1.0 - covered_ratio


def xquad_paper_single_user(
    user_scores,
    user_prior,
    short_head_mask,
    long_tail_mask,
    top_k=10,
    lambda_xquad=0.5,
    candidate_size=100,
    variant="binary",
    normalize_relevance=True
):
    """
    Paper-style xQuAD for popularity bias control.

    Categories:
        0 -> short head
        1 -> long tail
    """
    if top_k <= 0:
        return np.array([], dtype=np.int64)

    candidates = safe_candidate_selection(user_scores, candidate_size)
    if len(candidates) == 0:
        return np.array([], dtype=np.int64)

    candidate_scores = user_scores[candidates].astype(np.float64, copy=True)
    if normalize_relevance:
        candidate_scores = minmax_normalize_vector(candidate_scores)

    selected_items = []
    remaining_local = list(range(len(candidates)))
    max_output = min(top_k, len(candidates))

    while remaining_local and len(selected_items) < max_output:
        best_local = None
        best_score = -np.inf

        for i_local in remaining_local:
            item_id = candidates[i_local]
            relevance = candidate_scores[i_local]

            diversification_term = 0.0
            for c in [0, 1]:
                p_c_given_u = user_prior[c]
                p_item_given_c = p_v_given_c(item_id, c, short_head_mask, long_tail_mask)

                if variant == "binary":
                    uncovered_prob = binary_uncovered_prob(
                        selected_items, c, short_head_mask, long_tail_mask
                    )
                elif variant == "smooth":
                    uncovered_prob = smooth_uncovered_prob(
                        selected_items, c, short_head_mask, long_tail_mask
                    )
                else:
                    raise ValueError(f"Unsupported variant: {variant}")

                diversification_term += p_c_given_u * p_item_given_c * uncovered_prob

            final_score = (1.0 - lambda_xquad) * relevance + lambda_xquad * diversification_term

            if final_score > best_score:
                best_score = final_score
                best_local = i_local

        best_item_id = candidates[best_local]
        selected_items.append(best_item_id)
        remaining_local.remove(best_local)

    return np.asarray(selected_items, dtype=np.int64)


def xquad_paper_all_users(
    score_matrix,
    user_priors,
    short_head_mask,
    long_tail_mask,
    top_k=10,
    lambda_xquad=0.5,
    candidate_size=100,
    variant="binary",
    normalize_relevance=True
):
    num_users = score_matrix.shape[0]

    if user_priors.shape[0] != num_users:
        raise ValueError(
            f"User prior shape mismatch: score_matrix has {num_users} users, "
            f"but user_priors has {user_priors.shape[0]}"
        )

    reranked = []
    for u in tqdm(range(num_users), desc=f"xQuAD-{variant} (lambda={lambda_xquad})"):
        recs = xquad_paper_single_user(
            user_scores=score_matrix[u],
            user_prior=user_priors[u],
            short_head_mask=short_head_mask,
            long_tail_mask=long_tail_mask,
            top_k=top_k,
            lambda_xquad=lambda_xquad,
            candidate_size=candidate_size,
            variant=variant,
            normalize_relevance=normalize_relevance
        )
        reranked.append(recs)

    return np.asarray(reranked, dtype=np.int64)


def save_recs_tsv(reranked_items, user_map, item_map, out_path):
    num_users, top_k = reranked_items.shape

    users = np.repeat(np.arange(num_users), top_k)
    items = reranked_items.reshape(-1)
    scores = np.tile(np.arange(top_k, 0, -1), num_users)

    df = pd.DataFrame({
        "user": users,
        "item": items,
        "rating": scores
    })

    if user_map is not None:
        df["user"] = df["user"].map(pd.Series(user_map))

    if item_map is not None:
        df["item"] = df["item"].map(pd.Series(item_map))

    if df["user"].isnull().any():
        raise ValueError("User mapping failed: NaN values found after conversion")
    if df["item"].isnull().any():
        raise ValueError("Item mapping failed: NaN values found after conversion")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(out_path, sep="\t", index=False, header=False)


def save_internal_results_npz(reranked_items, out_path, meta=None):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    payload = {"reranked_items": reranked_items}
    if meta is not None:
        for k, v in meta.items():
            payload[k] = np.array([v], dtype=object)

    np.savez_compressed(out_path, **payload)


def inspect_user(reranked_items, score_matrix, item_map=None, user_id=0, n=10):
    print(f"\nExample internal user id: {user_id}")
    print("Top internal item ids:", reranked_items[user_id][:n].tolist())

    selected_scores = score_matrix[user_id][reranked_items[user_id][:n]]
    print("Associated base scores:", selected_scores.tolist())

    if item_map is not None:
        ext_items = [item_map[i] for i in reranked_items[user_id][:n]]
        print("Top external item ids:", ext_items)


def lambda_to_suffix(value):
    return str(value).replace(".", "p")


def build_output_path(base_path, lambda_value, variant):
    root, ext = os.path.splitext(base_path)
    suffix = lambda_to_suffix(lambda_value)
    return f"{root}_{variant}_lambda_{suffix}{ext}"


def parse_args():
    parser = argparse.ArgumentParser(description="Paper-style xQuAD reranking for popularity bias control")

    parser.add_argument("--input_npz", type=str, required=True,
                        help="Path to the input NPZ file containing score_matrix")
    parser.add_argument("--long_tail_items", type=str, required=True,
                        help="Path to a .npy file containing internal long-tail item ids")
    parser.add_argument("--user_history", type=str, required=True,
                        help="Path to a .npy file containing training user histories as internal item ids")
    parser.add_argument("--output_tsv", type=str, required=True,
                        help="Base path to the output TSV file")
    parser.add_argument("--output_npz", type=str, default=None,
                        help="Optional base path to the output NPZ file")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Final recommendation list length")
    parser.add_argument("--candidate_size", type=int, default=100,
                        help="Number of initial candidates per user")
    parser.add_argument("--lambdas", type=float, nargs="+", required=True,
                        help="One or more lambda values, for example: --lambdas 0.1 0.3 0.5")
    parser.add_argument("--variant", type=str, choices=["binary", "smooth"], required=True,
                        help="xQuAD variant")
    parser.add_argument("--no_normalize_relevance", action="store_true",
                        help="Disable min-max normalization of candidate relevance scores")
    parser.add_argument("--inspect_user", type=int, default=0,
                        help="Internal user id to inspect")

    return parser.parse_args()


def main():
    args = parse_args()

    score_matrix, user_map, item_map, meta = load_model_data(args.input_npz)
    num_items = score_matrix.shape[1]

    long_tail_items = load_long_tail_items(args.long_tail_items)
    user_history = load_user_history(args.user_history)

    short_head_mask, long_tail_mask = build_category_masks(num_items, long_tail_items)
    user_priors = compute_user_category_prior(user_history, short_head_mask, long_tail_mask)

    for lambda_value in args.lambdas:
        print(f"\nRunning paper-style xQuAD ({args.variant}) for lambda = {lambda_value}")

        reranked_items = xquad_paper_all_users(
            score_matrix=score_matrix,
            user_priors=user_priors,
            short_head_mask=short_head_mask,
            long_tail_mask=long_tail_mask,
            top_k=args.top_k,
            lambda_xquad=lambda_value,
            candidate_size=args.candidate_size,
            variant=args.variant,
            normalize_relevance=not args.no_normalize_relevance
        )

        output_tsv_path = build_output_path(args.output_tsv, lambda_value, args.variant)
        save_recs_tsv(
            reranked_items=reranked_items,
            user_map=user_map,
            item_map=item_map,
            out_path=output_tsv_path
        )

        output_npz_path = None
        if args.output_npz is not None:
            output_npz_path = build_output_path(args.output_npz, lambda_value, args.variant)
            lambda_meta = dict(meta)
            lambda_meta["lambda_xquad"] = lambda_value
            lambda_meta["variant"] = args.variant
            save_internal_results_npz(
                reranked_items=reranked_items,
                out_path=output_npz_path,
                meta=lambda_meta
            )

        inspect_user(
            reranked_items=reranked_items,
            score_matrix=score_matrix,
            item_map=item_map,
            user_id=args.inspect_user,
            n=min(args.top_k, 10)
        )

        print("Run completed.")
        print(f"TSV saved to: {output_tsv_path}")
        if output_npz_path is not None:
            print(f"NPZ saved to: {output_npz_path}")


if __name__ == "__main__":
    main()