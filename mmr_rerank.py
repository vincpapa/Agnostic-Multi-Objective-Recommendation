import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_model_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    required = ["item_emb", "score_matrix"]
    for key in required:
        if key not in data.files:
            raise KeyError(f"Missing field in npz file: {key}")

    item_emb = data["item_emb"]
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

    return item_emb, score_matrix, user_map, item_map, meta


def l2_normalize(x, axis=1, eps=1e-12):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, eps, None)


def safe_candidate_selection(user_scores, candidate_size):
    candidate_size = min(candidate_size, len(user_scores))
    candidate_idx = np.argpartition(user_scores, -candidate_size)[-candidate_size:]
    candidate_idx = candidate_idx[np.argsort(user_scores[candidate_idx])[::-1]]
    return candidate_idx


def minmax_normalize_vector(x, eps=1e-12):
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < eps:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def mmr_rerank_single_user(
    user_scores,
    item_emb,
    top_k=10,
    lambda_mmr=0.7,
    candidate_size=100,
    normalize_relevance=True
):
    """
    Apply MMR reranking for a single user.

    Args:
        user_scores: Array of shape (num_items,)
        item_emb: Array of shape (num_items, dim), already L2-normalized
        top_k: Final recommendation list length
        lambda_mmr: Relevance/diversity trade-off
        candidate_size: Number of initial candidates
        normalize_relevance: Whether to min-max normalize candidate relevance scores

    Returns:
        Array of selected internal item ids with shape (top_k,)
    """
    if top_k <= 0:
        return np.array([], dtype=np.int64)

    candidates = safe_candidate_selection(user_scores, candidate_size)

    if len(candidates) == 0:
        return np.array([], dtype=np.int64)

    candidate_scores = user_scores[candidates].astype(np.float64, copy=True)

    if normalize_relevance:
        candidate_scores = minmax_normalize_vector(candidate_scores)

    cand_emb = item_emb[candidates]
    sim_matrix = cand_emb @ cand_emb.T

    selected_local = []
    remaining_local = list(range(len(candidates)))

    max_output = min(top_k, len(candidates))

    while remaining_local and len(selected_local) < max_output:
        if not selected_local:
            best_local = remaining_local[0]
            selected_local.append(best_local)
            remaining_local.remove(best_local)
            continue

        best_local = None
        best_mmr_score = -np.inf

        for i_local in remaining_local:
            relevance = candidate_scores[i_local]
            max_sim = np.max(sim_matrix[i_local, selected_local])
            mmr_score = lambda_mmr * relevance - (1.0 - lambda_mmr) * max_sim

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_local = i_local

        selected_local.append(best_local)
        remaining_local.remove(best_local)

    selected_items = candidates[selected_local]
    return selected_items.astype(np.int64)


def mmr_rerank_all_users(
    score_matrix,
    item_emb,
    top_k=10,
    lambda_mmr=0.7,
    candidate_size=100,
    normalize_relevance=True
):
    """
    Apply MMR reranking for all users.

    Args:
        score_matrix: Array of shape (num_users, num_items)
        item_emb: Array of shape (num_items, dim)
        top_k: Final recommendation list length
        lambda_mmr: Relevance/diversity trade-off
        candidate_size: Number of initial candidates
        normalize_relevance: Whether to min-max normalize candidate relevance scores

    Returns:
        Array of shape (num_users, top_k)
    """
    num_users, num_items = score_matrix.shape

    if item_emb.shape[0] != num_items:
        raise ValueError(
            f"Shape mismatch: score_matrix has {num_items} items, "
            f"but item_emb has {item_emb.shape[0]} items"
        )

    item_emb = l2_normalize(item_emb)

    reranked = []
    for u in tqdm(range(num_users), desc=f"MMR reranking (lambda={lambda_mmr})"):
        user_ranking = mmr_rerank_single_user(
            user_scores=score_matrix[u],
            item_emb=item_emb,
            top_k=top_k,
            lambda_mmr=lambda_mmr,
            candidate_size=candidate_size,
            normalize_relevance=normalize_relevance
        )
        reranked.append(user_ranking)

    return np.array(reranked, dtype=np.int64)


def save_mmr_recs_tsv(reranked_items, user_map, item_map, out_path):
    """
    Save recommendations in TSV format compatible with Elliot.
    """
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
        user_series = pd.Series(user_map)
        df["user"] = df["user"].map(user_series)

    if item_map is not None:
        item_series = pd.Series(item_map)
        df["item"] = df["item"].map(item_series)

    if df["user"].isnull().any():
        raise ValueError("User mapping failed: NaN values found after conversion")
    if df["item"].isnull().any():
        raise ValueError("Item mapping failed: NaN values found after conversion")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(out_path, sep="\t", index=False, header=False)


def save_internal_results_npz(reranked_items, out_path, meta=None):
    """
    Save reranked internal item ids and optional metadata.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_dict = {"reranked_items": reranked_items}
    if meta is not None:
        for k, v in meta.items():
            save_dict[k] = np.array([v], dtype=object)

    np.savez_compressed(out_path, **save_dict)


def inspect_user(reranked_items, score_matrix, item_map=None, user_id=0, n=10):
    """
    Print an example recommendation list for one user.
    """
    print(f"\nExample internal user id: {user_id}")
    print("Top internal item ids:", reranked_items[user_id][:n].tolist())

    selected_scores = score_matrix[user_id][reranked_items[user_id][:n]]
    print("Associated scores:", selected_scores.tolist())

    if item_map is not None:
        ext_items = [item_map[i] for i in reranked_items[user_id][:n]]
        print("Top external item ids:", ext_items)


def lambda_to_suffix(value):
    """
    Convert a lambda value into a filename-safe string.
    Example: 0.7 -> 0p7
    """
    return str(value).replace(".", "p")


def build_output_path(base_path, lambda_value):
    """
    Insert the lambda suffix before the file extension.
    Example:
        results/recs.tsv + 0.7 -> results/recs_lambda_0p7.tsv
    """
    root, ext = os.path.splitext(base_path)
    suffix = lambda_to_suffix(lambda_value)
    return f"{root}_lambda_{suffix}{ext}"


def parse_args():
    parser = argparse.ArgumentParser(description="MMR reranking from an NPZ file")

    parser.add_argument(
        "--input_npz",
        type=str,
        required=True,
        help="Path to the input NPZ file containing item_emb and score_matrix"
    )
    parser.add_argument(
        "--output_tsv",
        type=str,
        required=True,
        help="Base path to the output TSV file"
    )
    parser.add_argument(
        "--output_npz",
        type=str,
        default=None,
        help="Optional base path to the output NPZ file containing reranked items"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Final recommendation list length"
    )
    parser.add_argument(
        "--candidate_size",
        type=int,
        default=100,
        help="Number of initial candidates per user"
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        required=True,
        help="One or more lambda values for MMR, for example: --lambdas 0.3 0.5 0.7 0.9"
    )
    parser.add_argument(
        "--no_normalize_relevance",
        action="store_true",
        help="Disable min-max normalization of candidate relevance scores"
    )
    parser.add_argument(
        "--inspect_user",
        type=int,
        default=0,
        help="Internal user id to inspect"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    item_emb, score_matrix, user_map, item_map, meta = load_model_data(args.input_npz)

    for lambda_value in args.lambdas:
        print(f"\nRunning MMR for lambda = {lambda_value}")

        reranked_items = mmr_rerank_all_users(
            score_matrix=score_matrix,
            item_emb=item_emb,
            top_k=args.top_k,
            lambda_mmr=lambda_value,
            candidate_size=args.candidate_size,
            normalize_relevance=not args.no_normalize_relevance
        )

        output_tsv_path = build_output_path(args.output_tsv, lambda_value)
        save_mmr_recs_tsv(
            reranked_items=reranked_items,
            user_map=user_map,
            item_map=item_map,
            out_path=output_tsv_path
        )

        output_npz_path = None
        if args.output_npz is not None:
            output_npz_path = build_output_path(args.output_npz, lambda_value)
            lambda_meta = dict(meta)
            lambda_meta["lambda_mmr"] = lambda_value
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