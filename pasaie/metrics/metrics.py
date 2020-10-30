def micro_p_r_f1_score(preds, golds):
    """calculate precision/recall/f1 score

    Args:
        preds (list[list]): seq tags predicted by model
        golds (list[list]): gold tags of corpus

    Returns:
        p (float): precision
        r (float): recall
        f1 (float): f1 score
    """
    assert len(preds) == len(golds)
    p_sum = 0
    r_sum = 0
    hits = 0
    for pred, gold in zip(preds, golds):
        p_sum += len(pred)
        r_sum += len(gold)
        for label in pred:
            if label in gold:
                hits += 1
    micro_p = hits / p_sum if p_sum > 0 else 0
    micro_r = hits / r_sum if r_sum > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
    return micro_p, micro_r, micro_f1