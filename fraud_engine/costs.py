"""Business-cost evaluation of binary and three-way (Approve / Review / Block) policies."""
import numpy as np
import pandas as pd

from .config import CostScenario


def fraud_loss(amount, sc: CostScenario):
    """Loss if a fraudulent transaction is approved."""
    return np.asarray(amount) * sc.loss_multiplier + sc.chargeback_fee


def binary_cost(p, y, amount, thr, sc, detail=False):
    """Alert (block) when p >= thr. Total cost in USD."""
    p, y = np.asarray(p), np.asarray(y)
    block = p >= thr
    loss = fraud_loss(amount, sc)
    per = np.where(block, sc.block_ops_cost + np.where(y == 1, 0.0, sc.block_fp_cost),
                   np.where(y == 1, loss, 0.0))
    if not detail:
        return per.sum()
    tp, fp = int((block & (y == 1)).sum()), int((block & (y == 0)).sum())
    fn, tn = int((~block & (y == 1)).sum()), int((~block & (y == 0)).sum())
    return {"cost": per.sum(), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "alert_rate": block.mean()}


def three_way_cost(p, y, amount, t_review, t_block, sc, detail=False):
    """Approve p < t_review <= Review < t_block <= Block (expected value over analyst catch rate)."""
    p, y = np.asarray(p), np.asarray(y)
    blk, rev = p >= t_block, (p >= t_review) & (p < t_block)
    app = ~(blk | rev)
    loss = fraud_loss(amount, sc)
    fr = y == 1
    per = np.zeros(len(p))
    per[app & fr] = loss[app & fr]
    per[rev & fr] = sc.review_cost + \
        (1 - sc.review_catch_rate) * loss[rev & fr]
    per[rev & ~fr] = sc.review_cost + sc.review_friction_cost
    per[blk & fr] = sc.block_ops_cost
    per[blk & ~fr] = sc.block_ops_cost + sc.block_fp_cost
    if not detail:
        return per.sum()
    fraud_total = max(fr.sum(), 1)
    caught = (blk & fr).sum() + sc.review_catch_rate * (rev & fr).sum()
    return {"cost": per.sum(), "n_approve": int(app.sum()), "n_review": int(rev.sum()), "n_block": int(blk.sum()),
            "fraud_in_approve": int((app & fr).sum()), "fraud_in_review": int((rev & fr).sum()),
            "fraud_in_block": int((blk & fr).sum()), "legit_in_review": int((rev & ~fr).sum()),
            "legit_in_block": int((blk & ~fr).sum()),
            "fraud_capture_rate": caught / fraud_total,
            "review_rate": rev.mean(), "block_rate": blk.mean(),
            "fraud_loss_usd": float(loss[app & fr].sum() + (1 - sc.review_catch_rate) * loss[rev & fr].sum()),
            "false_positive_blocks": int((blk & ~fr).sum())}


def threshold_grid(p):
    q = np.quantile(p, np.linspace(0.0, 1.0, 241))
    return np.unique(np.round(np.concatenate([np.linspace(0.005, 0.995, 199), q, [0.5]]), 5))


def optimize_binary(p, y, amount, sc, grid=None):
    """Cost-minimising block threshold, subject to the scenario's block-rate cap."""
    grid = threshold_grid(p) if grid is None else grid
    costs = np.array([binary_cost(p, y, amount, t, sc) for t in grid])
    if sc.max_block_rate is not None:
        costs = np.where(np.array([(np.asarray(p) >= t).mean()
                         for t in grid]) <= sc.max_block_rate, costs, np.inf)
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i]), grid, costs


def optimize_three_way(p, y, amount, sc, n_grid=70):
    cand = np.unique(np.round(np.concatenate([np.quantile(p, np.linspace(0.30, 0.999, n_grid)),
                                              np.linspace(0.02, 0.9, 45)]), 5))
    best = (None, None, np.inf)
    p = np.asarray(p)
    mr, mb = sc.max_review_rate, sc.max_block_rate
    for i, tr in enumerate(cand):
        for tb in cand[i:]:
            blk = (p >= tb).mean()
            rev = (p >= tr).mean() - blk
            if (mb is not None and blk > mb) or (mr is not None and rev > mr):
                continue
            c = three_way_cost(p, y, amount, tr, tb, sc)
            if c < best[2]:
                best = (float(tr), float(tb), float(c))
    return best


def bayes_thresholds(mean_loss, sc):
    """Closed-form thresholds for *perfectly calibrated* probabilities and a constant loss L.

    binary  : block if p > (b + fp) / (L + fp)
    approve -> review : p > (c_r + f_r) / (e L + f_r)
    review -> block   : p > (b + fp - c_r - f_r) / ((1 - e) L + fp - f_r)
    """
    L, e = mean_loss, sc.review_catch_rate
    t_bin = (sc.block_ops_cost + sc.block_fp_cost) / (L + sc.block_fp_cost)
    t_rev = (sc.review_cost + sc.review_friction_cost) / \
        (e * L + sc.review_friction_cost)
    den = (1 - e) * L + sc.block_fp_cost - sc.review_friction_cost
    t_blk = (sc.block_ops_cost + sc.block_fp_cost -
             sc.review_cost - sc.review_friction_cost) / den
    return {"binary": float(t_bin), "review": float(t_rev), "block": float(max(t_blk, t_rev))}


def sweep_table(p, y, amount, sc, grid=None):
    grid = threshold_grid(p) if grid is None else grid
    rows = []
    for t in grid:
        d = binary_cost(p, y, amount, t, sc, detail=True)
        d["threshold"] = float(t)
        d["scenario"] = sc.name
        d["within_block_cap"] = int(
            sc.max_block_rate is None or d["alert_rate"] <= sc.max_block_rate)
        d["f1"] = 2 * d["precision"] * d["recall"] / \
            max(d["precision"] + d["recall"], 1e-9)
        rows.append(d)
    return pd.DataFrame(rows)


APPROVE, REVIEW, BLOCK = 0, 1, 2


def decision_cost(dec, y, amount, sc, detail=False):
    """Cost of an arbitrary decision vector (0 approve, 1 review, 2 block)."""
    dec, y = np.asarray(dec), np.asarray(y)
    loss = fraud_loss(amount, sc)
    fr = y == 1
    app, rev, blk = dec == APPROVE, dec == REVIEW, dec == BLOCK
    per = np.zeros(len(y))
    per[app & fr] = loss[app & fr]
    per[rev & fr] = sc.review_cost + \
        (1 - sc.review_catch_rate) * loss[rev & fr]
    per[rev & ~fr] = sc.review_cost + sc.review_friction_cost
    per[blk & fr] = sc.block_ops_cost
    per[blk & ~fr] = sc.block_ops_cost + sc.block_fp_cost
    if not detail:
        return per.sum()
    caught = (blk & fr).sum() + sc.review_catch_rate * (rev & fr).sum()
    return {"cost": per.sum(), "n": len(y), "n_approve": int(app.sum()), "n_review": int(rev.sum()), "n_block": int(blk.sum()),
            "fraud_total": int(fr.sum()), "fraud_in_approve": int((app & fr).sum()), "fraud_in_review": int((rev & fr).sum()),
            "fraud_in_block": int((blk & fr).sum()), "legit_in_review": int((rev & ~fr).sum()),
            "legit_in_block": int((blk & ~fr).sum()), "fraud_capture_rate": caught / max(fr.sum(), 1),
            "fraud_loss_usd": float(loss[app & fr].sum() + (1 - sc.review_catch_rate) * loss[rev & fr].sum())}


def decisions_from_thresholds(p, t_review, t_block):
    p = np.asarray(p)
    return np.where(p >= t_block, BLOCK, np.where(p >= t_review, REVIEW, APPROVE))
