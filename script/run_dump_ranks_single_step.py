"""Single-step rolling-history per-query rank dumper for FITTER eval.

Twin of script/run_dump_ranks.py but follows the TKG-Forecasting-Eval /
gastinger2024 single-step protocol: at each test timestamp t, the
message-passing graph is

    train  +  valid  +  (test edges at all t' < t)

After scoring queries at t, the GROUND-TRUTH test edges at t are appended
(bidirectional) to the history for the next timestep. This matches the
TTRIX implementation in /mnt/nfs/home/ac139229/jiaxin/git/git/TTRIX/src/
run_entity.py::test_rolling(eval_mode="single_step").

Notable differences from the static run_dump_ranks.py:
  - ts_data.edge_index is rebuilt per test timestamp (rolling history),
    NOT the leaky train+valid+test that FITTER's default dataset
    processing produces.
  - ts_data.relation_graph is reused from the train-only relation graph
    that the FITTER dataset builds via train_sub_data (cheap, defensible
    -- the relation co-occurrence pattern changes negligibly with a
    few hundred new test edges per snapshot, mirroring the TTRIX
    optimization).
  - Per-timestep batch sizing follows cfg.train.batch_size.

Output CSV columns (per DDP rank suffix .rank<N>):
    h, t, r, time, side ('t' or 'h'), rank, num_neg

Usage (single GPU):
    python script/run_dump_ranks_single_step.py \
        --dump-path /path/yago_fitter_<src>_single_step_ranks.csv \
        -c config/transductive/eval_dump_yago.yaml \
        --dataset YAGOInd --epochs 0 --bpe null --gpus [0] \
        --output_dir /tmp/some_dir --ckpt /path/ckpts/SOURCE.pth
"""

import argparse
import csv
import os
import sys
import pprint

import torch
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fitter import tasks, util
from fitter.models import FITTER


def read_quads(path):
    """Read raw YAGO/ICEWS-Ind .txt: h \\t r \\t t \\t time [\\t time2]."""
    out = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            out.append((parts[0], parts[1], parts[2], parts[3]))
    return out


@torch.no_grad()
def test_dump_single_step(cfg, model, test_data, device, logger,
                          dump_path, filtered_data, train_data,
                          raw_dir):
    """Single-step rolling eval -> per-query rank CSV."""
    world_size = util.get_world_size()
    rank = util.get_rank()

    # --- (A) Reconstruct train/valid/test partitions from raw files in the
    # SAME load order as InductiveTemporalDataset.load_file() so that
    # entity / relation / time IDs match the CSV dump columns. ----
    inv_ent, inv_rel, inv_time = {}, {}, {}
    train_quads, valid_quads, test_quads = [], [], []
    for fname, dst in [("train.txt", train_quads),
                       ("valid.txt", valid_quads),
                       ("test.txt",  test_quads)]:
        for h, r, t, time_str in read_quads(os.path.join(raw_dir, fname)):
            if h not in inv_ent: inv_ent[h] = len(inv_ent)
            if t not in inv_ent: inv_ent[t] = len(inv_ent)
            if r not in inv_rel: inv_rel[r] = len(inv_rel)
            if time_str not in inv_time: inv_time[time_str] = len(inv_time)
            dst.append((inv_ent[h], inv_ent[t], inv_rel[r], inv_time[time_str]))

    num_rels_orig = len(inv_rel)
    num_rels_total = num_rels_orig * 2  # forward + inverse
    num_nodes = len(inv_ent)

    if rank == 0:
        logger.warning(
            f"[single_step] vocab: {num_nodes} ents, {num_rels_orig} rels, "
            f"{len(inv_time)} times"
        )
        logger.warning(
            f"[single_step] partitions: train={len(train_quads)}, "
            f"valid={len(valid_quads)}, test={len(test_quads)}"
        )

    def quads_to_edge_tensors(quads):
        if not quads:
            empty = torch.empty(2, 0, dtype=torch.long)
            return (empty,
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.long))
        ei = torch.tensor([[q[0], q[1]] for q in quads], dtype=torch.long).t()
        et = torch.tensor([q[2] for q in quads], dtype=torch.long)
        tt = torch.tensor([q[3] for q in quads], dtype=torch.long)
        return ei, et, tt

    def to_bidirectional(ei, et, tt):
        ei_bi = torch.cat([ei, ei.flip(0)], dim=1)
        et_bi = torch.cat([et, et + num_rels_orig])
        tt_bi = torch.cat([tt, tt])
        return ei_bi, et_bi, tt_bi

    # --- (B) Initial history = train + valid (bidirectional). ----
    train_ei, train_et, train_tt = quads_to_edge_tensors(train_quads)
    valid_ei, valid_et, valid_tt = quads_to_edge_tensors(valid_quads)
    test_ei,  test_et,  test_tt  = quads_to_edge_tensors(test_quads)

    base_ei, base_et, base_tt = to_bidirectional(
        torch.cat([train_ei, valid_ei], dim=1),
        torch.cat([train_et, valid_et]),
        torch.cat([train_tt, valid_tt]),
    )

    hist_ei, hist_et, hist_tt = base_ei, base_et, base_tt

    # --- (C) Borrow the train-only relation_graph that
    # InductiveTemporalDataset.process() attached to test_data. ----
    if hasattr(test_data, "relation_graph") and test_data.relation_graph is not None:
        relation_graph = test_data.relation_graph
    else:
        raise RuntimeError("expected test_data.relation_graph to be set by FITTER dataset")
    num_time = int(test_data.num_time)

    # --- (D) Sort unique test timestamps, roll forward. ----
    unique_times = sorted({q[3] for q in test_quads})

    # Open per-rank CSV shard.
    shard_path = f"{dump_path}.rank{rank}"
    os.makedirs(os.path.dirname(shard_path) or ".", exist_ok=True)
    fout = open(shard_path, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["h", "t", "r", "time", "side", "rank", "num_neg"])

    model.eval()
    total_q = 0
    for tau in unique_times:
        ts_quads = [q for q in test_quads if q[3] == tau]
        n_q = len(ts_quads)
        if n_q == 0:
            continue

        # Build per-timestep target arrays.
        ts_target_ei = torch.tensor([[q[0], q[1]] for q in ts_quads], dtype=torch.long).t()
        ts_target_et = torch.tensor([q[2] for q in ts_quads], dtype=torch.long)
        ts_target_tt = torch.tensor([q[3] for q in ts_quads], dtype=torch.long)

        # Construct ts_data with current rolling history as edge_index.
        ts_data = Data(
            edge_index=hist_ei,
            edge_type=hist_et,
            time_type=hist_tt,
            target_edge_index=ts_target_ei,
            target_edge_type=ts_target_et,
            target_time_type=ts_target_tt,
            num_nodes=num_nodes,
            num_relations=num_rels_total,
            num_time=num_time,
        ).to(device)
        ts_data.relation_graph = relation_graph

        if rank == 0:
            logger.warning(
                f"[single_step] t={tau}: {n_q} queries, history edges={hist_ei.shape[1]}"
            )

        # Score queries at this timestep.
        ts_batch = torch.cat([
            ts_data.target_edge_index,
            ts_data.target_edge_type.unsqueeze(0),
            ts_data.target_time_type.unsqueeze(0),
        ]).t()
        sampler = torch_data.DistributedSampler(ts_batch, world_size, rank)
        ts_loader = torch_data.DataLoader(ts_batch, cfg.train.batch_size, sampler=sampler)

        for batch in ts_loader:
            t_batch, h_batch = tasks.all_negative(ts_data, batch)
            t_pred = model(ts_data, t_batch)
            h_pred = model(ts_data, h_batch)

            t_mask, h_mask = tasks.strict_negative_time_mask(filtered_data, batch, train_data)

            pos_h_index, pos_t_index, pos_r_index, pos_time_index = batch.t()
            t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
            h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
            num_t_negative = t_mask.sum(dim=-1)
            num_h_negative = h_mask.sum(dim=-1)

            h_cpu = pos_h_index.cpu().tolist()
            t_cpu = pos_t_index.cpu().tolist()
            r_cpu = pos_r_index.cpu().tolist()
            ti_cpu = pos_time_index.cpu().tolist()
            t_rk = t_ranking.cpu().tolist()
            h_rk = h_ranking.cpu().tolist()
            t_nn = num_t_negative.cpu().tolist()
            h_nn = num_h_negative.cpu().tolist()
            for i in range(len(h_cpu)):
                writer.writerow([h_cpu[i], t_cpu[i], r_cpu[i], ti_cpu[i],
                                 "t", t_rk[i], t_nn[i]])
                writer.writerow([h_cpu[i], t_cpu[i], r_cpu[i], ti_cpu[i],
                                 "h", h_rk[i], h_nn[i]])
            total_q += len(h_cpu)

        # Append test edges at this t (bidirectional) to history.
        new_ei_bi, new_et_bi, new_tt_bi = to_bidirectional(
            ts_target_ei, ts_target_et, ts_target_tt,
        )
        hist_ei = torch.cat([hist_ei, new_ei_bi], dim=1)
        hist_et = torch.cat([hist_et, new_et_bi])
        hist_tt = torch.cat([hist_tt, new_tt_bi])

    fout.close()
    if rank == 0:
        logger.warning(f"[single_step] done. shards written; total local queries: {total_q}")


if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dump-path", required=True)
    pre.add_argument("--raw-dir", default=None,
                     help="raw .txt directory (default: <cfg.dataset.root>/<cls>/raw)")
    pre_args, rest = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    dump_path = pre_args.dump_path
    raw_dir_arg = pre_args.raw_dir

    args, vars_ = util.parse_args()
    cfg = util.load_config(args.config, context=vars_)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())
    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config: %s" % args.config)
        logger.warning("Dump path: %s" % dump_path)
        logger.warning(pprint.pformat(cfg))

    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)

    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data  = test_data.to(device)

    assert cfg.model.pop("class") == "FITTER"
    cfg.model.entity_model.num_time = int(test_data.num_time)
    model = FITTER(
        rel_model_cfg=cfg.model.relation_model,
        entity_model_cfg=cfg.model.entity_model,
        dataset_cfg=cfg.dataset,
    )
    assert cfg.checkpoint is not None
    state = torch.load(cfg.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if util.get_rank() == 0:
        logger.warning(f"Loaded ckpt: {cfg.checkpoint}")
        logger.warning(f"  missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device)

    # Filter graph: ALL true (h,r,t,tau) -- used by strict_negative_time_mask
    # for time-aware filtered ranking. Mirrors run.py L304-310.
    filtered_data = Data(
        edge_index=dataset._data.target_edge_index,
        edge_type=dataset._data.target_edge_type,
        num_nodes=dataset[0].num_nodes,
        time_type=dataset._data.target_time_type,
    ).to(device)

    # Determine the raw-data directory.
    if raw_dir_arg is None:
        raw_dir = os.path.join(os.path.expanduser(cfg.dataset.root),
                               cfg.dataset["class"], "raw")
    else:
        raw_dir = raw_dir_arg

    if util.get_rank() == 0:
        logger.warning(">" * 30)
        logger.warning(f"single-step rolling eval on TEST (raw dir: {raw_dir})")

    test_dump_single_step(cfg, model, test_data, device=device, logger=logger,
                          dump_path=dump_path, filtered_data=filtered_data,
                          train_data=train_data, raw_dir=raw_dir)

    if util.get_rank() == 0:
        logger.warning("done.")
