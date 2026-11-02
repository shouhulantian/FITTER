"""Per-query rank dumper for FITTER eval (used for §2 rebuttal analysis).

Standalone twin of script/run.py that runs only the test() pass and, on
top of the standard metric printout, writes one CSV row per (query,
side) with the filtered rank under the loaded ckpt. Source code does
not modify run.py.

Output CSV columns (per DDP rank, suffix .rank<N>):
    h, t, r, time, side ('t' or 'h'), rank, num_neg

Usage (single GPU):
    python script/run_dump_ranks.py -c config/transductive/eval_dump_yago_fitter_gdelt.yaml \
        --dataset YAGOInd --epochs 0 --gpus [0] --bpe null \
        --ckpt /path/to/ckpts/GDELT.pth \
        --dump-path /path/to/yago_fitter_gdelt_ranks.csv

Usage (DDP, 4 GPU):
    python -m torch.distributed.launch --nproc_per_node=4 \
        script/run_dump_ranks.py -c <cfg> --dataset YAGOInd --epochs 0 \
        --gpus [0,1,2,3] --bpe null --ckpt <ckpt> --dump-path <out.csv>
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


@torch.no_grad()
def test_dump(cfg, model, test_data, device, logger, dump_path,
              filtered_data=None, train_data=None):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_quadruples = torch.cat([
        test_data.target_edge_index,
        test_data.target_edge_type.unsqueeze(0),
        test_data.target_time_type.unsqueeze(0),
    ]).t()
    sampler = torch_data.DistributedSampler(test_quadruples, world_size, rank)
    test_loader = torch_data.DataLoader(test_quadruples, cfg.train.batch_size, sampler=sampler)

    model.eval()

    # Open per-rank CSV shard.
    shard_path = f"{dump_path}.rank{rank}"
    os.makedirs(os.path.dirname(shard_path) or ".", exist_ok=True)
    fout = open(shard_path, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["h", "t", "r", "time", "side", "rank", "num_neg"])

    rankings, num_negatives = [], []
    for batch_idx, batch in enumerate(test_loader):
        t_batch, h_batch = tasks.all_negative(test_data, batch)
        t_pred = model(test_data, t_batch)
        h_pred = model(test_data, h_batch)

        if filtered_data is None:
            t_mask, h_mask = tasks.strict_negative_time_mask(test_data, batch, train_data)
        else:
            t_mask, h_mask = tasks.strict_negative_time_mask(filtered_data, batch, train_data)

        pos_h_index, pos_t_index, pos_r_index, pos_time_index = batch.t()
        t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
        h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)
        num_t_negative = t_mask.sum(dim=-1)
        num_h_negative = h_mask.sum(dim=-1)

        h_cpu = pos_h_index.detach().cpu().tolist()
        t_cpu = pos_t_index.detach().cpu().tolist()
        r_cpu = pos_r_index.detach().cpu().tolist()
        time_cpu = pos_time_index.detach().cpu().tolist()
        t_rank_cpu = t_ranking.detach().cpu().tolist()
        h_rank_cpu = h_ranking.detach().cpu().tolist()
        t_nneg_cpu = num_t_negative.detach().cpu().tolist()
        h_nneg_cpu = num_h_negative.detach().cpu().tolist()

        for i in range(len(h_cpu)):
            writer.writerow([h_cpu[i], t_cpu[i], r_cpu[i], time_cpu[i],
                             "t", t_rank_cpu[i], t_nneg_cpu[i]])
            writer.writerow([h_cpu[i], t_cpu[i], r_cpu[i], time_cpu[i],
                             "h", h_rank_cpu[i], h_nneg_cpu[i]])

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

        if rank == 0 and batch_idx % 10 == 0:
            logger.warning(
                f"[dump] batch {batch_idx}/{len(test_loader)}; "
                f"shard={shard_path}"
            )

    fout.close()

    # Local-only metric printout (skips DDP all_reduce; the CSVs are the
    # ground truth, we'll reconstruct aggregate metrics from those).
    if rankings:
        all_ranking = torch.cat(rankings)
        mrr = (1.0 / all_ranking.float()).mean().item()
        h10 = (all_ranking <= 10).float().mean().item()
        h1 = (all_ranking <= 1).float().mean().item()
        logger.warning(
            f"[dump rank={rank}] local n={len(all_ranking)} "
            f"MRR={mrr:.4f} H@1={h1:.4f} H@10={h10:.4f}"
        )


if __name__ == "__main__":
    # Cf. run.py: parse_args + load_config + create_working_directory.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--dump-path", required=True,
                     help="output CSV path; .rank<N> suffix appended per rank")
    pre_args, rest = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + rest  # leave the rest for util.parse_args
    dump_path = pre_args.dump_path

    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
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
    test_data = test_data.to(device)

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
        logger.warning(f"  missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        logger.warning(f"  unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    model = model.to(device)

    # InductiveTemporalDataset: train_data.target_edge_index is the union
    # train+valid+test quadruples (used as the test-time filtering graph).
    # Replicates the run.py transductive branch.
    filtered_data = Data(
        edge_index=dataset._data.target_edge_index,
        edge_type=dataset._data.target_edge_type,
        num_nodes=dataset[0].num_nodes,
        time_type=dataset._data.target_time_type,
    ).to(device)

    if util.get_rank() == 0:
        logger.warning(">" * 30)
        logger.warning("Dump per-query ranks on TEST")

    test_dump(cfg, model, test_data, device=device, logger=logger,
              filtered_data=filtered_data, train_data=train_data,
              dump_path=dump_path)

    if util.get_rank() == 0:
        logger.warning("done.")
