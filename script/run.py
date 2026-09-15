import os
import sys
import math
import pprint
from itertools import islice

import torch
import torch_geometric as pyg
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from fitter import tasks, util
from fitter.models import FITTER

separator = ">" * 30
line = "-" * 30

def sample_quadruples(quadruples, sample_ratio=0.2):
    num_samples = int(quadruples.shape[0] * sample_ratio)  # Compute 30% of the quadruples
    indices = torch.randperm(quadruples.shape[0])[:num_samples]  # Randomly shuffle and select indices
    return quadruples[indices]  # Select the sampled quadruples

def train_and_validate_time(cfg, model, train_data, valid_data, device, logger, filtered_data=None, batch_per_epoch=None):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    train_quadruples = torch.cat([train_data.target_edge_index, train_data.target_edge_type.unsqueeze(0), train_data.target_time_type.unsqueeze(0)]).t()
    if cfg.dataset['class'] == 'GDELT':
        train_quadruples = sample_quadruples(train_quadruples)
    sampler = torch_data.DistributedSampler(train_quadruples, world_size, rank)
    train_loader = torch_data.DataLoader(train_quadruples, cfg.train.batch_size, sampler=sampler)

    batch_per_epoch = batch_per_epoch or len(train_loader)

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1

    batch_id = 0
    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            sampler.set_epoch(epoch)
            for batch in islice(train_loader, batch_per_epoch):
                # if cfg.task.num_negative == -1:
                #     num_negative = len()
                batch = tasks.negative_sampling(train_data, batch, cfg.task.num_negative,
                                                strict=cfg.task.strict_negative)
                pred = parallel_model(train_data, batch)
                #os.system('nvidia-smi')
                target = torch.zeros_like(pred)
                target[:, 0] = 1
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                neg_weight = torch.ones_like(pred)
                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        neg_weight[:, 1:] = F.softmax(pred[:, 1:] / cfg.task.adversarial_temperature, dim=-1)
                else:
                    neg_weight[:, 1:] = 1 / cfg.task.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                loss = loss.mean()

                loss.backward()
                for name, param in parallel_model.named_parameters():
                    if param.grad is None:
                        print(name)
                optimizer.step()
                optimizer.zero_grad()

                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % loss)
                losses.append(loss.item())
                batch_id += 1

            if util.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average binary cross entropy: %g" % avg_loss)

        epoch = min(cfg.train.num_epoch, i + step)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        result = test_time(cfg, model, valid_data, filtered_data=filtered_data, device=device, logger=logger,train_data=train_data)
        # if rank == 0:
        #     logger.warning(separator)
        #     logger.warning("Evaluate on test")
        # result = test_time(cfg, model, test_data, filtered_data=filtered_data, device=device, logger=logger,train_data=train_data)
        if result > best_result:
            best_result = result
            best_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % best_epoch)
    state = torch.load("model_epoch_%d.pth" % best_epoch, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()


@torch.no_grad()
def test_time(cfg, model, test_data, device, logger, filtered_data=None, return_metrics=False,train_data=None):
    world_size = util.get_world_size()
    rank = util.get_rank()

    test_quadruples = torch.cat([test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0), test_data.target_time_type.unsqueeze(0)]).t()
    sampler = torch_data.DistributedSampler(test_quadruples, world_size, rank)
    test_loader = torch_data.DataLoader(test_quadruples, cfg.train.batch_size, sampler=sampler)

    model.eval()
    rankings = []
    num_negatives = []
    tail_rankings, num_tail_negs = [], []  # for explicit tail-only evaluation needed for 5 datasets
    for batch in test_loader:
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

        rankings += [t_ranking, h_ranking]
        num_negatives += [num_t_negative, num_h_negative]

        tail_rankings += [t_ranking]
        num_tail_negs += [num_t_negative]

    ranking = torch.cat(rankings)
    num_negative = torch.cat(num_negatives)
    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)

    # ugly repetitive code for tail-only ranks processing
    tail_ranking = torch.cat(tail_rankings)
    num_tail_neg = torch.cat(num_tail_negs)
    all_size_t = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_t[rank] = len(tail_ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_t, op=dist.ReduceOp.SUM)

    # obtaining all ranks
    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative

    # the same for tails-only ranks
    cum_size_t = all_size_t.cumsum(0)
    all_ranking_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_ranking_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = tail_ranking
    all_num_negative_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_num_negative_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = num_tail_neg
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_ranking_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative_t, op=dist.ReduceOp.SUM)

    metrics = {}
    if rank == 0:
        for metric in cfg.task.metric:
            if "-tail" in metric:
                _metric_name, direction = metric.split("-")
                if direction != "tail":
                    raise ValueError("Only tail metric is supported in this mode")
                _ranking = all_ranking_t
                _num_neg = all_num_negative_t
            else:
                _ranking = all_ranking
                _num_neg = all_num_negative
                _metric_name = metric

            if _metric_name == "mr":
                score = _ranking.float().mean()
            elif _metric_name == "mrr":
                score = (1 / _ranking.float()).mean()
            elif _metric_name.startswith("hits@"):
                values = _metric_name[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (_ranking - 1).float() / _num_neg
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample - 1 negatives
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (_ranking <= threshold).float().mean()
            logger.warning("%s: %g" % (metric, score))
            metrics[metric] = score
    mrr = (1 / all_ranking.float()).mean()

    return mrr if not return_metrics else metrics


@torch.no_grad()
def test_time_single_step(cfg, model, test_data, device, logger, filtered_data=None,
                          train_data=None, return_metrics=True, eval_mode="single_step"):
    """Per-timestep rolling-history forecasting eval.

    Ported from TTRIX's test_rolling (src/run_entity.py). Attribute names
    differ: FITTER stores timestamps in `time_type` / `target_time_type` and
    the meta-graph as a `Data` object in `relation_graph` (vs TTRIX's
    `edge_time` / `target_edge_time` / `relation_adj`).

    eval_mode:
        "single_step" -- after scoring queries at time t, append GROUND-TRUTH
                         test events at t to the history graph for use at t+1.
        "multi_step"  -- append MODEL TOP-1 PREDICTIONS at t (errors compound).

    Base message-passing graph is test_data.edge_index (train edges in FITTER's
    TransductiveTemporalDataset). Rolled-in edges accumulate on top of that.

    Optimization: FITTER.forward reads data.relation_graph for the top-level
    (non-Ind) global relation model call. Rebuilding it per timestep is
    O(|R|^2) and barely changes when we add a few thousand edges. Default is
    to reuse the base relation_graph across timesteps; set
    cfg.task.rebuild_relation_graph_per_step=True to override. For the "Ind"
    branch the model rebuilds the relation graph internally per query, so this
    flag has no effect there.
    """
    world_size = util.get_world_size()
    rank = util.get_rank()

    if not hasattr(test_data, 'target_time_type') or test_data.target_time_type is None:
        raise ValueError("test_time_single_step requires temporal data with target_time_type")

    if isinstance(test_data.num_relations, torch.Tensor):
        num_rels_total = int(test_data.num_relations.item())
    else:
        num_rels_total = int(test_data.num_relations)
    num_rels_orig = num_rels_total // 2  # forward count; inverses use offset +num_rels_orig

    rebuild_per_step = False
    if hasattr(cfg.task, 'get'):
        rebuild_per_step = cfg.task.get('rebuild_relation_graph_per_step', False)
    elif hasattr(cfg.task, 'rebuild_relation_graph_per_step'):
        rebuild_per_step = cfg.task.rebuild_relation_graph_per_step

    if rebuild_per_step:
        from fitter.tasks import build_relation_graph
    base_relation_graph = getattr(test_data, 'relation_graph', None)

    target_times = test_data.target_time_type
    unique_times, _ = torch.sort(torch.unique(target_times))

    # Pull base history off-device; rebuilding relation graphs uses CPU tensors
    hist_edge_index = test_data.edge_index.detach().cpu()
    hist_edge_type = test_data.edge_type.detach().cpu()
    hist_time_type = test_data.time_type.detach().cpu()
    base_target_edge_index = test_data.target_edge_index.detach().cpu()
    base_target_edge_type = test_data.target_edge_type.detach().cpu()
    base_target_time_type = test_data.target_time_type.detach().cpu()

    all_rankings = []
    all_num_neg = []
    all_tail_rankings = []
    all_tail_num_neg = []

    model.eval()
    for t in unique_times.tolist():
        ts_mask_cpu = (base_target_time_type == t)
        n_queries = int(ts_mask_cpu.sum().item())
        if n_queries == 0:
            continue
        if rank == 0:
            logger.warning(
                f"[{eval_mode}] timestep t={t}: {n_queries} queries, history edges={hist_edge_index.shape[1]}"
            )

        ts_data = Data(
            edge_index=hist_edge_index,
            edge_type=hist_edge_type,
            time_type=hist_time_type,
            target_edge_index=base_target_edge_index[:, ts_mask_cpu],
            target_edge_type=base_target_edge_type[ts_mask_cpu],
            target_time_type=base_target_time_type[ts_mask_cpu],
            num_relations=test_data.num_relations.cpu() if isinstance(test_data.num_relations, torch.Tensor) else test_data.num_relations,
            num_nodes=test_data.num_nodes,
            num_time=test_data.num_time,
        )
        if rebuild_per_step:
            ts_data = build_relation_graph(ts_data)
        elif base_relation_graph is not None:
            ts_data.relation_graph = base_relation_graph
        ts_data = ts_data.to(device)

        # Score queries at this timestep (mirrors test_time inner loop)
        ts_quadruples = torch.cat([
            ts_data.target_edge_index,
            ts_data.target_edge_type.unsqueeze(0),
            ts_data.target_time_type.unsqueeze(0)
        ]).t()
        sampler = torch_data.DistributedSampler(ts_quadruples, world_size, rank)
        ts_loader = torch_data.DataLoader(ts_quadruples, cfg.train.batch_size, sampler=sampler)

        local_pos_h = []
        local_pos_t = []
        local_pos_r = []
        local_pos_time = []
        local_pred_t_top = []
        local_pred_h_top = []
        for batch in ts_loader:
            t_batch, h_batch = tasks.all_negative(ts_data, batch)
            t_pred = model(ts_data, t_batch)
            h_pred = model(ts_data, h_batch)

            if filtered_data is None:
                t_mask, h_mask = tasks.strict_negative_time_mask(ts_data, batch, train_data)
            else:
                t_mask, h_mask = tasks.strict_negative_time_mask(filtered_data, batch, train_data)

            pos_h_index, pos_t_index, pos_r_index, pos_time_index = batch.t()
            t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
            h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)

            all_rankings += [t_ranking, h_ranking]
            all_num_neg += [t_mask.sum(dim=-1), h_mask.sum(dim=-1)]
            all_tail_rankings += [t_ranking]
            all_tail_num_neg += [t_mask.sum(dim=-1)]

            if eval_mode == "multi_step":
                local_pos_h.append(pos_h_index)
                local_pos_t.append(pos_t_index)
                local_pos_r.append(pos_r_index)
                local_pos_time.append(pos_time_index)
                local_pred_t_top.append(t_pred.argmax(dim=-1))
                local_pred_h_top.append(h_pred.argmax(dim=-1))

        # Update history for next timestep
        if eval_mode == "single_step":
            new_fwd_edges = base_target_edge_index[:, ts_mask_cpu]
            new_fwd_etypes = base_target_edge_type[ts_mask_cpu]
            new_fwd_times = base_target_time_type[ts_mask_cpu]
            new_edges_bi = torch.cat([new_fwd_edges, new_fwd_edges.flip(0)], dim=1)
            new_etypes_bi = torch.cat([new_fwd_etypes, new_fwd_etypes + num_rels_orig])
            new_times_bi = torch.cat([new_fwd_times, new_fwd_times])
            hist_edge_index = torch.cat([hist_edge_index, new_edges_bi], dim=1)
            hist_edge_type = torch.cat([hist_edge_type, new_etypes_bi])
            hist_time_type = torch.cat([hist_time_type, new_times_bi])
        elif eval_mode == "multi_step":
            l_h = torch.cat(local_pos_h) if local_pos_h else torch.empty(0, dtype=torch.long, device=device)
            l_t = torch.cat(local_pos_t) if local_pos_t else torch.empty(0, dtype=torch.long, device=device)
            l_r = torch.cat(local_pos_r) if local_pos_r else torch.empty(0, dtype=torch.long, device=device)
            l_pt = torch.cat(local_pred_t_top) if local_pred_t_top else torch.empty(0, dtype=torch.long, device=device)
            l_ph = torch.cat(local_pred_h_top) if local_pred_h_top else torch.empty(0, dtype=torch.long, device=device)

            if world_size > 1:
                local_n = torch.tensor([len(l_h)], device=device)
                sizes = [torch.zeros_like(local_n) for _ in range(world_size)]
                dist.all_gather(sizes, local_n)
                max_n = int(torch.stack(sizes).max().item())
                def _pad(x):
                    if len(x) == max_n:
                        return x
                    pad = torch.zeros(max_n - len(x), dtype=x.dtype, device=x.device)
                    return torch.cat([x, pad])
                gh_list = [torch.zeros(max_n, dtype=torch.long, device=device) for _ in range(world_size)]
                gt_list = [torch.zeros(max_n, dtype=torch.long, device=device) for _ in range(world_size)]
                gr_list = [torch.zeros(max_n, dtype=torch.long, device=device) for _ in range(world_size)]
                gpt_list = [torch.zeros(max_n, dtype=torch.long, device=device) for _ in range(world_size)]
                gph_list = [torch.zeros(max_n, dtype=torch.long, device=device) for _ in range(world_size)]
                dist.all_gather(gh_list, _pad(l_h))
                dist.all_gather(gt_list, _pad(l_t))
                dist.all_gather(gr_list, _pad(l_r))
                dist.all_gather(gpt_list, _pad(l_pt))
                dist.all_gather(gph_list, _pad(l_ph))
                gh = torch.cat([gh_list[i][:int(sizes[i].item())] for i in range(world_size)])
                gt = torch.cat([gt_list[i][:int(sizes[i].item())] for i in range(world_size)])
                gr = torch.cat([gr_list[i][:int(sizes[i].item())] for i in range(world_size)])
                gpt = torch.cat([gpt_list[i][:int(sizes[i].item())] for i in range(world_size)])
                gph = torch.cat([gph_list[i][:int(sizes[i].item())] for i in range(world_size)])
            else:
                gh, gt, gr, gpt, gph = l_h, l_t, l_r, l_pt, l_ph

            tail_pred_edges = torch.stack([gh, gpt], dim=0).cpu()
            head_pred_edges = torch.stack([gph, gt], dim=0).cpu()
            new_edges_fwd = torch.cat([tail_pred_edges, head_pred_edges], dim=1)
            new_etypes_fwd = torch.cat([gr, gr]).cpu()
            new_times_fwd = torch.full((new_edges_fwd.shape[1],), t, dtype=torch.long)
            new_edges_bi = torch.cat([new_edges_fwd, new_edges_fwd.flip(0)], dim=1)
            new_etypes_bi = torch.cat([new_etypes_fwd, new_etypes_fwd + num_rels_orig])
            new_times_bi = torch.cat([new_times_fwd, new_times_fwd])
            hist_edge_index = torch.cat([hist_edge_index, new_edges_bi], dim=1)
            hist_edge_type = torch.cat([hist_edge_type, new_etypes_bi])
            hist_time_type = torch.cat([hist_time_type, new_times_bi])

    # Aggregate metrics
    ranking = torch.cat(all_rankings)
    num_negative = torch.cat(all_num_neg)
    tail_ranking = torch.cat(all_tail_rankings)
    num_tail_neg = torch.cat(all_tail_num_neg)

    all_size = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size[rank] = len(ranking)
    all_size_t = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_t[rank] = len(tail_ranking)
    if world_size > 1:
        dist.all_reduce(all_size, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_t, op=dist.ReduceOp.SUM)

    cum_size = all_size.cumsum(0)
    all_ranking = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_ranking[cum_size[rank] - all_size[rank]: cum_size[rank]] = ranking
    all_num_negative = torch.zeros(all_size.sum(), dtype=torch.long, device=device)
    all_num_negative[cum_size[rank] - all_size[rank]: cum_size[rank]] = num_negative

    cum_size_t = all_size_t.cumsum(0)
    all_ranking_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_ranking_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = tail_ranking
    all_num_negative_t = torch.zeros(all_size_t.sum(), dtype=torch.long, device=device)
    all_num_negative_t[cum_size_t[rank] - all_size_t[rank]: cum_size_t[rank]] = num_tail_neg
    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_ranking_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_negative_t, op=dist.ReduceOp.SUM)

    metrics = {}
    if rank == 0:
        logger.warning(line)
        logger.warning(f"[{eval_mode}] aggregated over {len(unique_times)} timesteps, {len(all_ranking)} total queries")
        for metric in cfg.task.metric:
            if "-tail" in metric:
                _metric_name = metric.split("-")[0]
                _ranking = all_ranking_t
                _num_neg = all_num_negative_t
            else:
                _ranking = all_ranking
                _num_neg = all_num_negative
                _metric_name = metric

            if _metric_name == "mr":
                score = _ranking.float().mean()
            elif _metric_name == "mrr":
                score = (1 / _ranking.float()).mean()
            elif _metric_name.startswith("hits@"):
                values = _metric_name[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    fp_rate = (_ranking - 1).float() / _num_neg
                    score = 0
                    for i in range(threshold):
                        num_comb = math.factorial(num_sample - 1) / \
                                   math.factorial(i) / math.factorial(num_sample - i - 1)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i - 1))
                    score = score.mean()
                else:
                    score = (_ranking <= threshold).float().mean()
            logger.warning("[%s] %s: %g" % (eval_mode, metric, score))
            metrics[metric] = score

    mrr = (1 / all_ranking.float()).mean()
    return metrics if return_metrics else mrr


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    
    task_name = cfg.task["name"]
    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)
    
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)

    if cfg.model.pop('class') == 'FITTER':
        #cfg.model.rule_model.num_relation = test_data.relation_graph.num_nodes
        cfg.model.entity_model.num_time = int(test_data.num_time)
        model = FITTER(
            rel_model_cfg=cfg.model.relation_model,
            entity_model_cfg=cfg.model.entity_model,
            dataset_cfg=cfg.dataset
        )

    if "checkpoint" in cfg and cfg.checkpoint is not None:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"],strict=False)

    #model = pyg.compile(model, dynamic=True)
    model = model.to(device)
    
    if task_name == "InductiveInference":
        # filtering for inductive datasets
        # Grail, MTDEA, HM datasets have validation sets based off the training graph
        # ILPC, Ingram have validation sets from the inference graph
        # filtering dataset should contain all true edges (base graph + (valid) + test) 
        if "ILPC" in cfg.dataset['class'] or "Ingram" in cfg.dataset['class']:
            # add inference, valid, test as the validation and test filtering graphs
            full_inference_edges = torch.cat([valid_data.edge_index, valid_data.target_edge_index, test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat([valid_data.edge_type, valid_data.target_edge_type, test_data.target_edge_type])
            test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)
            val_filtered_data = test_filtered_data
        else:
            # test filtering graph: inference edges + test edges
            full_inference_edges = torch.cat([test_data.edge_index, test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat([test_data.edge_type, test_data.target_edge_type])
            test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)

            # validation filtering graph: train edges + validation edges
            val_filtered_data = Data(
                edge_index=torch.cat([train_data.edge_index, valid_data.target_edge_index], dim=1),
                edge_type=torch.cat([train_data.edge_type, valid_data.target_edge_type])
            )
    else:
        # for transductive setting, use the whole graph for filtered ranking
        if 'time_type' in dataset._data.keys():
            filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=dataset[0].num_nodes,time_type=dataset._data.target_time_type)
        else:
            filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type,
                                 num_nodes=dataset[0].num_nodes)
        val_filtered_data = test_filtered_data = filtered_data
    
    val_filtered_data = val_filtered_data.to(device)
    test_filtered_data = test_filtered_data.to(device)
    if 'time_type' in dataset._data.keys():
        train_and_validate_time(cfg, model, train_data, valid_data, filtered_data=val_filtered_data, device=device,
                           batch_per_epoch=cfg.train.batch_per_epoch, logger=logger)
    eval_mode = cfg.task.get("eval_mode", "static") if hasattr(cfg.task, "get") else getattr(cfg.task, "eval_mode", "static")

    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on test")
    if 'time_type' in dataset._data.keys():
        if eval_mode in ("single_step", "multi_step"):
            test_time_single_step(cfg, model, test_data, filtered_data=test_filtered_data, device=device,
                                  logger=logger, train_data=train_data, eval_mode=eval_mode)
        elif eval_mode == "both":
            if util.get_rank() == 0:
                logger.warning("--- eval_mode=static ---")
            test_time(cfg, model, test_data, filtered_data=test_filtered_data, device=device, logger=logger, train_data=train_data)
            if util.get_rank() == 0:
                logger.warning("--- eval_mode=single_step ---")
            test_time_single_step(cfg, model, test_data, filtered_data=test_filtered_data, device=device,
                                  logger=logger, train_data=train_data, eval_mode="single_step")
        else:
            test_time(cfg, model, test_data, filtered_data=test_filtered_data, device=device, logger=logger, train_data=train_data)

    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate on valid")
    if 'time_type' in dataset._data.keys():
        if eval_mode in ("single_step", "multi_step"):
            test_time_single_step(cfg, model, valid_data, filtered_data=val_filtered_data, device=device,
                                  logger=logger, train_data=train_data, eval_mode=eval_mode)
        elif eval_mode == "both":
            if util.get_rank() == 0:
                logger.warning("--- eval_mode=static ---")
            test_time(cfg, model, valid_data, filtered_data=val_filtered_data, device=device, logger=logger, train_data=train_data)
            if util.get_rank() == 0:
                logger.warning("--- eval_mode=single_step ---")
            test_time_single_step(cfg, model, valid_data, filtered_data=val_filtered_data, device=device,
                                  logger=logger, train_data=train_data, eval_mode="single_step")
        else:
            test_time(cfg, model, valid_data, filtered_data=val_filtered_data, device=device, logger=logger, train_data=train_data)
