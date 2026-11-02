import os
import csv
import shutil
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip

from fitter.tasks import build_relation_graph

class TransductiveDataset(InMemoryDataset):

    delimiter = None
    
    def __init__(self, root, transform=None, pre_transform=build_relation_graph, **kwargs):

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["train.txt", "valid.txt", "test.txt"]
    
    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url, self.raw_dir)
            os.rename(download_path, path)
    
    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, r, v = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab), #entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }
    
    # default loading procedure: process train/valid/test files, create graphs from them
    def process(self):

        train_files = self.raw_paths[:3]

        train_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        valid_results = self.load_file(train_files[1], 
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        test_results = self.load_file(train_files[2],
                        train_results["inv_entity_vocab"], train_results["inv_rel_vocab"])
        
        # in some datasets, there are several new nodes in the test set, eg 123,143 YAGO train adn 123,182 in YAGO test
        # for consistency with other experimental results, we'll include those in the full vocab and num nodes
        num_node = test_results["num_node"] 
        # the same for rels: in most cases train == test for transductive
        # for AristoV4 train rels 1593, test 1604
        num_relations = test_results["num_relation"]

        train_triplets = train_results["triplets"]
        valid_triplets = valid_results["triplets"]
        test_triplets = test_results["triplets"]

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_triplets], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_triplets])

        valid_edges = torch.tensor([[t[0], t[1]] for t in valid_triplets], dtype=torch.long).t()
        valid_etypes = torch.tensor([t[2] for t in valid_triplets])

        test_edges = torch.tensor([[t[0], t[1]] for t in test_triplets], dtype=torch.long).t()
        test_etypes = torch.tensor([t[2] for t in test_triplets])

        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat([train_target_etypes, train_target_etypes+num_relations])

        train_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_relations*2)
        valid_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations*2)
        test_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                         target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations*2)

        # build graphs of relations
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def __repr__(self):
        return "%s()" % (self.name)
    
    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"


class TransductiveTemporalDataset(InMemoryDataset):
    delimiter = None

    def __init__(self, root, transform=None, pre_transform=build_relation_graph, **kwargs):

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["train.txt", "valid.txt", "test.txt"]

    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url, self.raw_dir)
            os.rename(download_path, path)

    def load_file(self, quadruple_file, inv_entity_vocab={}, inv_rel_vocab={},inv_time_vocab={}):

        quadruples = []
        entity_cnt, rel_cnt, time_cnt = len(inv_entity_vocab), len(inv_rel_vocab), len(inv_time_vocab)

        with open(quadruple_file, "r", encoding="utf-8") as fin:
            for l in fin:
                t_instance = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                #print(t_instance)
                if t_instance[0] not in inv_entity_vocab:
                    inv_entity_vocab[t_instance[0]] = entity_cnt
                    entity_cnt += 1
                if t_instance[2] not in inv_entity_vocab:
                    inv_entity_vocab[t_instance[2]] = entity_cnt
                    entity_cnt += 1
                if t_instance[1] not in inv_rel_vocab:
                    inv_rel_vocab[t_instance[1]] = rel_cnt
                    rel_cnt += 1
                if t_instance[3] not in inv_time_vocab:
                    inv_time_vocab[t_instance[3]] = time_cnt
                    time_cnt += 1
                # For interval-valued datasets (5 cols: u r v t_start t_end)
                # the 5th column carries no signal used downstream and is
                # dropped at parse time — every row is stored as a 4-tuple.
                u = inv_entity_vocab[t_instance[0]]
                r = inv_rel_vocab[t_instance[1]]
                v = inv_entity_vocab[t_instance[2]]
                t1 = inv_time_vocab[t_instance[3]]
                quadruples.append((u, v, r, t1))

        return {
            "quadruples": quadruples,
            "num_node": len(inv_entity_vocab),  # entity_cnt,
            "num_relation": len(inv_rel_vocab),
            "num_time": len(inv_time_vocab),
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab,
            "inv_time_vocab": inv_time_vocab
        }

    # default loading procedure: process train/valid/test files, create graphs from them
    def process(self):

        train_files = self.raw_paths[:3]

        train_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={}, inv_time_vocab={})
        valid_results = self.load_file(train_files[1],
                                       train_results["inv_entity_vocab"], train_results["inv_rel_vocab"], train_results['inv_time_vocab'])
        test_results = self.load_file(train_files[2],
                                      valid_results["inv_entity_vocab"], valid_results["inv_rel_vocab"],valid_results['inv_time_vocab'])

        # in some datasets, there are several new nodes in the test set, eg 123,143 YAGO train adn 123,182 in YAGO test
        # for consistency with other experimental results, we'll include those in the full vocab and num nodes
        num_node = test_results["num_node"]
        # the same for rels: in most cases train == test for transductive
        # for AristoV4 train rels 1593, test 1604
        num_relations = test_results["num_relation"]
        num_time = test_results['num_time']
        #self.num_relations_real = num_relations

        train_quadruples = train_results["quadruples"]
        valid_quadruples = valid_results["quadruples"]
        test_quadruples = test_results["quadruples"]

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_quadruples], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_quadruples])

        valid_edges = torch.tensor([[t[0], t[1]] for t in valid_quadruples], dtype=torch.long).t()
        valid_etypes = torch.tensor([t[2] for t in valid_quadruples])

        test_edges = torch.tensor([[t[0], t[1]] for t in test_quadruples], dtype=torch.long).t()
        test_etypes = torch.tensor([t[2] for t in test_quadruples])

        train_target_ttypes = torch.tensor([t[3] for t in train_quadruples])
        valid_target_ttypes = torch.tensor([t[3] for t in valid_quadruples])
        test_target_ttypes = torch.tensor([t[3] for t in test_quadruples])

        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat([train_target_etypes, train_target_etypes + num_relations])
        train_ttypes = torch.cat([train_target_ttypes, train_target_ttypes])

        train_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes,
                          num_relations=num_relations * 2, num_time=num_time, time_type=train_ttypes, target_time_type=train_target_ttypes)
        valid_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations * 2,num_time=num_time,time_type=train_ttypes, target_time_type=valid_target_ttypes)
        test_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                         target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations * 2,num_time=num_time,time_type=train_ttypes, target_time_type=test_target_ttypes)

        # build graphs of relations
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def provide_vocab(self):

        train_files = self.raw_paths[:3]

        train_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={}, inv_time_vocab={})
        valid_results = self.load_file(train_files[1],
                                       train_results["inv_entity_vocab"], train_results["inv_rel_vocab"], train_results['inv_time_vocab'])
        test_results = self.load_file(train_files[2],
                                      valid_results["inv_entity_vocab"], valid_results["inv_rel_vocab"],valid_results['inv_time_vocab'])

        return test_results["inv_entity_vocab"], test_results["inv_rel_vocab"], test_results["inv_time_vocab"]

    def __repr__(self):
        return "%s()" % (self.name)

    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

class InductiveTemporalDataset(InMemoryDataset):
    delimiter = None

    def __init__(self, root, transform=None, pre_transform=build_relation_graph, **kwargs):

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["train.txt", "valid.txt", "test.txt"]

    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url, self.raw_dir)
            os.rename(download_path, path)

    def load_file(self, quadruple_file, inv_entity_vocab={}, inv_rel_vocab={},inv_time_vocab={}):

        quadruples = []
        entity_cnt, rel_cnt, time_cnt = len(inv_entity_vocab), len(inv_rel_vocab), len(inv_time_vocab)

        with open(quadruple_file, "r", encoding="utf-8") as fin:
            for l in fin:
                t_instance = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                #print(t_instance)
                if t_instance[0] not in inv_entity_vocab:
                    inv_entity_vocab[t_instance[0]] = entity_cnt
                    entity_cnt += 1
                if t_instance[2] not in inv_entity_vocab:
                    inv_entity_vocab[t_instance[2]] = entity_cnt
                    entity_cnt += 1
                if t_instance[1] not in inv_rel_vocab:
                    inv_rel_vocab[t_instance[1]] = rel_cnt
                    rel_cnt += 1
                if t_instance[3] not in inv_time_vocab:
                    inv_time_vocab[t_instance[3]] = time_cnt
                    time_cnt += 1
                # For interval-valued datasets (5 cols: u r v t_start t_end)
                # the 5th column carries no signal used downstream and is
                # dropped at parse time — every row is stored as a 4-tuple.
                u = inv_entity_vocab[t_instance[0]]
                r = inv_rel_vocab[t_instance[1]]
                v = inv_entity_vocab[t_instance[2]]
                t1 = inv_time_vocab[t_instance[3]]
                quadruples.append((u, v, r, t1))

        return {
            "quadruples": quadruples,
            "num_node": len(inv_entity_vocab),  # entity_cnt,
            "num_relation": len(inv_rel_vocab),
            "num_time": len(inv_time_vocab),
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab,
            "inv_time_vocab": inv_time_vocab
        }

    # default loading procedure: process train/valid/test files, create graphs from them
    def process(self):

        train_files = self.raw_paths[:3]

        train_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={}, inv_time_vocab={})
        valid_results = self.load_file(train_files[1],
                                       train_results["inv_entity_vocab"], train_results["inv_rel_vocab"], train_results['inv_time_vocab'])
        test_results = self.load_file(train_files[2],
                                      valid_results["inv_entity_vocab"], valid_results["inv_rel_vocab"],valid_results['inv_time_vocab'])

        # in some datasets, there are several new nodes in the test set, eg 123,143 YAGO train adn 123,182 in YAGO test
        # for consistency with other experimental results, we'll include those in the full vocab and num nodes
        num_node = test_results["num_node"]
        # the same for rels: in most cases train == test for transductive
        # for AristoV4 train rels 1593, test 1604
        num_relations = test_results["num_relation"]
        num_time = test_results['num_time']
        #self.num_relations_real = num_relations

        train_quadruples = train_results["quadruples"] + valid_results['quadruples'] + test_results['quadruples']
        valid_quadruples = valid_results["quadruples"]
        test_quadruples = test_results["quadruples"]

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_quadruples], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_quadruples])

        valid_edges = torch.tensor([[t[0], t[1]] for t in valid_quadruples], dtype=torch.long).t()
        valid_etypes = torch.tensor([t[2] for t in valid_quadruples])

        test_edges = torch.tensor([[t[0], t[1]] for t in test_quadruples], dtype=torch.long).t()
        test_etypes = torch.tensor([t[2] for t in test_quadruples])

        train_target_ttypes = torch.tensor([t[3] for t in train_quadruples])
        valid_target_ttypes = torch.tensor([t[3] for t in valid_quadruples])
        test_target_ttypes = torch.tensor([t[3] for t in test_quadruples])

        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat([train_target_etypes, train_target_etypes + num_relations])
        train_ttypes = torch.cat([train_target_ttypes, train_target_ttypes])

        train_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes,
                          num_relations=num_relations * 2, num_time=num_time, time_type=train_ttypes, target_time_type=train_target_ttypes)
        valid_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=valid_edges, target_edge_type=valid_etypes, num_relations=num_relations * 2,num_time=num_time,time_type=train_ttypes, target_time_type=valid_target_ttypes)
        test_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                         target_edge_index=test_edges, target_edge_type=test_etypes, num_relations=num_relations * 2,num_time=num_time,time_type=train_ttypes, target_time_type=test_target_ttypes)

        train_quadruples = train_results["quadruples"]

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_quadruples], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_quadruples])
        train_target_ttypes = torch.tensor([t[3] for t in train_quadruples])

        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat([train_target_etypes, train_target_etypes + num_relations])
        train_ttypes = torch.cat([train_target_ttypes, train_target_ttypes])

        train_sub_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes,
                          num_relations=num_relations * 2, num_time=num_time, time_type=train_ttypes, target_time_type=train_target_ttypes)

        # build graphs of relations
        if self.pre_transform is not None:
            train_data = self.pre_transform(train_sub_data)
            valid_data.relation_graph = train_data.relation_graph
            test_data.relation_graph = train_data.relation_graph

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def provide_vocab(self):

        train_files = self.raw_paths[:3]

        train_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={}, inv_time_vocab={})
        valid_results = self.load_file(train_files[1],
                                       train_results["inv_entity_vocab"], train_results["inv_rel_vocab"], train_results['inv_time_vocab'])
        test_results = self.load_file(train_files[2],
                                      valid_results["inv_entity_vocab"], valid_results["inv_rel_vocab"],valid_results['inv_time_vocab'])

        return test_results["inv_entity_vocab"], test_results["inv_rel_vocab"], test_results["inv_time_vocab"]

    def __repr__(self):
        return "%s()" % (self.name)

    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"


class ICEWS14(TransductiveTemporalDataset):
    name = "ICEWS14"
    delimiter = "\t"

class ICEWS0515_sym(TransductiveTemporalDataset):
    name = "ICEWS0515_sym"
    delimiter = "\t"

class ICEWS0515(TransductiveTemporalDataset):
    name = "ICEWS0515"
    delimiter = "\t"


class GDELT(TransductiveTemporalDataset):
    name = "GDELT"
    delimiter = "\t"

class Smallpedia(TransductiveTemporalDataset):
    name = "Smallpedia"
    delimiter = "\t"

class SmallpediaInd(InductiveTemporalDataset):
    name = "SmallpediaInd"
    delimiter = "\t"

class Polecat(TransductiveTemporalDataset):
    name = "Polecat"
    delimiter = "\t"

class PolecatInd(InductiveTemporalDataset):
    name = "PolecatInd"
    delimiter = "\t"

class ICEWS14Ind(InductiveTemporalDataset):
    name = "ICEWS14Ind"
    delimiter = "\t"

class ICEWS18Ind(InductiveTemporalDataset):
    name = "ICEWS18Ind"
    delimiter = "\t"


class YAGOInd(InductiveTemporalDataset):
    name = "YAGOInd"
    delimiter = "\t"

class GDELTInd(InductiveTemporalDataset):
    name = "GDELTInd"
    delimiter = "\t"


class WIKIInd(InductiveTemporalDataset):
    name = "WIKIInd"
    delimiter = "\t"


class TransductiveTemporalForecastDataset(TransductiveTemporalDataset):
    """Chronological-split temporal KG datasets (YAGO, ICEWS18, WIKI + IndT sweeps).

    Mirrors TTRIX's _TemporalForecastDataset: test_data.edge_index is
    train + valid bidirectional (with inverse-relation offsets), giving
    the rolling forecaster in run.py's test_time_single_step a base
    context that includes all validation history. The base
    TransductiveTemporalDataset uses train only for the test MP graph,
    which is the strictest extrapolation protocol but does not match the
    RE-GCN / RE-Net feedgt=True forecasting benchmarks.

    Also fixes time-index ordering: after loading, all time strings are
    remapped so integer indices reflect calendar order (via _parse_time).
    This matters because run.py's rolling loop iterates over sorted
    unique times, and FITTER's RoPE-style temporal encoding assumes
    consecutive indices represent close-in-time events.
    """

    @property
    def processed_dir(self):
        # Cache under processed_forecast/ so we do not collide with the
        # older InductiveTemporalDataset-based classes that share `name`
        # (e.g. YAGOInd, ICEWS18Ind, WIKIInd) but process() differently.
        return os.path.join(self.root, self.name, "processed_forecast")

    def _parse_time(self, s):
        """Parse a timestamp string to something totally-orderable in calendar order.

        Default: try int, fall back to the string itself. Handles both
        integer indices ("0", "1", ...) and ISO 8601 dates ("2000-01-05",
        which sort lexicographically in calendar order). Subclasses can
        override for other formats (e.g. dd/mm/yyyy).
        """
        try:
            return int(s)
        except (ValueError, TypeError):
            return s

    def _remap_time_vocab(self, inv_time_vocab, *quad_lists):
        """Reassign time indices so they sort in calendar order.

        Returns (new_vocab, remapped_quad_lists). Each quad is
        (u, v, r, t_old) -> (u, v, r, t_new).
        """
        sorted_keys = sorted(inv_time_vocab.keys(), key=self._parse_time)
        new_vocab = {s: i for i, s in enumerate(sorted_keys)}
        old_to_new = {inv_time_vocab[s]: new_vocab[s] for s in inv_time_vocab}
        remapped = [
            [(u, v, r, old_to_new[t]) for (u, v, r, t) in lst]
            for lst in quad_lists
        ]
        return new_vocab, remapped

    def process(self):
        train_files = self.raw_paths[:3]

        train_results = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={}, inv_time_vocab={})
        valid_results = self.load_file(train_files[1],
                                       train_results["inv_entity_vocab"], train_results["inv_rel_vocab"], train_results['inv_time_vocab'])
        test_results = self.load_file(train_files[2],
                                      valid_results["inv_entity_vocab"], valid_results["inv_rel_vocab"], valid_results['inv_time_vocab'])

        num_node = test_results["num_node"]
        num_relations = test_results["num_relation"]
        num_time = test_results["num_time"]

        # Calendar-order remap so unique_times.sort() reflects real time.
        _, (train_quadruples, valid_quadruples, test_quadruples) = self._remap_time_vocab(
            test_results["inv_time_vocab"],
            train_results["quadruples"],
            valid_results["quadruples"],
            test_results["quadruples"],
        )

        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_quadruples], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_quadruples])
        train_target_ttypes = torch.tensor([t[3] for t in train_quadruples])

        valid_edges = torch.tensor([[t[0], t[1]] for t in valid_quadruples], dtype=torch.long).t()
        valid_etypes = torch.tensor([t[2] for t in valid_quadruples])
        valid_ttypes = torch.tensor([t[3] for t in valid_quadruples])

        test_edges = torch.tensor([[t[0], t[1]] for t in test_quadruples], dtype=torch.long).t()
        test_etypes = torch.tensor([t[2] for t in test_quadruples])
        test_ttypes = torch.tensor([t[3] for t in test_quadruples])

        # Train MP graph = train only (bidirectional + inverse offsets)
        train_edges_bi = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes_bi = torch.cat([train_target_etypes, train_target_etypes + num_relations])
        train_ttypes_bi = torch.cat([train_target_ttypes, train_target_ttypes])

        # Test MP graph = train + valid (feedgt=True forecast baseline)
        trainval_edges_fwd = torch.cat([train_target_edges, valid_edges], dim=1)
        trainval_etypes_fwd = torch.cat([train_target_etypes, valid_etypes])
        trainval_ttypes_fwd = torch.cat([train_target_ttypes, valid_ttypes])
        trainval_edges_bi = torch.cat([trainval_edges_fwd, trainval_edges_fwd.flip(0)], dim=1)
        trainval_etypes_bi = torch.cat([trainval_etypes_fwd, trainval_etypes_fwd + num_relations])
        trainval_ttypes_bi = torch.cat([trainval_ttypes_fwd, trainval_ttypes_fwd])

        train_data = Data(edge_index=train_edges_bi, edge_type=train_etypes_bi, num_nodes=num_node,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes,
                          num_relations=num_relations * 2, num_time=num_time,
                          time_type=train_ttypes_bi, target_time_type=train_target_ttypes)
        # Valid: MP graph = train only, targets = valid edges
        valid_data = Data(edge_index=train_edges_bi, edge_type=train_etypes_bi, num_nodes=num_node,
                          target_edge_index=valid_edges, target_edge_type=valid_etypes,
                          num_relations=num_relations * 2, num_time=num_time,
                          time_type=train_ttypes_bi, target_time_type=valid_ttypes)
        # Test: MP graph = train + valid, targets = test edges
        test_data = Data(edge_index=trainval_edges_bi, edge_type=trainval_etypes_bi, num_nodes=num_node,
                         target_edge_index=test_edges, target_edge_type=test_etypes,
                         num_relations=num_relations * 2, num_time=num_time,
                         time_type=trainval_ttypes_bi, target_time_type=test_ttypes)

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])


class MsgAwareForecastDataset(TransductiveTemporalForecastDataset):
    """Fully-inductive temporal KG dataset with DISJOINT G_tr / G_inf vocabs.

    Direct port of TTRIX's InductiveTemporalDatasetINGRAM
    (src/trix/datasets.py:1989). For the IndT sweep variants
    (WIKIIndT_*, GDELTIndT_*, ICEWS*IndT_*) with INGRAM's 4-file layout:

        train.txt -- G_tr training quadruples
        msg.txt   -- G_inf observed graph (disjoint vocab from G_tr)
        valid.txt -- G_tr held-out validation queries (transductive valid)
        test.txt  -- G_inf inductive test queries

    Per-split Data:
        train_data: edge_index = train.txt bidirectional (G_tr vocab)
                    target_* = train.txt
                    num_nodes = |G_tr|,  num_relations = num_train_rels*2
        valid_data: edge_index = train.txt bidirectional (G_tr vocab)
                    target_* = valid.txt
                    num_nodes = |G_tr|,  num_relations = num_train_rels*2
        test_data:  edge_index = msg.txt bidirectional (G_inf vocab, disjoint)
                    target_* = test.txt
                    num_nodes = |G_inf|, num_relations = num_inf_rels*2

    Time offsets: shared min-date across all four files, so time_type
    values live in a common absolute-time index space even though the
    entity/relation vocabularies are disjoint.
    """

    @property
    def raw_file_names(self):
        # Match TTRIX order at src/trix/datasets.py:2141.
        return ["train.txt", "msg.txt", "valid.txt", "test.txt"]

    @property
    def processed_dir(self):
        # Fully-inductive shape -> distinct cache from parent forecast class.
        return os.path.join(self.root, self.name, "processed_forecast_msg")

    def _parse_date(self, s):
        """Return integer ordinal for a timestamp string.

        Default: try int (handles zero-padded integer strings like WIKI's
        "092"), fall back to ISO date parsing (GDELT-style "2018-07-31").
        Subclasses override for other formats.
        """
        try:
            return int(s)
        except (ValueError, TypeError):
            from datetime import datetime
            return int(datetime.strptime(s, "%Y-%m-%d").toordinal())

    def _load_indt_file(self, path, ent_vocab=None, rel_vocab=None):
        """Load a 4-column (h r t date) file. Returns triplets + raw timestamps."""
        ent_vocab = dict(ent_vocab) if ent_vocab else {}
        rel_vocab = dict(rel_vocab) if rel_vocab else {}
        ec, rc = len(ent_vocab), len(rel_vocab)
        triplets, timestamps = [], []
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                parts = line.rstrip("\n").split(self.delimiter)
                if len(parts) < 4:
                    continue
                u, r, v, ts = parts[0], parts[1], parts[2], parts[3]
                if u not in ent_vocab:
                    ent_vocab[u] = ec; ec += 1
                if v not in ent_vocab:
                    ent_vocab[v] = ec; ec += 1
                if r not in rel_vocab:
                    rel_vocab[r] = rc; rc += 1
                triplets.append((ent_vocab[u], ent_vocab[v], rel_vocab[r]))
                timestamps.append(ts)
        return {
            "triplets": triplets,
            "timestamps": timestamps,
            "num_node": len(ent_vocab),
            "num_relation": len(rel_vocab),
            "inv_entity_vocab": ent_vocab,
            "inv_rel_vocab": rel_vocab,
        }

    def process(self):
        train_path, msg_path, valid_path, test_path = self.raw_paths[:4]

        if not os.path.exists(msg_path):
            # No msg.txt -> degrade to parent class (train+valid MP prefix).
            super().process()
            return

        # G_tr: train.txt fresh; valid.txt extends the same vocab
        train_res = self._load_indt_file(train_path)
        valid_res = self._load_indt_file(
            valid_path,
            ent_vocab=train_res["inv_entity_vocab"],
            rel_vocab=train_res["inv_rel_vocab"],
        )
        # G_inf: msg.txt fresh (DISJOINT from G_tr); test.txt extends it
        inf_res = self._load_indt_file(msg_path)
        test_res = self._load_indt_file(
            test_path,
            ent_vocab=inf_res["inv_entity_vocab"],
            rel_vocab=inf_res["inv_rel_vocab"],
        )

        num_train_nodes = valid_res["num_node"]
        num_train_rels = valid_res["num_relation"]
        num_inf_nodes = test_res["num_node"]
        num_inf_rels = test_res["num_relation"]

        # Shared min-date offset across all four files -> aligned time axis.
        all_dates = (train_res["timestamps"] + valid_res["timestamps"]
                     + inf_res["timestamps"] + test_res["timestamps"])
        parsed = [self._parse_date(d) for d in all_dates]
        min_ord = min(parsed)
        max_ord = max(parsed)
        num_time = int(max_ord - min_ord + 1)

        def to_t(stamps):
            return torch.tensor([self._parse_date(d) - min_ord for d in stamps], dtype=torch.long)

        train_t = to_t(train_res["timestamps"])
        valid_t = to_t(valid_res["timestamps"])
        inf_t = to_t(inf_res["timestamps"])
        test_t = to_t(test_res["timestamps"])

        # G_tr message-passing graph = train.txt bidirectional
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_res["triplets"]], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_res["triplets"]])
        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])
        train_fact_time = torch.cat([train_t, train_t])

        # G_inf message-passing graph = msg.txt bidirectional (disjoint IDs)
        inf_edge_index = torch.tensor([[t[0], t[1]] for t in inf_res["triplets"]], dtype=torch.long).t()
        inf_edge_index_bi = torch.cat([inf_edge_index, inf_edge_index.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_res["triplets"]])
        inf_etypes_bi = torch.cat([inf_etypes, inf_etypes + num_inf_rels])
        inf_time_bi = torch.cat([inf_t, inf_t])

        valid_q = torch.tensor(valid_res["triplets"], dtype=torch.long)  # (n, 3) as (h, t, r)
        test_q = torch.tensor(test_res["triplets"], dtype=torch.long)

        # Wrap num_time so InMemoryDataset.collate preserves it as an attribute.
        num_time_t = torch.tensor([num_time])

        train_data = Data(
            edge_index=train_fact_index, edge_type=train_fact_type,
            num_nodes=num_train_nodes,
            target_edge_index=train_target_edges, target_edge_type=train_target_etypes,
            num_relations=num_train_rels * 2, num_time=num_time_t,
            time_type=train_fact_time, target_time_type=train_t,
        )
        valid_data = Data(
            edge_index=train_fact_index, edge_type=train_fact_type,
            num_nodes=num_train_nodes,
            target_edge_index=valid_q[:, :2].T, target_edge_type=valid_q[:, 2],
            num_relations=num_train_rels * 2, num_time=num_time_t,
            time_type=train_fact_time, target_time_type=valid_t,
        )
        test_data = Data(
            edge_index=inf_edge_index_bi, edge_type=inf_etypes_bi,
            num_nodes=num_inf_nodes,
            target_edge_index=test_q[:, :2].T, target_edge_type=test_q[:, 2],
            num_relations=num_inf_rels * 2, num_time=num_time_t,
            time_type=inf_time_bi, target_time_type=test_t,
        )

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])


# --- Chronological-split base datasets ------------------------------------

class TemporalYAGO(TransductiveTemporalForecastDataset):
    name = "yago"
    delimiter = "\t"

class TemporalICEWS18(TransductiveTemporalForecastDataset):
    name = "icews18"
    delimiter = "\t"

class TemporalWIKI(TransductiveTemporalForecastDataset):
    name = "wiki"
    delimiter = "\t"

# Forecast variants that load from the shipped kg-datasets/<name>/raw dirs
# (same file layout as YAGOInd / ICEWS18Ind / WIKIInd but with proper
# train+valid MP prefix + calendar-order time indices).
class YAGOIndForecast(TransductiveTemporalForecastDataset):
    name = "YAGOInd"
    delimiter = "\t"

class ICEWS18IndForecast(TransductiveTemporalForecastDataset):
    name = "ICEWS18Ind"
    delimiter = "\t"

class WIKIIndForecast(TransductiveTemporalForecastDataset):
    name = "WIKIInd"
    delimiter = "\t"


# --- IndT sweep variants (built from TTRIX's dataset construction) --------

class WIKIIndT_25_inter(MsgAwareForecastDataset):
    name = "WIKIIndT_25_inter"
    delimiter = "\t"

class WIKIIndT_25_extra(MsgAwareForecastDataset):
    name = "WIKIIndT_25_extra"
    delimiter = "\t"

class WIKIIndT_50_inter(MsgAwareForecastDataset):
    name = "WIKIIndT_50_inter"
    delimiter = "\t"

class WIKIIndT_50_extra(MsgAwareForecastDataset):
    name = "WIKIIndT_50_extra"
    delimiter = "\t"

class WIKIIndT_75_inter(MsgAwareForecastDataset):
    name = "WIKIIndT_75_inter"
    delimiter = "\t"

class WIKIIndT_75_extra(MsgAwareForecastDataset):
    name = "WIKIIndT_75_extra"
    delimiter = "\t"

class WIKIIndT_100_inter(MsgAwareForecastDataset):
    name = "WIKIIndT_100_inter"
    delimiter = "\t"

class WIKIIndT_100_extra(MsgAwareForecastDataset):
    name = "WIKIIndT_100_extra"
    delimiter = "\t"


class GDELTIndT_25_inter(MsgAwareForecastDataset):
    name = "GDELTIndT_25_inter"
    delimiter = "\t"

class GDELTIndT_25_extra(MsgAwareForecastDataset):
    name = "GDELTIndT_25_extra"
    delimiter = "\t"

class GDELTIndT_50_inter(MsgAwareForecastDataset):
    name = "GDELTIndT_50_inter"
    delimiter = "\t"

class GDELTIndT_50_extra(MsgAwareForecastDataset):
    name = "GDELTIndT_50_extra"
    delimiter = "\t"

class GDELTIndT_75_inter(MsgAwareForecastDataset):
    name = "GDELTIndT_75_inter"
    delimiter = "\t"

class GDELTIndT_75_extra(MsgAwareForecastDataset):
    name = "GDELTIndT_75_extra"
    delimiter = "\t"

class GDELTIndT_100_inter(MsgAwareForecastDataset):
    name = "GDELTIndT_100_inter"
    delimiter = "\t"

class GDELTIndT_100_extra(MsgAwareForecastDataset):
    name = "GDELTIndT_100_extra"
    delimiter = "\t"

class GDELTIndT_100(MsgAwareForecastDataset):
    name = "GDELTIndT_100"
    delimiter = "\t"


class ICEWS14IndT_100(MsgAwareForecastDataset):
    name = "ICEWS14IndT_100"
    delimiter = "\t"

class ICEWS0515IndT_100(MsgAwareForecastDataset):
    name = "ICEWS0515IndT_100"
    delimiter = "\t"


class InductiveDataset(InMemoryDataset):

    delimiter = None
    # some datasets (4 from Hamaguchi et al and Indigo) have validation set based off the train graph, not inference
    valid_on_inf = True  # 
    
    def __init__(self, root, version, transform=None, pre_transform=build_relation_graph, **kwargs):

        self.version = str(version)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        for url, path in zip(self.urls, self.raw_paths):
            download_path = download_url(url % self.version, self.raw_dir)
            os.rename(download_path, path)
    
    def load_file(self, triplet_file, inv_entity_vocab={}, inv_rel_vocab={}):

        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, "r", encoding="utf-8") as fin:
            for l in fin:
                u, r, v = l.split() if self.delimiter is None else l.strip().split(self.delimiter)
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab), #entity_cnt,
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab
        }
    
    def process(self):
        
        train_files = self.raw_paths[:4]

        train_res = self.load_file(train_files[0], inv_entity_vocab={}, inv_rel_vocab={})
        inference_res = self.load_file(train_files[1], inv_entity_vocab={}, inv_rel_vocab={})
        valid_res = self.load_file(
            train_files[2], 
            inference_res["inv_entity_vocab"] if self.valid_on_inf else train_res["inv_entity_vocab"], 
            inference_res["inv_rel_vocab"] if self.valid_on_inf else train_res["inv_rel_vocab"]
        )
        test_res = self.load_file(train_files[3], inference_res["inv_entity_vocab"], inference_res["inv_rel_vocab"])

        num_train_nodes, num_train_rels = train_res["num_node"], train_res["num_relation"]
        inference_num_nodes, inference_num_rels = test_res["num_node"], test_res["num_relation"]

        train_edges, inf_graph, inf_valid_edges, inf_test_edges = train_res["triplets"], inference_res["triplets"], valid_res["triplets"], test_res["triplets"]
        
        train_target_edges = torch.tensor([[t[0], t[1]] for t in train_edges], dtype=torch.long).t()
        train_target_etypes = torch.tensor([t[2] for t in train_edges])

        train_fact_index = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_fact_type = torch.cat([train_target_etypes, train_target_etypes + num_train_rels])

        inf_edges = torch.tensor([[t[0], t[1]] for t in inf_graph], dtype=torch.long).t()
        inf_edges = torch.cat([inf_edges, inf_edges.flip(0)], dim=1)
        inf_etypes = torch.tensor([t[2] for t in inf_graph])
        inf_etypes = torch.cat([inf_etypes, inf_etypes + inference_num_rels])
        
        inf_valid_edges = torch.tensor(inf_valid_edges, dtype=torch.long)
        inf_test_edges = torch.tensor(inf_test_edges, dtype=torch.long)

        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=num_train_nodes,
                          target_edge_index=train_target_edges, target_edge_type=train_target_etypes, num_relations=num_train_rels*2)
        valid_data = Data(edge_index=inf_edges if self.valid_on_inf else train_fact_index, 
                          edge_type=inf_etypes if self.valid_on_inf else train_fact_type, 
                          num_nodes=inference_num_nodes if self.valid_on_inf else num_train_nodes,
                          target_edge_index=inf_valid_edges[:, :2].T, 
                          target_edge_type=inf_valid_edges[:, 2], 
                          num_relations=inference_num_rels*2 if self.valid_on_inf else num_train_rels*2)
        test_data = Data(edge_index=inf_edges, edge_type=inf_etypes, num_nodes=inference_num_nodes,
                         target_edge_index=inf_test_edges[:, :2].T, target_edge_type=inf_test_edges[:, 2], num_relations=inference_num_rels*2)

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])
    
    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, self.version, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, self.version, "processed")
    
    @property
    def raw_file_names(self):
        return [
            "transductive_train.txt", "inference_graph.txt", "inf_valid.txt", "inf_test.txt"
        ]

    @property
    def processed_file_names(self):
        return "data.pt"

    def __repr__(self):
        return "%s(%s)" % (self.name, self.version)
