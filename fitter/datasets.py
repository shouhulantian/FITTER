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
                if len(t_instance) == 5 and t_instance[4] not in inv_time_vocab:
                    inv_time_vocab[t_instance[4]] = time_cnt
                    time_cnt += 1
                if len(t_instance) == 4:
                    u, r, v, t1 = inv_entity_vocab[t_instance[0]], inv_rel_vocab[t_instance[1]], inv_entity_vocab[t_instance[2]], inv_time_vocab[t_instance[3]]
                    quadruples.append((u, v, r, t1))
                else:
                    u, r, v, t1, t2 = inv_entity_vocab[t_instance[0]], inv_rel_vocab[t_instance[1]],inv_entity_vocab[t_instance[2]], inv_time_vocab[t_instance[3]], inv_time_vocab[t_instance[4]]
                    quadruples.append((u, v, r, t1, t2))

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
                if len(t_instance) == 5 and t_instance[4] not in inv_time_vocab:
                    inv_time_vocab[t_instance[4]] = time_cnt
                    time_cnt += 1
                if len(t_instance) == 4:
                    u, r, v, t1 = inv_entity_vocab[t_instance[0]], inv_rel_vocab[t_instance[1]], inv_entity_vocab[t_instance[2]], inv_time_vocab[t_instance[3]]
                    quadruples.append((u, v, r, t1))
                else:
                    u, r, v, t1, t2 = inv_entity_vocab[t_instance[0]], inv_rel_vocab[t_instance[1]],inv_entity_vocab[t_instance[2]], inv_time_vocab[t_instance[3]], inv_time_vocab[t_instance[4]]
                    quadruples.append((u, v, r, t1, t2))

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
