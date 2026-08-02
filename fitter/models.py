import torch
from torch import nn
from . import tasks, layers
from .base_nbfnet import BaseNBFNet
import torch_geometric
from .tasks import build_relation_graph
import math

class FITTER(nn.Module):

    def __init__(self, rel_model_cfg, entity_model_cfg, rule_model_cfg=None, dataset_cfg=None):
        super(FITTER, self).__init__()

        self.relation_model = RelNBFNet(**rel_model_cfg)
        self.entity_model = EntityNBFNet(**entity_model_cfg)
        self.window_size = rel_model_cfg['window_size']
        self.alpha = entity_model_cfg['alpha']
        self.inductive = dataset_cfg['class']
        
    def forward(self, data, batch):
        
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        query_rels = batch[:, 0, 2]
        query_times = batch[:, 0, 3]
        if self.alpha == 1:
            score = 0
        else:
            relation_representations = self.relation_model(data.relation_graph, query=query_rels)
            # here we compute global quadruple representation
            if 'Ind' in self.inductive:
                entity_graph_global, relation_graph_t = self.generate_graph_global(data, query_times)
                output_t = []
                score_t = []
                for i in range(len(entity_graph_global)):
                    if 'ICEWS18Ind' in self.inductive:
                        relation_representations_t = relation_representations[i, :].unsqueeze(0)
                    elif 'Ind' in self.inductive:
                        relation_representations_t = self.relation_model(entity_graph_global[i].relation_graph,
                                                                         query=query_rels[i])
                    else:
                        relation_representations_t = relation_representations[i, :].unsqueeze(0)
                    score_t_ind, output_t_ind = self.entity_model(entity_graph_global[i], relation_representations_t,
                                                                  batch[i, :].unsqueeze(0))
                    output_t.append(output_t_ind)
                    score_t.append(score_t_ind)
                output = torch.stack(output_t).squeeze(dim=1)
                score = torch.stack(score_t).squeeze(dim=1)
            else:
                score, output = self.entity_model(data, relation_representations, batch)
        #here we construct the local graph and calculate the local quadruple representation
        if self.window_size >= 0:
            entity_graph_t, relation_graph_t = self.generate_graph_t(data, query_times, self.window_size)
            output_t = []
            score_t = []
            for i in range(len(entity_graph_t)):
                # if we are predicting temporal extrapolation dataset, we only build relation graph on local graph
                if 'Ind' in self.inductive:
                    relation_representations_t = self.relation_model(entity_graph_t[i].relation_graph, query=query_rels[i])
                else:
                #else, we use the relation graph on the whole observed set
                    relation_representations_t = relation_representations[i,:].unsqueeze(0)
                score_t_ind, output_t_ind = self.entity_model(entity_graph_t[i], relation_representations_t, batch[i,:].unsqueeze(0))
                output_t.append(output_t_ind)
                score_t.append(score_t_ind)
            output_t = torch.stack(output_t).squeeze(dim=1)
            score_t = torch.stack(score_t).squeeze(dim=1)
            # we use alpha to balance the local quadruple representation and global quadruple representation
            score = score_t*self.alpha + score * (1-self.alpha)
        return score

    def generate_graph_t(self, data, times, window_size=3):
        # we only use local graph before the query time for temporal extrapolation task to avoid leakage
        if 'Ind' in self.inductive:
            time_start = times - window_size - 1
            time_end = times - 1
        else:
            time_start = times - window_size
            time_end = times + window_size

        relation_graph_t = []
        entity_graph_t = []
        for i in range(times.shape[0]):
            index = torch.ge(data.time_type, time_start[i]) & torch.le(data.time_type, time_end[i])
            edge_subset = data.edge_index[:,index]
            edge_type_subset=data.edge_type[index]
            time_type_subset = data.time_type[index]

            graph_t = torch_geometric.data.Data(edge_index=edge_subset, edge_type= edge_type_subset,
                                                                                    num_nodes=data.num_nodes,
                                                                                    num_relations=data.num_relations,time_type=time_type_subset,
                                                                                    num_time=data.num_time)
            graph_t = build_relation_graph(graph_t)
            entity_graph_t.append(graph_t)
            relation_graph_t.append(graph_t.relation_graph)

        return entity_graph_t, relation_graph_t

    def generate_graph_global(self, data, times):
        # if 'Ind' in self.inductive:
        #     time_max = max(data.time_type)
        #     time_start = torch.full_like(times, time_max-2*window_size)
        #     time_end = torch.full_like(times,time_max)
        if 'Ind' in self.inductive:
            time_start = torch.full_like(times, 0)
            time_end = times - 1

        relation_graph_t = []
        entity_graph_t = []
        for i in range(times.shape[0]):
            if isinstance(data.time_type, list):
                edge_subset = []
                edge_type_subset = []
                time_type_subset = []

                start = time_start[i].item()
                end = time_end[i].item()
                for j in range(len(data.time_type)):
                    filtered_list = [t for t in data.time_type[j] if start <= t <= end]

                    if filtered_list:  # Only keep non-empty filtered lists and corresponding edges
                        time_type_subset.append(filtered_list)
                        edge_subset.append(data.edge_index[:,j])
                        edge_type_subset.append(data.edge_type[j])
                edge_subset = torch.stack(edge_subset, dim=1)
                edge_type_subset = torch.stack(edge_type_subset, dim=0)
            else:
                index = torch.ge(data.time_type, time_start[i]) & torch.le(data.time_type, time_end[i])
                edge_subset = data.edge_index[:,index]
                edge_type_subset=data.edge_type[index]
                time_type_subset = data.time_type[index]

            graph_t = torch_geometric.data.Data(edge_index=edge_subset, edge_type= edge_type_subset,
                                                                                    num_nodes=data.num_nodes,
                                                                                    num_relations=data.num_relations,time_type=time_type_subset,
                                                                                    num_time=data.num_time)
            graph_t = build_relation_graph(graph_t)
            entity_graph_t.append(graph_t)
            relation_graph_t.append(graph_t.relation_graph)

        return entity_graph_t, relation_graph_t
        # #train_data = Data(edge_index=train_edges, edge_type=train_etypes, num_nodes=num_node,
        #                   target_edge_index=train_target_edges, target_edge_type=train_target_etypes,
        #                   num_relations=num_relations * 2, num_time=num_time, time_type=train_ttypes, target_time_type=train_target_ttypes)

# NBFNet to work on the graph of relations with 4 fundamental interactions
# Doesn't have the final projection MLP from hidden dim -> 1, returns all node representations 
# of shape [bs, num_rel, hidden]
class RelNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=4, time_graph = 'null', window_size=0 , **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False)
                )
        self.time_graph = time_graph
        self.window_size = window_size

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            if time_graph != 'null':
                feature_dim = feature_dim + self.dims[0]
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )

        if 'ind' in self.time_graph:
            self.layers_t = nn.ModuleList()
            for i in range(len(self.dims) - 1):
                self.layers_t.append(
                    layers.GeneralizedRelationalConv(
                        self.dims[i], self.dims[i + 1], num_relation,
                        self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                        self.activation, dependent=False)
                )
    
    def bellmanford(self, data, h_index, nbf_layers='static', separate_grad=False):
        try:
            batch_size = len(h_index)
        except:
            batch_size = 1
        # initialize initial nodes (relations of interest in the batcj) with all ones
        query = torch.ones(batch_size, self.dims[0], device=h_index.device, dtype=torch.float)
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        #boundary = torch.zeros(data.num_nodes, *query.shape, device=h_index.device)
        # Indicator function: by the scatter operation we put ones as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        if nbf_layers == 'static':
            for layer in self.layers:
                # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
                hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
                if self.short_cut and hidden.shape == layer_input.shape:
                    # residual connection here
                    hidden = hidden + layer_input
                hiddens.append(hidden)
                edge_weights.append(edge_weight)
                layer_input = hidden
        else:
            for layer in self.layers_t:
                # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
                hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
                if self.short_cut and hidden.shape == layer_input.shape:
                    # residual connection here
                    hidden = hidden + layer_input
                hiddens.append(hidden)
                edge_weights.append(edge_weight)
                layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, rel_graph, query, relation_graph_t=None):

        # message passing and updated node representations (that are in fact relations)
        output = self.bellmanford(rel_graph, h_index=query)["node_feature"]  # (batch_size, num_nodes, hidden_dim）
        return output

class EntityNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, use_time='null', num_relation=1, num_time=365, remove_edge='default', project_times=True, boundary='default', time_dependent=False, **kwargs):

        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        self.use_time = use_time
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True, time_dependent=time_dependent, project_times=project_times, num_time=num_time)
            )
        feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim
        if 'concat' in self.use_time:
            feature_dim = feature_dim + self.dims[0]
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)
        self.remove_edge = remove_edge
        self.project_times = project_times
        self.boundary = boundary
        self.num_time = num_time
        self.time_dependent = time_dependent

        if 'nbf' in self.use_time:
            if self.time_dependent or not self.project_times:
                # relation embeddings as an independent embedding matrix per each layer
                self.time_projection = nn.Embedding(self.num_time, input_dim)
            else:
                # will be initialized after the pass over relation graph
                self.time_projection = nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.ReLU(),
                    nn.Linear(input_dim, input_dim)
                )
    
    def bellmanford(self, data, h_index, r_index, time_index=None, separate_grad=False):
        try:
            batch_size = len(h_index)
        except:
            batch_size = 1

        # initialize queries (relation types of the given triples)
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        if self.boundary == 'default':
            boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))

        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:

            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight=edge_weight, time_type=data.time_type)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
         }

    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0, device='cpu'):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end.item(), device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cos = torch.cos(freqs)  # real part
        freqs_sin = torch.sin(freqs)  # imaginary part
        return freqs_cos.to(device), freqs_sin.to(device)

    def precompute_trans_pe(self, dim:int, end:int, device='cpu'):
        pe = torch.zeros(end, dim)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        position = torch.arange(0, int(end), dtype=torch.float).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.to(device)

    def forward(self, data, relation_representations, batch, entity_graph_t=None):
        if batch.shape[2] == 3:
            h_index, t_index, r_index = batch.unbind(-1)
        else:
            h_index, t_index, r_index, time_index = batch.unbind(-1)

        # initial query representations are those from the relation graph
        self.query = relation_representations
        freqs_cos, freqs_sin = self.precompute_freqs_cis(self.dims[0], data.num_time)
        #self.time_query = torch.cat([freqs_cos, freqs_sin], dim=-1).expand(batch.shape[0], -1, -1).to(batch.device)
        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        if 'add' in self.use_time:
            pe = self.precompute_trans_pe(self.dims[0],data.num_time,time_index.device)
            time_query = pe.expand(batch.shape[0], -1, -1).to(batch.device)
            self.time_query = self.time_projection(time_query)
            for layer in self.layers:
                layer.time = self.time_query
        else:
            time_query = torch.cat([freqs_cos,freqs_sin],dim=-1).expand(batch.shape[0], -1, -1).to(batch.device)
            self.time_query = self.time_projection(time_query)
            for layer in self.layers:
                layer.time = self.time_query

        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            if self.remove_edge == 'time':
                data = self.remove_easy_edges(data, h_index, t_index, r_index, time_index=time_index)
            elif self.remove_edge == 'default':
                data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0],
                                      time_index=time_index[:, 0])["node_feature"]  # (num_nodes, batch_size, feature_dim）

        feature = output
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        if 'concat' in self.use_time:
            freqs_cos, freqs_sin = self.precompute_freqs_cis(self.dims[0], data.num_time,device=time_index.device)
            # freqs_cos = freqs_cos.to(time_index.device)
            # freqs_sin = freqs_sin.to(time_index.device)
            freqs_cos, freqs_sin = freqs_cos[time_index], freqs_sin[time_index]
            feature = torch.cat([feature,freqs_cos,freqs_sin],dim=-1)
        elif 'add' in self.use_time:
            #freqs_cos, freqs_sin = self.precompute_freqs_cis(self.dims[0], data.num_time,device=time_index.device)
            pe = self.precompute_trans_pe(self.dims[0]*2, data.num_time, time_index.device)
            feature = feature + pe[time_index]
        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape), output


    

    


