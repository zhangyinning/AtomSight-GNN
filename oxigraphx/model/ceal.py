# This is the implementation of CEAL layer

import torch
from torch.nn import ReLU, ModuleList, Sequential
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MessagePassing, BatchNorm, global_add_pool, global_mean_pool
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter
from torch_geometric.nn.inits import reset
from torch_geometric.utils import degree

from utils import *
from .aggregators import AGGREGATORS
from .scalers import SCALERS

##########################################################################

class CEALConv(MessagePassing):
   def __init__(self, 
                in_channels, 
                out_channels, 
                aggregators,
                scalers,
                deg,
                edge_dim = None,
                towers = 1,
                pre_layers = 1,
                post_layers = 1,
                divide_input = False,
                **kwargs):
      """
        in_channels: dimension of initial node embedding
        out_challens: dimension of node message (after linear transform)
        weights: the weights applied to aggregated messages 
                 (it can be graph-level or node-level)
      """

      # Set the node_dim=0, and aggr='None'(b/c we will define our own aggregate() method)
      """
      why node_dim=0:
      The node_dim parameter specifies the axis along which the node features are expected 
      to be present in the x tensor. 
      For example, if you have a node feature tensor x with shape (num_nodes, input_dim) 
      where num_nodes is the number of nodes in the graph and input_dim is the dimension 
      of the node features, you would set node_dim=0 because the node features are present 
      along the first dimension.
      """
      super().__init__(node_dim=0, aggr=None)

      if divide_input:
         assert in_channels % towers == 0
      assert out_channels % towers == 0

      self.in_channels = in_channels
      self.out_channels = out_channels
      self.aggregators = aggregators
      self.scalers = scalers
      self.edge_dim = edge_dim
      self.towers = towers
      self.divide_input = divide_input

      self.F_in = in_channels // towers if divide_input else in_channels
      self.F_out = self.out_channels // towers


      # If edge_dim is not None, set up the edge attribute encoder
      if self.edge_dim is not None:
         """
          It uses the torch_geometric.nn.dense.linear.Linear
          The input dimension is the edge attribute dim, 
          The output dimension=in_channels (node feature dimension)
          Need to find why edge feature dim is set to node feature dim?
         """
         self.edge_encoder = Linear(edge_dim, self.F_in)
 
      # Pre/Post MLPs
      self.pre_nns = ModuleList()
      self.post_nns = ModuleList()
      for _ in range(towers):
         modules = [Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
         for _ in range(pre_layers - 1):
            modules += [ReLU()]
            modules += [Linear(self.F_in, self.F_in)]
         self.pre_nns.append(Sequential(*modules))
      
         in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in
            
         modules = [Linear(in_channels, self.F_out)]
         for _ in range(post_layers - 1):
            modules += [ReLU()]
            modules += [Linear(self.F_out, self.F_out)]
         self.post_nns.append(Sequential(*modules))
      
      self.lin = Linear(out_channels, out_channels)

      
      # One MLP for each aggregator        
      factor = 1
      self.mlp_w0 = Sequential( torch.nn.Linear(self.F_in, round(self.F_in*factor)),
                            ReLU(),
                            torch.nn.Linear(round(self.F_in*factor), self.F_in)
                            ) 
      self.mlp_w1 = Sequential( torch.nn.Linear(self.F_in, round(self.F_in*factor)),
                            ReLU(),
                            torch.nn.Linear(round(self.F_in*factor), self.F_in)
                            )
      self.mlp_w2 = Sequential( torch.nn.Linear(self.F_in, round(self.F_in*factor)),
                            ReLU(),
                            torch.nn.Linear(round(self.F_in*factor), self.F_in)
                            )
      self.mlp_w3 = Sequential( torch.nn.Linear(self.F_in, round(self.F_in*factor)),
                            ReLU(),
                            torch.nn.Linear(round(self.F_in*factor), self.F_in)
                            )
      self.mlp_w4 = Sequential( torch.nn.Linear(self.F_in, round(self.F_in*factor)),
                            ReLU(),
                            torch.nn.Linear(round(self.F_in*factor), self.F_in)
                                 )
      


      """
      Reset parameters
       
      It is typically used after creating an instance of a Graph Neural Network (GNN) model 
      to ensure that the model's parameters are initialized in a consistent manner before 
      training or inference.
      """
      self.reset_parameters()

   def reset_parameters(self):
      if self.edge_dim is not None:
         self.edge_encoder.reset_parameters()
      for nn in self.pre_nns:
         reset(nn)
      for nn in self.post_nns:
         reset(nn)
      self.lin.reset_parameters()

      self.mlp_w0[0].reset_parameters()
      self.mlp_w1[0].reset_parameters()
      self.mlp_w2[0].reset_parameters()
      self.mlp_w3[0].reset_parameters()
      self.mlp_w4[0].reset_parameters()
   
      
   """
   The forward method defines the framwork of a layer's logic.
     Details of how messages are generated are in method message()
     Details of how messages are aggregated are in method aggregate()
     Details of how node embeddings are updated are in method update()
   """
   def forward(self, x, edge_index, batch, edge_attr=None):
      #edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

      if self.divide_input:
         x = x.view(-1, self.towers, self.F_in)
      else:
         x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

      # The progagate() method is the center of a message passing layer
      out = self.propagate(edge_index, x=x, edge_attr=edge_attr, weights=None, batch=batch, size=None)

      # Put the out from the graph part of the layer to a post-MLP
      out = torch.cat([x, out], dim=-1)
      outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns) ]
      out = torch.cat(outs, dim=1)

      # Why another linear layer here?
      return self.lin(out)


   """
   The message() methods
     1. Collects information from the constructor + kwargs in propagate. e.g x_j, edge_attr, size
     2. Construct central node i's messages by using variables suffixed with _i, _j
   """
   def message(self, x_i, x_j, edge_attr):
   
      # Concatenate central node i and neighbour nodes j and (if any) edge features to form messages
      h: Tensor = x_i  # Dummy.
      if edge_attr is not None:
         edge_attr = self.edge_encoder(edge_attr)
         edge_attr = edge_attr.view(-1, 1, self.F_in)
         edge_attr = edge_attr.repeat(1, self.towers, 1)
         h = torch.cat([x_i, x_j, edge_attr], dim=-1)
      else:
         h = torch.cat([x_i, x_j], dim=-1)
      # Run the messages through the pre-processing MLP
      hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
      return torch.stack(hs, dim=1)


   """
   This is the heart of ChemGNN algorithm.
   Normally, a single aggregator is defined by the argument to the constructor of the MessagePassing class.
   In ChemGNN, multiple aggregators are utilized to aggregate messages.
   The output of each aggregator has a weight (graph-level) or weighits (node-level, num_weights=num_nodes).
   In the current implement, it only shows the graph-level weights. It's easy to extend to node-level weights.
   """
   def aggregate(self, inputs, index, weights, batch, dim_size = None):
      outs = []

      for aggregator in self.aggregators:
        if aggregator == 'sum':
            a = scatter(inputs, index, 0, None, dim_size, reduce='sum')
            a_size = a.size()

            w0 = self.mlp_w0(a[:, 0])
            out = w0 * a[:,0]   
            out = out.view(a_size[0], a_size[1], a_size[2])
        elif aggregator == 'mean':
            a = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            a_size = a.size()

            w1 = self.mlp_w1(a[:, 0])
            out = w1 * a[:,0]               
            out = out.view(a_size[0], a_size[1], a_size[2])
        elif aggregator == 'min':
            a = scatter(inputs, index, 0, None, dim_size, reduce='min')
            a_size = a.size()

            w2 = self.mlp_w2(a[:, 0])
            out = w2 * a[:,0]
            out = out.view(a_size[0], a_size[1], a_size[2])
        elif aggregator == 'max':
            a = scatter(inputs, index, 0, None, dim_size, reduce='max')
            a_size = a.size()

            w3 = self.mlp_w3(a[:, 0])
            out = w3 * a[:,0]

            out = out.view(a_size[0], a_size[1], a_size[2])
        elif aggregator == 'var' or aggregator == 'std':
            mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            mean_squares = scatter(inputs * inputs, index, 0, None,
                                        dim_size, reduce='mean')
            a = (mean_squares - mean * mean)
            a_size = a.size()

            w4 = self.mlp_w4(a[:, 0])
            out = w4 * a[:,0]
   
            out = out.view(a_size[0], a_size[1], a_size[2])

            if aggregator == 'std':
                out = torch.sqrt(torch.relu(out) + 1e-5)
        else:
            raise ValueError(f'Unknown aggregator "{aggregator}".')
        outs.append(out)


      out = torch.cat(outs, dim=-1)

      deg = degree(index, dim_size, dtype=inputs.dtype)
      deg = deg.view(-1, 1, 1)
      # deg = deg.clamp_(1).view(-1, 1, 1)

      outs = []
      for scaler in self.scalers:
         if scaler == 'identity':
            pass
         elif scaler == 'amplification':
            out = out * (torch.log(deg + 1) / self.avg_deg['log'])
         elif scaler == 'attenuation':
            out = out * (self.avg_deg['log'] / torch.log(deg + 1))
         elif scaler == 'linear':
            out = out * (deg / self.avg_deg['lin'])
         elif scaler == 'inverse_linear':
            out = out * (self.avg_deg['lin'] / deg)
         else:
            raise ValueError(f'Unknown scaler "{scaler}".')
         outs.append(out)

      return torch.cat(outs, dim=-1)

   def __repr__(self):
      return (f'{self.__class__.__name__}({self.in_channels}, '
              f'{self.out_channels}, towers={self.towers}, '
              f'edge_dim={self.edge_dim})')












































