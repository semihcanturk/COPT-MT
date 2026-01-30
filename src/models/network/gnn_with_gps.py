import copy
from typing import Optional, Union

import torch

from src.models.head import GNNEdgeHead, COPTGraphHead, COPTNodeHead, COPTInductiveNodeHead
from src.models.layer import GeneralMultiLayer
from src.models.layer.gps_layer import GPSLayer
from src.models.stage import GNNStackStage, GNNConcatStage


class GNNWithGPSStack(torch.nn.Module):
    """GNN model with an intermediate GraphGPS stack for transfer learning.
    
    This model is designed for transfer learning scenarios where:
    - The backbone (encoder, pre_mp, mp) is frozen from pretrained weights
    - The GPS stack (gt_stack) is trainable and incorporates global information
    - The prediction head (post_mp) is trainable for the target task
    
    Architecture: encoder -> pre_mp -> mp -> gt_stack -> post_mp
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_inner: int,
        head_type: str = 'inductive_node',
        encoder: Optional[torch.nn.Module] = None,
        conv: Optional[torch.nn.Module] = None,
        conv_ffn: bool = False,
        layers_pre_mp: int = 1,
        layers_mp: int = 5,
        layers_post_mp: int = 2,
        stage_type: str = 'skipsum',
        batch_norm: bool = True,
        l2_norm: bool = True,
        final_l2_norm: bool = True,
        gsn: bool = False,
        dropout: float = 0.2,
        has_act: bool = True,
        act: Optional[Union[str, torch.nn.Module]] = 'relu',
        last_act: Optional[Union[str, torch.nn.Module]] = 'sigmoid',
        edge_decoding: str = 'concat',
        graph_pooling: str = 'add',
        # GPS stack parameters
        gt_stack_layers: int = 3,
        gt_stack_local_model: Optional[torch.nn.Module] = None,
        gt_stack_self_attn: Optional[torch.nn.Module] = None,
        gt_stack_num_heads: int = 4,
        gt_stack_pna_degrees: Optional[list] = None,
        gt_stack_equivstable_pe: bool = False,
        gt_stack_dropout: float = 0.0,
        gt_stack_attn_dropout: float = 0.0,
        gt_stack_layer_norm: bool = False,
        gt_stack_batch_norm: bool = True,
        gt_stack_bigbird_cfg: Optional[dict] = None,
        gt_stack_log_attn_weights: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        dim_in = self.encoder.dim_in if encoder is not None else dim_in

        # Backbone: encoder -> pre_mp -> mp (will be frozen in transfer learning)
        if layers_pre_mp > 0:
            self.pre_mp = GeneralMultiLayer(
                layer='linear',
                in_dim=dim_in,
                out_dim=dim_inner,
                hid_dim=dim_inner,
                num_layers=layers_pre_mp,
                batch_norm=batch_norm,
                l2_norm=l2_norm,
                dropout=dropout,
                act=act,
                final_act=True,
            )
            dim_in = dim_inner
        if layers_mp > 0:
            self.mp = GNNStackStage(
                in_dim=dim_in,
                out_dim=dim_inner,
                num_layers=layers_mp,
                conv=conv,
                conv_ffn=conv_ffn,
                stage_type=stage_type,
                final_l2_norm=final_l2_norm,
                gsn=gsn,
                batch_norm=batch_norm,
                l2_norm=l2_norm,
                dropout=dropout,
                act=act if has_act else None,
            )

        # GPS stack: trainable intermediate transformer stack
        # Convert act to string for GPSLayer (if using string-based construction)
        if isinstance(act, torch.nn.Module):
            act_class_name = act.__class__.__name__
            if act_class_name == 'LeakyReLU':
                act_str = 'leaky_relu'
            elif act_class_name == 'ReLU':
                act_str = 'relu'
            elif act_class_name == 'ELU':
                act_str = 'elu'
            elif act_class_name == 'GELU':
                act_str = 'gelu'
            else:
                act_str = 'relu'
        else:
            act_str = act

        gt_layers = []
        for _ in range(gt_stack_layers):
            layer_local_model = copy.deepcopy(gt_stack_local_model) if gt_stack_local_model is not None else None
            layer_self_attn = copy.deepcopy(gt_stack_self_attn) if gt_stack_self_attn is not None else None

            gt_layers.append(GPSLayer(
                dim_h=dim_inner,
                local_model=layer_local_model,
                self_attn=layer_self_attn,
                equivstable_pe=gt_stack_equivstable_pe,
                dropout=gt_stack_dropout,
                attn_dropout=gt_stack_attn_dropout,
                layer_norm=gt_stack_layer_norm,
                batch_norm=gt_stack_batch_norm,
                log_attn_weights=gt_stack_log_attn_weights,
                act=act_str,
            ))
        self.gt_stack = torch.nn.Sequential(*gt_layers)

        # Prediction head: trainable task-specific head
        if head_type == 'inductive_node':
            GNNHead = COPTInductiveNodeHead
        elif head_type == 'node':
            GNNHead = COPTNodeHead
        elif head_type == 'edge':
            GNNHead = GNNEdgeHead
        elif head_type == 'graph':
            GNNHead = COPTGraphHead
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        self.post_mp = GNNHead(
            dim_in=dim_inner,
            dim_out=dim_out,
            dim_hid=dim_inner,
            layers_post_mp=layers_post_mp,
            batch_norm=batch_norm,
            l2_norm=l2_norm,
            act=act,
            dropout=dropout,
            last_act=last_act,
            graph_pooling=graph_pooling,
            edge_decoding=edge_decoding,
        )

        self.reset_parameters()

    def reset_parameters(self):
        from torch_geometric.graphgym.init import init_weights
        self.apply(init_weights)

    def reset_head(self):
        """Reset only the prediction head parameters."""
        from torch_geometric.graphgym.init import init_weights
        self.post_mp.apply(init_weights)

    def forward(self, batch):
        """Forward pass: encoder -> pre_mp -> mp -> gt_stack -> post_mp"""
        if hasattr(self, 'encoder') and self.encoder is not None:
            batch = self.encoder(batch)
        if hasattr(self, 'pre_mp'):
            batch = self.pre_mp(batch)
        if hasattr(self, 'mp'):
            batch = self.mp(batch)
        if hasattr(self, 'gt_stack'):
            batch = self.gt_stack(batch)
        batch = self.post_mp(batch)
        return batch


class HybridGNNWithGPSStack(torch.nn.Module):
    """HybridGNN model with an intermediate GraphGPS stack for transfer learning.
    
    This model is similar to GNNWithGPSStack but uses GNNConcatStage (like HybridGNN)
    which produces intermediate representations from each layer. The GPS stack is
    applied after combining these intermediate representations.
    
    This model is designed for transfer learning scenarios where:
    - The backbone (encoder, pre_mp, mp) is frozen from pretrained weights
    - The GPS stack (gt_stack) is trainable and incorporates global information
    - The prediction head (post_mp) is trainable for the target task
    
    Architecture: encoder -> pre_mp -> mp (GNNConcatStage) -> combine x_list -> gt_stack -> post_mp
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_inner: int,
        head_type: str = 'inductive_node',
        encoder: Optional[torch.nn.Module] = None,
        conv: Optional[torch.nn.Module] = None,
        conv_ffn: bool = False,
        layers_pre_mp: int = 1,
        layers_mp: int = 5,
        layers_post_mp: int = 2,
        stage_type: str = 'skipsum',
        batch_norm: bool = True,
        l2_norm: bool = True,
        final_l2_norm: bool = True,
        gsn: bool = False,
        dropout: float = 0.2,
        has_act: bool = True,
        act: Optional[Union[str, torch.nn.Module]] = 'relu',
        last_act: Optional[Union[str, torch.nn.Module]] = 'sigmoid',
        edge_decoding: str = 'concat',
        graph_pooling: str = 'add',
        hybrid_stage: str = 'concat',
        # GPS stack parameters
        gt_stack_layers: int = 3,
        gt_stack_local_model: Optional[torch.nn.Module] = None,
        gt_stack_self_attn: Optional[torch.nn.Module] = None,
        gt_stack_num_heads: int = 4,
        gt_stack_pna_degrees: Optional[list] = None,
        gt_stack_equivstable_pe: bool = False,
        gt_stack_dropout: float = 0.0,
        gt_stack_attn_dropout: float = 0.0,
        gt_stack_layer_norm: bool = False,
        gt_stack_batch_norm: bool = True,
        gt_stack_bigbird_cfg: Optional[dict] = None,
        gt_stack_log_attn_weights: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        dim_in = self.encoder.dim_in if encoder is not None else dim_in

        # Backbone: encoder -> pre_mp -> mp (will be frozen in transfer learning)
        if layers_pre_mp > 0:
            self.pre_mp = GeneralMultiLayer(
                layer='linear',
                in_dim=dim_in,
                out_dim=dim_inner,
                hid_dim=dim_inner,
                num_layers=layers_pre_mp,
                batch_norm=batch_norm,
                l2_norm=l2_norm,
                dropout=dropout,
                act=act,
                final_act=True,
            )
            dim_in = dim_inner
        if layers_mp > 0:
            self.mp = GNNConcatStage(
                in_dim=dim_in,
                out_dim=dim_inner,
                num_layers=layers_mp,
                conv=conv,
                conv_ffn=conv_ffn,
                stage_type=stage_type,
                final_l2_norm=final_l2_norm,
                gsn=gsn,
                batch_norm=batch_norm,
                l2_norm=l2_norm,
                dropout=dropout,
                act=act if has_act else None,
            )

        # Store hybrid_stage for combining x_list
        self.hybrid_stage = hybrid_stage
        
        # Calculate dimension after combining x_list from GNNConcatStage
        # GNNConcatStage outputs all have dimension dim_inner, but combining changes the dimension
        if layers_mp > 0:
            if hybrid_stage == 'sum':
                # Sum: stack and sum outputs (all dim_inner), result is dim_inner
                gt_stack_dim_in = dim_inner
            elif hybrid_stage == 'concat':
                # Concat: concatenate outputs (all dim_inner), result is layers_mp * dim_inner
                gt_stack_dim_in = layers_mp * dim_inner
            else:
                raise ValueError(f'Stage {hybrid_stage} is not supported.')
        else:
            gt_stack_dim_in = dim_inner
        
        # Projection layer if needed (when hybrid_stage='concat', dimension differs)
        if gt_stack_dim_in != dim_inner:
            self.gt_stack_proj = torch.nn.Linear(gt_stack_dim_in, dim_inner)
        else:
            self.gt_stack_proj = None

        # GPS stack: trainable intermediate transformer stack
        # Convert act to string for GPSLayer (if using string-based construction)
        if isinstance(act, torch.nn.Module):
            act_class_name = act.__class__.__name__
            if act_class_name == 'LeakyReLU':
                act_str = 'leaky_relu'
            elif act_class_name == 'ReLU':
                act_str = 'relu'
            elif act_class_name == 'ELU':
                act_str = 'elu'
            elif act_class_name == 'GELU':
                act_str = 'gelu'
            else:
                act_str = 'relu'
        else:
            act_str = act

        gt_layers = []
        for _ in range(gt_stack_layers):
            layer_local_model = copy.deepcopy(gt_stack_local_model) if gt_stack_local_model is not None else None
            layer_self_attn = copy.deepcopy(gt_stack_self_attn) if gt_stack_self_attn is not None else None

            gt_layers.append(GPSLayer(
                dim_h=dim_inner,
                local_model=layer_local_model,
                self_attn=layer_self_attn,
                equivstable_pe=gt_stack_equivstable_pe,
                dropout=gt_stack_dropout,
                attn_dropout=gt_stack_attn_dropout,
                layer_norm=gt_stack_layer_norm,
                batch_norm=gt_stack_batch_norm,
                log_attn_weights=gt_stack_log_attn_weights,
                act=act_str,
            ))
        # Store layers as a ModuleList so we can iterate and collect intermediate outputs
        self.gt_stack = torch.nn.ModuleList(gt_layers)

        # Prediction head: trainable task-specific head
        # Input dimension: concatenate all intermediate outputs from mp (x_list) and gt_stack (gt_x_list)
        # Each output in x_list and gt_x_list has dimension dim_inner
        if layers_mp > 0:
            num_mp_outputs = layers_mp
        else:
            num_mp_outputs = 0
        num_gt_outputs = gt_stack_layers
        # Total number of intermediate outputs to concatenate
        total_outputs = num_mp_outputs + num_gt_outputs
        post_mp_dim_in = total_outputs * dim_inner
        
        if head_type == 'inductive_node':
            GNNHead = COPTInductiveNodeHead
        elif head_type == 'node':
            GNNHead = COPTNodeHead
        elif head_type == 'edge':
            GNNHead = GNNEdgeHead
        elif head_type == 'graph':
            GNNHead = COPTGraphHead
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        self.post_mp = GNNHead(
            dim_in=post_mp_dim_in,  # Combined x_stack + gt_stack output dimension
            dim_out=dim_out,
            dim_hid=dim_inner,
            layers_post_mp=layers_post_mp,
            batch_norm=batch_norm,
            l2_norm=l2_norm,
            act=act,
            dropout=dropout,
            last_act=last_act,
            graph_pooling=graph_pooling,
            edge_decoding=edge_decoding,
        )

        self.reset_parameters()

    def reset_parameters(self):
        from torch_geometric.graphgym.init import init_weights
        self.apply(init_weights)

    def reset_head(self):
        """Reset only the prediction head parameters."""
        from torch_geometric.graphgym.init import init_weights
        self.post_mp.apply(init_weights)

    def forward(self, batch):
        """Forward pass: encoder -> pre_mp -> mp -> gt_stack -> concat(all intermediate outputs) -> post_mp"""
        if hasattr(self, 'encoder') and self.encoder is not None:
            batch = self.encoder(batch)
        if hasattr(self, 'pre_mp'):
            batch = self.pre_mp(batch)
        
        # Collect intermediate outputs from mp (GNNConcatStage)
        if hasattr(self, 'mp'):
            batch = self.mp(batch)
            # x_list contains intermediate outputs from each mp layer
            mp_x_list = torch.cat(batch.x_list, dim=-1)
        else:
            # If no mp, use current batch.x as single output
            mp_x_list = batch.x
        
        if self.gt_stack_proj is not None:
            gt_stack_input = self.gt_stack_proj(mp_x_list)

        # Pass through GPS stack and collect intermediate outputs
        if hasattr(self, 'gt_stack') and len(self.gt_stack) > 0:
            batch.x = gt_stack_input  # Set x for GPS processing
            gt_x_list = []
            for layer in self.gt_stack:
                batch = layer(batch)
                gt_x_list.append(batch.x)
            batch.gt_x_list = gt_x_list
        else:
            # Fallback if no gt_stack
            batch.gt_x_list = [gt_stack_input]
        
        # Concatenate all intermediate outputs: mp outputs + gt_stack outputs
        all_outputs = [mp_x_list] + batch.gt_x_list
        combined = torch.cat(all_outputs, dim=-1)
        
        batch.x = combined
        batch = self.post_mp(batch)
        return batch
