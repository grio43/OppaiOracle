#!/usr/bin/env python3
"""
Progressive Model Scaling for Anime Image Tagger
Utilities for scaling from 1B to 3B parameters progressively
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_, xavier_uniform_, kaiming_uniform_
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our modules
from model_architecture import (
    VisionTransformerConfig, 
    AnimeVisionTransformer,
    AnimeTransformerWithHead,
    PatchEmbed,
    Attention,
    MLP,
    TransformerBlock,
    create_model
)


logger = logging.getLogger(__name__)


def compute_model_stats(model: nn.Module) -> Dict[str, Any]:
    """
    Compute statistics about a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Layer-wise statistics
    layer_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
            layer_params = sum(p.numel() for p in module.parameters())
            layer_stats[name] = layer_params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'layer_stats': layer_stats,
        'dtype': next(model.parameters()).dtype,
        'device': next(model.parameters()).device
    }


@dataclass
class ScalingConfig:
    """Configuration for model scaling"""
    # Scaling strategy
    scaling_method: str = "depth_width"  # depth_only, width_only, depth_width, compound
    
    # Target sizes (1B -> 1.5B -> 2B -> 3B)
    scaling_stages: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "1B", "hidden_size": 1536, "num_layers": 28, "num_heads": 24},
        {"name": "1.5B", "hidden_size": 1792, "num_layers": 32, "num_heads": 28},
        {"name": "2B", "hidden_size": 2048, "num_layers": 36, "num_heads": 32},
        {"name": "3B", "hidden_size": 2304, "num_layers": 40, "num_heads": 36}
    ])
    
    # Initialization
    init_method: str = "progressive"  # progressive, interpolate, random
    copy_weights: bool = True
    interpolate_positions: bool = True
    
    # Layer mapping strategy
    layer_mapping: str = "interleave"  # sequential, interleave, optimal
    
    # Training after scaling
    initial_lr_multiplier: float = 0.1  # Lower LR for new parameters
    warmup_epochs: int = 5
    freeze_old_layers: bool = False
    freeze_epochs: int = 2
    
    # Performance tracking
    track_metrics: bool = True
    compare_models: bool = True
    
    # Output
    output_dir: str = "./scaled_models"
    save_intermediate: bool = True


class ParameterInitializer:
    """Smart initialization for new parameters during scaling"""
    
    @staticmethod
    def initialize_expanded_linear(
        old_weight: torch.Tensor,
        new_shape: Tuple[int, int],
        method: str = "progressive"
    ) -> torch.Tensor:
        """
        Initialize expanded linear layer weights
        
        Args:
            old_weight: Original weight tensor
            new_shape: Target shape (out_features, in_features)
            method: Initialization method
            
        Returns:
            Initialized weight tensor
        """
        old_out, old_in = old_weight.shape
        new_out, new_in = new_shape
        
        if method == "progressive":
            # Create new tensor
            new_weight = torch.zeros(new_shape, dtype=old_weight.dtype, device=old_weight.device)
            
            # Copy old weights
            new_weight[:min(old_out, new_out), :min(old_in, new_in)] = old_weight[:min(old_out, new_out), :min(old_in, new_in)]
            
            # Initialize new output features
            if new_out > old_out:
                # Use statistics from existing weights
                std = old_weight.std().item()
                new_weight[old_out:, :min(old_in, new_in)] = torch.randn(
                    new_out - old_out, min(old_in, new_in), 
                    dtype=old_weight.dtype, 
                    device=old_weight.device
                ) * std
            
            # Initialize new input features
            if new_in > old_in:
                # Use fan-in based initialization
                fan_in = new_in
                std = math.sqrt(2.0 / fan_in)
                new_weight[:, old_in:] = torch.randn(
                    new_out, new_in - old_in,
                    dtype=old_weight.dtype,
                    device=old_weight.device
                ) * std
                
        elif method == "interpolate":
            # Interpolate weights to new size
            old_weight_2d = old_weight.unsqueeze(0).unsqueeze(0)
            new_weight_2d = F.interpolate(
                old_weight_2d,
                size=new_shape,
                mode='bilinear',
                align_corners=False
            )
            new_weight = new_weight_2d.squeeze(0).squeeze(0)
            
        else:  # random
            new_weight = torch.empty(new_shape, dtype=old_weight.dtype, device=old_weight.device)
            xavier_uniform_(new_weight)
            
        return new_weight
    
    @staticmethod
    def initialize_expanded_embedding(
        old_embedding: torch.Tensor,
        new_shape: Tuple[int, int],
        method: str = "progressive"
    ) -> torch.Tensor:
        """Initialize expanded embedding matrix"""
        old_num, old_dim = old_embedding.shape
        new_num, new_dim = new_shape
        
        new_embedding = torch.zeros(new_shape, dtype=old_embedding.dtype, device=old_embedding.device)
        
        if method == "progressive":
            # Copy old embeddings
            new_embedding[:min(old_num, new_num), :min(old_dim, new_dim)] = old_embedding[:min(old_num, new_num), :min(old_dim, new_dim)]
            
            # Initialize new embeddings
            if new_num > old_num:
                # Use mean and std of existing embeddings
                mean = old_embedding.mean(dim=0)
                std = old_embedding.std(dim=0)
                noise = torch.randn(
                    new_num - old_num, old_dim,
                    dtype=old_embedding.dtype,
                    device=old_embedding.device
                )
                new_embedding[old_num:, :old_dim] = mean + noise * std
            
            # Expand embedding dimension
            if new_dim > old_dim:
                # Initialize new dimensions with small values
                new_embedding[:, old_dim:] = torch.randn(
                    new_num, new_dim - old_dim,
                    dtype=old_embedding.dtype,
                    device=old_embedding.device
                ) * 0.02
                
        elif method == "interpolate":
            # Interpolate to new size
            if new_num != old_num:
                # Interpolate along token dimension
                old_embedding = old_embedding.t().unsqueeze(0)
                new_embedding_partial = F.interpolate(
                    old_embedding,
                    size=(old_dim, new_num),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).t()
                new_embedding[:, :old_dim] = new_embedding_partial
            else:
                new_embedding[:, :old_dim] = old_embedding
                
            if new_dim > old_dim:
                # Small random init for new dims
                new_embedding[:, old_dim:] = torch.randn(
                    new_num, new_dim - old_dim,
                    dtype=old_embedding.dtype,
                    device=old_embedding.device
                ) * 0.02
                
        return new_embedding


class ModelScaler:
    """Main class for progressive model scaling"""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track scaling history
        self.scaling_history = []
        
    def scale_model(
        self,
        source_model: AnimeTransformerWithHead,
        source_config: VisionTransformerConfig,
        target_stage: Dict[str, Any]
    ) -> Tuple[AnimeTransformerWithHead, VisionTransformerConfig]:
        """
        Scale model to target configuration
        
        Args:
            source_model: Source model to scale from
            source_config: Source model configuration
            target_stage: Target configuration dict
            
        Returns:
            Scaled model and configuration
        """
        logger.info(f"Scaling model to {target_stage['name']} configuration")
        
        # Create target configuration
        target_config = self._create_target_config(source_config, target_stage)
        
        # Create new model with target architecture
        target_model = create_model(config=target_config)
        
        # Transfer weights based on scaling method
        if self.config.scaling_method == "depth_only":
            self._scale_depth_only(source_model, target_model, source_config, target_config)
        elif self.config.scaling_method == "width_only":
            self._scale_width_only(source_model, target_model, source_config, target_config)
        elif self.config.scaling_method == "depth_width":
            self._scale_depth_and_width(source_model, target_model, source_config, target_config)
        else:  # compound
            self._scale_compound(source_model, target_model, source_config, target_config)
        
        # Log scaling results
        self._log_scaling_results(source_model, target_model, target_stage)
        
        return target_model, target_config
    
    def _create_target_config(
        self,
        source_config: VisionTransformerConfig,
        target_stage: Dict[str, Any]
    ) -> VisionTransformerConfig:
        """Create target model configuration"""
        # Copy source config
        target_config = copy.deepcopy(source_config)
        
        # Update with target parameters
        target_config.hidden_size = target_stage.get('hidden_size', source_config.hidden_size)
        target_config.num_hidden_layers = target_stage.get('num_layers', source_config.num_hidden_layers)
        target_config.num_attention_heads = target_stage.get('num_heads', source_config.num_attention_heads)
        target_config.intermediate_size = target_stage.get(
            'intermediate_size', 
            target_config.hidden_size * 4
        )
        
        return target_config
    
    def _scale_depth_only(
        self,
        source_model: AnimeTransformerWithHead,
        target_model: AnimeTransformerWithHead,
        source_config: VisionTransformerConfig,
        target_config: VisionTransformerConfig
    ):
        """Scale model depth (add layers)"""
        logger.info("Scaling depth only...")
        
        # Copy non-layer weights directly
        self._copy_non_layer_weights(source_model, target_model, source_config, target_config)
        
        # Map layers based on strategy
        layer_mapping = self._create_layer_mapping(
            source_config.num_hidden_layers,
            target_config.num_hidden_layers
        )
        
        # Copy and initialize transformer layers
        for target_idx, source_idx in enumerate(layer_mapping):
            if source_idx >= 0:
                # Copy from source layer
                self._copy_layer_weights(
                    source_model.transformer.blocks[source_idx],
                    target_model.transformer.blocks[target_idx]
                )
            else:
                # Initialize new layer
                self._initialize_new_layer(target_model.transformer.blocks[target_idx])
    
    def _scale_width_only(
        self,
        source_model: AnimeTransformerWithHead,
        target_model: AnimeTransformerWithHead,
        source_config: VisionTransformerConfig,
        target_config: VisionTransformerConfig
    ):
        """Scale model width (hidden size)"""
        logger.info("Scaling width only...")
        
        # Scale embeddings
        self._scale_embeddings(source_model, target_model, source_config, target_config)
        
        # Scale each transformer layer
        for i in range(min(source_config.num_hidden_layers, target_config.num_hidden_layers)):
            self._scale_transformer_layer(
                source_model.transformer.blocks[i],
                target_model.transformer.blocks[i],
                source_config.hidden_size,
                target_config.hidden_size
            )
        
        # Scale output head
        self._scale_output_head(source_model, target_model, source_config, target_config)
    
    def _scale_depth_and_width(
        self,
        source_model: AnimeTransformerWithHead,
        target_model: AnimeTransformerWithHead,
        source_config: VisionTransformerConfig,
        target_config: VisionTransformerConfig
    ):
        """Scale both depth and width"""
        logger.info("Scaling both depth and width...")
        
        # First scale embeddings
        self._scale_embeddings(source_model, target_model, source_config, target_config)
        
        # Create layer mapping for depth scaling
        layer_mapping = self._create_layer_mapping(
            source_config.num_hidden_layers,
            target_config.num_hidden_layers
        )
        
        # Scale transformer layers
        for target_idx, source_idx in enumerate(layer_mapping):
            if source_idx >= 0:
                # Scale existing layer
                self._scale_transformer_layer(
                    source_model.transformer.blocks[source_idx],
                    target_model.transformer.blocks[target_idx],
                    source_config.hidden_size,
                    target_config.hidden_size
                )
            else:
                # Initialize new layer with target width
                self._initialize_new_layer(target_model.transformer.blocks[target_idx])
        
        # Scale output head
        self._scale_output_head(source_model, target_model, source_config, target_config)
    
    def _scale_compound(
        self,
        source_model: AnimeTransformerWithHead,
        target_model: AnimeTransformerWithHead,
        source_config: VisionTransformerConfig,
        target_config: VisionTransformerConfig
    ):
        """
        Scale using compound method (depth, width, and heads together)
        This is the most sophisticated scaling approach
        """
        logger.info("Scaling using compound method...")
        
        # Scale embeddings with special handling for compound scaling
        self._scale_embeddings(source_model, target_model, source_config, target_config)
        
        # Calculate scaling factors
        depth_factor = target_config.num_hidden_layers / source_config.num_hidden_layers
        width_factor = target_config.hidden_size / source_config.hidden_size
        head_factor = target_config.num_attention_heads / source_config.num_attention_heads
        
        logger.info(f"Compound scaling factors - Depth: {depth_factor:.2f}, Width: {width_factor:.2f}, Heads: {head_factor:.2f}")
        
        # Create sophisticated layer mapping for compound scaling
        layer_mapping = self._create_compound_layer_mapping(
            source_config.num_hidden_layers,
            target_config.num_hidden_layers,
            depth_factor
        )
        
        # Scale transformer layers with compound approach
        for target_idx, (source_idx, weight) in enumerate(layer_mapping):
            if source_idx >= 0:
                if weight == 1.0:
                    # Direct scaling of single source layer
                    self._scale_transformer_layer(
                        source_model.transformer.blocks[source_idx],
                        target_model.transformer.blocks[target_idx],
                        source_config.hidden_size,
                        target_config.hidden_size
                    )
                else:
                    # Weighted combination of layers (for fractional scaling)
                    self._scale_transformer_layer_weighted(
                        source_model.transformer.blocks,
                        target_model.transformer.blocks[target_idx],
                        source_idx,
                        weight,
                        source_config.hidden_size,
                        target_config.hidden_size
                    )
            else:
                # Initialize completely new layer
                self._initialize_new_layer(target_model.transformer.blocks[target_idx])
        
        # Scale output head with compound considerations
        self._scale_output_head(source_model, target_model, source_config, target_config)
    
    def _create_compound_layer_mapping(
        self, 
        source_layers: int, 
        target_layers: int, 
        depth_factor: float
    ) -> List[Tuple[int, float]]:
        """
        Create sophisticated layer mapping for compound scaling
        Returns list of (source_layer_idx, weight) tuples
        """
        mapping = []
        
        for target_idx in range(target_layers):
            # Calculate which source layer(s) this target layer should draw from
            source_position = target_idx / depth_factor
            
            if source_position < source_layers:
                source_idx = int(source_position)
                weight = 1.0  # For simplicity, use full weight
                mapping.append((source_idx, weight))
            else:
                # New layer beyond source depth
                mapping.append((-1, 0.0))
        
        return mapping
    
    def _scale_transformer_layer_weighted(
        self,
        source_blocks: nn.ModuleList,
        target_layer: TransformerBlock,
        source_idx: int,
        weight: float,
        source_hidden: int,
        target_hidden: int
    ):
        """Scale transformer layer with weighted combination"""
        # For simplicity, just scale from the primary source layer
        # In a more sophisticated implementation, could blend multiple layers
        self._scale_transformer_layer(
            source_blocks[source_idx],
            target_layer,
            source_hidden,
            target_hidden
        )
    
    def _copy_non_layer_weights(
        self,
        source_model: AnimeTransformerWithHead,
        target_model: AnimeTransformerWithHead,
        source_config: VisionTransformerConfig,
        target_config: VisionTransformerConfig
    ):
        """Copy weights that don't need scaling"""
        # Tag embeddings (vocabulary doesn't change)
        if hasattr(source_model, 'tag_embeddings') and hasattr(target_model, 'tag_embeddings'):
            target_model.tag_embeddings.weight.data.copy_(source_model.tag_embeddings.weight.data)
    
    def _scale_embeddings(
        self,
        source_model: AnimeTransformerWithHead,
        target_model: AnimeTransformerWithHead,
        source_config: VisionTransformerConfig,
        target_config: VisionTransformerConfig
    ):
        """Scale embedding layers"""
        source_transformer = source_model.transformer
        target_transformer = target_model.transformer
        
        # Scale patch embeddings
        if source_config.hidden_size != target_config.hidden_size:
            old_weight = source_transformer.patch_embed.projection.weight
            old_shape = old_weight.shape  # (out_channels, in_channels, H, W)
            
            # Create new weight tensor with correct shape
            new_weight = torch.zeros(
                target_config.hidden_size, old_shape[1], old_shape[2], old_shape[3],
                dtype=old_weight.dtype, device=old_weight.device
            )
            
            # Copy old weights
            min_channels = min(old_shape[0], target_config.hidden_size)
            new_weight[:min_channels] = old_weight[:min_channels]
            
            # Initialize new channels if needed
            if target_config.hidden_size > old_shape[0]:
                # Use Kaiming initialization for new channels
                nn.init.kaiming_uniform_(new_weight[old_shape[0]:], a=math.sqrt(5))
            
            target_transformer.patch_embed.projection.weight.data = new_weight
            
            # Handle bias if it exists
            if source_transformer.patch_embed.projection.bias is not None:
                old_bias = source_transformer.patch_embed.projection.bias
                new_bias = torch.zeros(target_config.hidden_size, dtype=old_bias.dtype, device=old_bias.device)
                new_bias[:min(old_bias.shape[0], target_config.hidden_size)] = old_bias[:min(old_bias.shape[0], target_config.hidden_size)]
                target_transformer.patch_embed.projection.bias.data = new_bias
        
        # Scale special tokens
        for token_name in ['cls_token', 'style_token', 'line_token', 'color_token']:
            if hasattr(source_transformer, token_name):
                old_token = getattr(source_transformer, token_name)
                new_token = ParameterInitializer.initialize_expanded_embedding(
                    old_token.squeeze(0).squeeze(0).unsqueeze(0),
                    (1, target_config.hidden_size),
                    method=self.config.init_method
                ).unsqueeze(0)
                getattr(target_transformer, token_name).data = new_token
        
        # Scale position embeddings
        if self.config.interpolate_positions and hasattr(source_transformer, 'pos_embed'):
            old_pos = source_transformer.pos_embed
            if old_pos.shape[-1] != target_config.hidden_size:
                new_pos = ParameterInitializer.initialize_expanded_embedding(
                    old_pos.squeeze(0),
                    (old_pos.shape[1], target_config.hidden_size),
                    method=self.config.init_method
                ).unsqueeze(0)
                target_transformer.pos_embed.data = new_pos
    
    def _scale_transformer_layer(
        self,
        source_layer: TransformerBlock,
        target_layer: TransformerBlock,
        source_hidden: int,
        target_hidden: int
    ):
        """Scale a single transformer layer"""
        # Scale attention
        self._scale_attention(source_layer.attn, target_layer.attn, source_hidden, target_hidden)
        
        # Scale MLP
        self._scale_mlp(source_layer.mlp, target_layer.mlp, source_hidden, target_hidden)
        
        # Scale layer norms
        self._scale_layer_norm(source_layer.norm1, target_layer.norm1, source_hidden, target_hidden)
        self._scale_layer_norm(source_layer.norm2, target_layer.norm2, source_hidden, target_hidden)
    
    def _scale_attention(
        self,
        source_attn: Attention,
        target_attn: Attention,
        source_hidden: int,
        target_hidden: int
    ):
        """Scale attention module"""
        # Scale QKV projection
        old_qkv = source_attn.qkv.weight
        new_qkv_shape = (target_attn.all_head_size * 3, target_hidden)
        new_qkv = ParameterInitializer.initialize_expanded_linear(
            old_qkv,
            new_qkv_shape,
            method=self.config.init_method
        )
        target_attn.qkv.weight.data = new_qkv
        
        # Handle QKV bias if exists
        if source_attn.qkv.bias is not None:
            old_bias = source_attn.qkv.bias
            new_bias = torch.zeros(new_qkv_shape[0], dtype=old_bias.dtype, device=old_bias.device)
            min_size = min(old_bias.shape[0], new_bias.shape[0])
            new_bias[:min_size] = old_bias[:min_size]
            target_attn.qkv.bias.data = new_bias
        
        # Scale output projection
        old_proj = source_attn.proj.weight
        new_proj_shape = (target_hidden, target_attn.all_head_size)
        new_proj = ParameterInitializer.initialize_expanded_linear(
            old_proj,
            new_proj_shape,
            method=self.config.init_method
        )
        target_attn.proj.weight.data = new_proj
        
        # Handle projection bias if exists
        if source_attn.proj.bias is not None:
            old_bias = source_attn.proj.bias
            new_bias = torch.zeros(target_hidden, dtype=old_bias.dtype, device=old_bias.device)
            min_size = min(old_bias.shape[0], new_bias.shape[0])
            new_bias[:min_size] = old_bias[:min_size]
            target_attn.proj.bias.data = new_bias
    
    def _scale_mlp(
        self,
        source_mlp: MLP,
        target_mlp: MLP,
        source_hidden: int,
        target_hidden: int
    ):
        """Scale MLP module"""
        # Scale first linear
        old_fc1 = source_mlp.fc1.weight
        new_fc1_shape = (target_mlp.fc1.out_features, target_hidden)
        new_fc1 = ParameterInitializer.initialize_expanded_linear(
            old_fc1,
            new_fc1_shape,
            method=self.config.init_method
        )
        target_mlp.fc1.weight.data = new_fc1
        
        # Handle fc1 bias
        if source_mlp.fc1.bias is not None:
            old_bias = source_mlp.fc1.bias
            new_bias = torch.zeros(target_mlp.fc1.out_features, dtype=old_bias.dtype, device=old_bias.device)
            min_size = min(old_bias.shape[0], new_bias.shape[0])
            new_bias[:min_size] = old_bias[:min_size]
            target_mlp.fc1.bias.data = new_bias
        
        # Scale second linear
        old_fc2 = source_mlp.fc2.weight
        new_fc2_shape = (target_hidden, target_mlp.fc2.in_features)
        new_fc2 = ParameterInitializer.initialize_expanded_linear(
            old_fc2,
            new_fc2_shape,
            method=self.config.init_method
        )
        target_mlp.fc2.weight.data = new_fc2
        
        # Handle fc2 bias
        if source_mlp.fc2.bias is not None:
            old_bias = source_mlp.fc2.bias
            new_bias = torch.zeros(target_hidden, dtype=old_bias.dtype, device=old_bias.device)
            min_size = min(old_bias.shape[0], new_bias.shape[0])
            new_bias[:min_size] = old_bias[:min_size]
            target_mlp.fc2.bias.data = new_bias
    
    def _scale_layer_norm(
        self,
        source_ln: nn.LayerNorm,
        target_ln: nn.LayerNorm,
        source_hidden: int,
        target_hidden: int
    ):
        """Scale layer normalization"""
        if source_hidden != target_hidden:
            min_size = min(source_hidden, target_hidden)
            # Initialize new dimensions
            target_ln.weight.data[:min_size] = source_ln.weight.data[:min_size]
            target_ln.weight.data[min_size:] = 1.0
            
            target_ln.bias.data[:min_size] = source_ln.bias.data[:min_size]
            target_ln.bias.data[min_size:] = 0.0
        else:
            target_ln.weight.data = source_ln.weight.data
            target_ln.bias.data = source_ln.bias.data
    
    def _scale_output_head(
        self,
        source_model: AnimeTransformerWithHead,
        target_model: AnimeTransformerWithHead,
        source_config: VisionTransformerConfig,
        target_config: VisionTransformerConfig
    ):
        """Scale the output prediction head"""
        # Scale shared MLP in tag head
        source_head = source_model.tag_head
        target_head = target_model.tag_head
        
        # This depends on the specific head architecture
        # For hierarchical head, we mainly need to scale the shared layers
        if hasattr(source_head, 'shared_mlp'):
            # Scale shared MLP layers
            for source_layer, target_layer in zip(
                source_head.shared_mlp.layers,
                target_head.shared_mlp.layers
            ):
                if isinstance(source_layer, nn.Linear) and isinstance(target_layer, nn.Linear):
                    if source_layer.in_features != target_layer.in_features:
                        # Scale linear layer
                        new_weight = ParameterInitializer.initialize_expanded_linear(
                            source_layer.weight,
                            (target_layer.out_features, target_layer.in_features),
                            method=self.config.init_method
                        )
                        target_layer.weight.data = new_weight
    
    def _create_layer_mapping(self, source_layers: int, target_layers: int) -> List[int]:
        """Create mapping from target layers to source layers"""
        if self.config.layer_mapping == "sequential":
            # Map first N layers directly, initialize rest
            return list(range(min(source_layers, target_layers))) + [-1] * max(0, target_layers - source_layers)
        
        elif self.config.layer_mapping == "interleave":
            # Interleave source layers across target
            mapping = []
            for i in range(target_layers):
                source_idx = int(i * source_layers / target_layers)
                if source_idx < source_layers:
                    mapping.append(source_idx)
                else:
                    mapping.append(-1)
            return mapping
        
        elif self.config.layer_mapping == "optimal":
            # Use optimal mapping strategy - distribute layers evenly
            mapping = []
            for i in range(target_layers):
                # Find the closest source layer
                position = i / (target_layers - 1) if target_layers > 1 else 0
                source_idx = min(int(position * (source_layers - 1)), source_layers - 1)
                
                # Check if this is a repeated layer or new layer
                if i < source_layers:
                    mapping.append(i)
                elif source_layers > 0:
                    # Repeat layers in a pattern
                    mapping.append(i % source_layers)
                else:
                    mapping.append(-1)
            
            return mapping
        
        else:
            # Default to interleave
            return self._create_layer_mapping_interleave(source_layers, target_layers)
    
    def _create_layer_mapping_interleave(self, source_layers: int, target_layers: int) -> List[int]:
        """Helper method for interleave mapping"""
        mapping = []
        for i in range(target_layers):
            source_idx = int(i * source_layers / target_layers)
            if source_idx < source_layers:
                mapping.append(source_idx)
            else:
                mapping.append(-1)
        return mapping
    
    def _copy_layer_weights(self, source_layer: nn.Module, target_layer: nn.Module):
        """Copy weights from source to target layer"""
        source_state = source_layer.state_dict()
        target_state = target_layer.state_dict()
        
        for key in source_state:
            if key in target_state and source_state[key].shape == target_state[key].shape:
                target_state[key].copy_(source_state[key])
        
        target_layer.load_state_dict(target_state)
    
    def _initialize_new_layer(self, layer: nn.Module):
        """Initialize a new layer"""
        # Layer is already initialized by model creation
        # Apply custom initialization if needed
        for module in layer.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _log_scaling_results(
        self,
        source_model: AnimeTransformerWithHead,
        target_model: AnimeTransformerWithHead,
        target_stage: Dict[str, Any]
    ):
        """Log scaling results"""
        source_stats = compute_model_stats(source_model)
        target_stats = compute_model_stats(target_model)
        
        scaling_info = {
            'timestamp': datetime.now().isoformat(),
            'target_stage': target_stage['name'],
            'source_params': source_stats['total_params'],
            'target_params': target_stats['total_params'],
            'param_increase': target_stats['total_params'] / source_stats['total_params'],
            'method': self.config.scaling_method
        }
        
        self.scaling_history.append(scaling_info)
        
        logger.info(f"Scaling complete:")
        logger.info(f"  Source params: {source_stats['total_params']:,}")
        logger.info(f"  Target params: {target_stats['total_params']:,}")
        logger.info(f"  Increase: {scaling_info['param_increase']:.2f}x")
    
    def progressive_scaling_pipeline(
        self,
        initial_checkpoint: str,
        stages_to_run: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run complete progressive scaling pipeline
        
        Args:
            initial_checkpoint: Path to initial model checkpoint
            stages_to_run: Which stages to run (default: all)
            
        Returns:
            Results dictionary
        """
        results = {
            'stages': {},
            'history': []
        }
        
        # Load initial model
        checkpoint = torch.load(initial_checkpoint, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'config' in checkpoint and 'model_config' in checkpoint['config']:
            config_dict = checkpoint['config']['model_config']
        elif 'model_config' in checkpoint:
            config_dict = checkpoint['model_config']
        else:
            # Try to infer config from model state dict
            raise ValueError("Cannot find model configuration in checkpoint")
        
        current_config = VisionTransformerConfig(**config_dict)
        current_model = create_model(config=current_config)
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            current_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume checkpoint is the state dict itself
            current_model.load_state_dict(checkpoint)
        
        # Determine stages to run
        if stages_to_run is None:
            stages_to_run = [stage['name'] for stage in self.config.scaling_stages[1:]]  # Skip first (1B)
        
        # Run scaling for each stage
        for stage in self.config.scaling_stages:
            if stage['name'] not in stages_to_run:
                continue
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Scaling to {stage['name']}")
            logger.info(f"{'='*60}")
            
            # Scale model
            scaled_model, scaled_config = self.scale_model(
                current_model,
                current_config,
                stage
            )
            
            # Save scaled model
            if self.config.save_intermediate:
                save_path = self.output_dir / f"model_{stage['name']}.pt"
                self._save_scaled_model(scaled_model, scaled_config, save_path)
            
            # Compare performance if requested
            if self.config.compare_models:
                comparison = self._compare_models(current_model, scaled_model)
                results['stages'][stage['name']] = comparison
            
            # Update current model for next iteration
            current_model = scaled_model
            current_config = scaled_config
        
        # Save final results
        results['history'] = self.scaling_history
        results_path = self.output_dir / 'scaling_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nScaling pipeline complete. Results saved to {results_path}")
        
        return results
    
    def _save_scaled_model(
        self,
        model: AnimeTransformerWithHead,
        config: VisionTransformerConfig,
        save_path: Path
    ):
        """Save scaled model checkpoint"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': {
                'model_config': asdict(config)
            },
            'scaling_info': {
                'method': self.config.scaling_method,
                'init_method': self.config.init_method,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved scaled model to {save_path}")
    
    def _compare_models(
        self,
        source_model: AnimeTransformerWithHead,
        target_model: AnimeTransformerWithHead
    ) -> Dict[str, Any]:
        """Compare source and target models"""
        comparison = {}
        
        # Parameter count comparison
        source_stats = compute_model_stats(source_model)
        target_stats = compute_model_stats(target_model)
        
        comparison['parameter_comparison'] = {
            'source_total': source_stats['total_params'],
            'target_total': target_stats['total_params'],
            'increase_factor': target_stats['total_params'] / source_stats['total_params'] if source_stats['total_params'] > 0 else 0
        }
        
        # Layer-wise comparison
        comparison['layer_comparison'] = {
            'source_layers': len(source_model.transformer.blocks) if hasattr(source_model.transformer, 'blocks') else 0,
            'target_layers': len(target_model.transformer.blocks) if hasattr(target_model.transformer, 'blocks') else 0,
            'source_hidden': source_model.transformer.config.hidden_size if hasattr(source_model.transformer, 'config') else 0,
            'target_hidden': target_model.transformer.config.hidden_size if hasattr(target_model.transformer, 'config') else 0
        }
        
        # Memory footprint (approximate)
        source_memory = sum(p.numel() * p.element_size() for p in source_model.parameters()) / (1024**3)
        target_memory = sum(p.numel() * p.element_size() for p in target_model.parameters()) / (1024**3)
        
        comparison['memory_comparison'] = {
            'source_gb': source_memory,
            'target_gb': target_memory,
            'increase_gb': target_memory - source_memory
        }
        
        return comparison


class ScalingOptimizer:
    """Optimizes training after model scaling"""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.scaled_param_names = set()  # Track which parameters were scaled
    
    def mark_scaled_parameters(self, model: AnimeTransformerWithHead, original_param_count: int):
        """Mark which parameters are new/scaled"""
        current_param_count = sum(p.numel() for p in model.parameters())
        
        # Simple heuristic: mark parameters based on size changes
        # In practice, you'd track this during scaling
        for name, param in model.named_parameters():
            param_size = param.numel()
            # This is a simplified check - in reality, track during scaling
            if 'new' in name or param_size > 1000000:  # Large parameters likely scaled
                self.scaled_param_names.add(name)
    
    def create_scaled_optimizer(
        self,
        model: AnimeTransformerWithHead,
        base_lr: float,
        scaled_from_checkpoint: bool = True
    ) -> torch.optim.Optimizer:
        """
        Create optimizer with different learning rates for scaled components
        
        Args:
            model: Scaled model
            base_lr: Base learning rate
            scaled_from_checkpoint: Whether model was scaled from checkpoint
            
        Returns:
            Configured optimizer
        """
        param_groups = []
        
        if scaled_from_checkpoint and self.config.initial_lr_multiplier != 1.0:
            # Separate parameters into old and new
            old_params = []
            new_params = []
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                    
                # Check if parameter is marked as scaled/new
                if name in self.scaled_param_names:
                    new_params.append(param)
                else:
                    old_params.append(param)
            
            # Create parameter groups
            if old_params:
                param_groups.append({
                    'params': old_params,
                    'lr': base_lr,
                    'name': 'old_params'
                })
            
            if new_params:
                param_groups.append({
                    'params': new_params,
                    'lr': base_lr * self.config.initial_lr_multiplier,
                    'name': 'new_params'
                })
                
            logger.info(f"Created optimizer with {len(old_params)} old params and {len(new_params)} new params")
        else:
            # Standard optimizer
            param_groups = [{'params': model.parameters(), 'lr': base_lr}]
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        return optimizer
    
    def get_scaling_schedule(self, num_epochs: int) -> Dict[str, Any]:
        """Get training schedule for scaled model"""
        schedule = {
            'warmup_epochs': self.config.warmup_epochs,
            'freeze_epochs': self.config.freeze_epochs if self.config.freeze_old_layers else 0,
            'lr_schedule': 'cosine_with_warmup',
            'min_lr_ratio': 0.1,
            'total_epochs': num_epochs
        }
        
        return schedule


def create_scaling_config(
    method: str = "depth_width",
    target_size: str = "3B",
    **kwargs
) -> ScalingConfig:
    """Create scaling configuration"""
    # Default stage configurations
    default_stages = {
        "1B": {"name": "1B", "hidden_size": 1536, "num_layers": 28, "num_heads": 24},
        "1.5B": {"name": "1.5B", "hidden_size": 1792, "num_layers": 32, "num_heads": 28},
        "2B": {"name": "2B", "hidden_size": 2048, "num_layers": 36, "num_heads": 32},
        "3B": {"name": "3B", "hidden_size": 2304, "num_layers": 40, "num_heads": 36}
    }
    
    # Get stages up to target
    stage_names = ["1B", "1.5B", "2B", "3B"]
    if target_size not in stage_names:
        raise ValueError(f"Invalid target size: {target_size}. Must be one of {stage_names}")
    
    target_idx = stage_names.index(target_size)
    scaling_stages = [default_stages[name] for name in stage_names[:target_idx + 1]]
    
    config = ScalingConfig(
        scaling_method=method,
        scaling_stages=scaling_stages,
        **kwargs
    )
    
    return config


def scale_model_checkpoint(
    checkpoint_path: str,
    target_size: str = "1.5B",
    output_path: Optional[str] = None,
    method: str = "depth_width"
) -> str:
    """
    Convenience function to scale a model checkpoint
    
    Args:
        checkpoint_path: Path to source checkpoint
        target_size: Target model size
        output_path: Where to save scaled model
        method: Scaling method
        
    Returns:
        Path to scaled model
    """
    config = create_scaling_config(method=method, target_size=target_size)
    scaler = ModelScaler(config)
    
    # Load source model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config
    if 'config' in checkpoint and 'model_config' in checkpoint['config']:
        config_dict = checkpoint['config']['model_config']
    elif 'model_config' in checkpoint:
        config_dict = checkpoint['model_config']
    else:
        raise ValueError("Cannot find model configuration in checkpoint")
    
    source_config = VisionTransformerConfig(**config_dict)
    source_model = create_model(config=source_config)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        source_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        source_model.load_state_dict(checkpoint)
    
    # Get target stage
    target_stage = next(s for s in config.scaling_stages if s['name'] == target_size)
    
    # Scale model
    scaled_model, scaled_config = scaler.scale_model(source_model, source_config, target_stage)
    
    # Save scaled model
    if output_path is None:
        output_path = f"{checkpoint_path.replace('.pt', '')}_{target_size}.pt"
    
    scaler._save_scaled_model(scaled_model, scaled_config, Path(output_path))
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Progressive model scaling")
    parser.add_argument('checkpoint', type=str, help='Path to initial model checkpoint')
    parser.add_argument('--target', type=str, default='3B', choices=['1.5B', '2B', '3B'],
                       help='Target model size')
    parser.add_argument('--method', type=str, default='depth_width',
                       choices=['depth_only', 'width_only', 'depth_width', 'compound'],
                       help='Scaling method')
    parser.add_argument('--output-dir', type=str, default='./scaled_models',
                       help='Output directory')
    parser.add_argument('--all-stages', action='store_true',
                       help='Run all scaling stages progressively')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.all_stages:
        # Run complete pipeline
        config = create_scaling_config(
            method=args.method,
            target_size='3B',
            output_dir=args.output_dir
        )
        scaler = ModelScaler(config)
        results = scaler.progressive_scaling_pipeline(args.checkpoint)
        
        print("\nScaling Pipeline Results:")
        for stage, comparison in results['stages'].items():
            print(f"\n{stage}:")
            print(f"  Parameters: {comparison['parameter_comparison']['target_total']:,}")
            print(f"  Memory: {comparison['memory_comparison']['target_gb']:.2f} GB")
    else:
        # Scale to specific target
        output_path = scale_model_checkpoint(
            args.checkpoint,
            target_size=args.target,
            method=args.method
        )
        print(f"\nScaled model saved to: {output_path}")