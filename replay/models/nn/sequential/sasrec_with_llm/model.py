
import abc
import contextlib
from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
from torch import nn

from replay.data.nn import TensorMap, TensorSchema

from replay.models.nn.sequential.sasrec import SasRecModel
from replay.models.nn.sequential.sasrec.model import SasRecPositionalEmbedding
from replay.models.nn.sequential.sasrec_with_llm.utils import mean_weightening, exponential_weightening, \
    SimpleAttentionAggregator


class SasRecLLMModel(SasRecModel):
    """
    SasRec model
    """

    def __init__(
        self,
        schema: TensorSchema,
        profile_emb_dim: int,
        num_blocks: int = 2,
        num_heads: int = 1,
        hidden_size: int = 50,
        max_len: int = 200,
        dropout: float = 0.2,
        ti_modification: bool = False,
        time_span: int = 256,
        reconstruction_layer: int = -1,
        weighting_scheme='mean',
        use_down_scale=True,
        use_upscale=False,
        weight_scale=None,
        multi_profile=False,
        multi_profile_aggr_scheme='mean'
    ) -> None:
        """
        :param schema: Tensor schema of features.
        :param num_blocks: Number of Transformer blocks.
            Default: ``2``.
        :param num_heads: Number of Attention heads.
            Default: ``1``.
        :param hidden_size: Hidden size of transformer.
            Default: ``50``.
        :param max_len: Max length of sequence.
            Default: ``200``.
        :param dropout: Dropout rate.
            Default: ``0.2``.
        :param ti_modification: Enable time relation.
            Default: ``False``.
        :param time_span: Time span if `ti_modification` is `True`.
            Default: ``256``.
        """
        super().__init__(schema=schema,
                         num_blocks=num_blocks,
                         num_heads=num_heads,
                         hidden_size=hidden_size,
                         max_len=max_len,
                         dropout=dropout,
                         ti_modification=ti_modification,
                         time_span=time_span)
        if weighting_scheme == 'mean':
            self.weighting_fn = mean_weightening
            self.weighting_kwargs = {}
        elif weighting_scheme == 'exponential':
            self.weighting_fn = exponential_weightening
            self.weighting_kwargs = {'weight_scale': weight_scale}
        elif weighting_scheme == 'attention':
            self.weighting_fn = SimpleAttentionAggregator(self.hidden_size)
            self.weighting_kwargs = {}
        else:
            raise NotImplementedError(f'No such weighting_scheme {weighting_scheme} exists')

        # Агрегация нескольких профилей
        if multi_profile_aggr_scheme == 'mean':
            self.profile_aggregator = mean_weightening
            self.multi_profile_weighting_kwargs = {}
        elif multi_profile_aggr_scheme == 'attention':
            self.profile_aggregator = SimpleAttentionAggregator(profile_emb_dim if not use_down_scale
                                                                else self.hidden_size)
            self.multi_profile_weighting_kwargs = {}
        else:
            raise NotImplementedError(f'No such multi_profile_aggr_scheme {multi_profile_aggr_scheme} exists')

        self.use_down_scale = use_down_scale
        self.use_upscale = use_upscale
        self.multi_profile = multi_profile
        self.reconstruction_layer = reconstruction_layer

        if use_down_scale:
            self.profile_transform = nn.Linear(profile_emb_dim, self.hidden_size)
        if use_upscale:
            self.hidden_layer_transform = nn.Linear(self.hidden_size, profile_emb_dim)

    def forward(
        self,
        feature_tensor: TensorMap,
        padding_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param feature_tensor: Batch of features.
        :param padding_mask: Padding mask where 0 - <PAD>, 1 otherwise.

        :returns: Calculated scores.
        """
        output_embeddings, hidden_states = self.forward_step(feature_tensor, padding_mask)
        all_scores = self.get_logits(output_embeddings)
        return all_scores, hidden_states

    def get_query_embeddings(
        self,
        feature_tensor: TensorMap,
        padding_mask: torch.BoolTensor,
    ):
        """
        :param feature_tensor: Batch of features.
        :param padding_mask: Padding mask where 0 - <PAD>, 1 otherwise.

        :returns: Query embeddings.
        """
        output, hidden_state = self.forward_step(feature_tensor, padding_mask)
        return output[:, -1, :]

    def forward_step(
        self,
        feature_tensor: TensorMap,
        padding_mask: torch.BoolTensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """
        :param feature_tensor: Batch of features.
        :param padding_mask: Padding mask where 0 - <PAD>, 1 otherwise.

        :returns: Output embeddings.
        """
        device = feature_tensor[self.item_feature_name].device
        attention_mask, padding_mask, feature_tensor = self.masking(feature_tensor, padding_mask)
        if self.ti_modification:
            seqs, ti_embeddings = self.item_embedder(feature_tensor, padding_mask)
            seqs, hidden_states__list = self.sasrec_layers(seqs, attention_mask, padding_mask, ti_embeddings, device, return_hidden_states=True)
        else:
            seqs = self.item_embedder(feature_tensor, padding_mask)
            seqs, hidden_states__list = self.sasrec_layers(seqs, attention_mask, padding_mask, return_hidden_states=True)
        output_embeddings = self.output_normalization(seqs)
        if self.reconstruction_layer == -1:
            hidden_states = output_embeddings
        else:
            hidden_states = hidden_states__list[self.reconstruction_layer]
        hidden_states_agg = self.weighting_fn(hidden_states, **self.weighting_kwargs)
        return output_embeddings, hidden_states_agg

    def aggregate_profile(self, user_profile_emb):
        """
        user_profile_emb: [batch_size, emb_dim]  или  [batch_size, K, emb_dim]
        Возвращает: [batch_size, hidden_units] (если use_down_scale=True) либо [batch_size, emb_dim].
        """
        if user_profile_emb is None:
            return None

        if user_profile_emb.dim() == 2:
            # Случай single-profile (batch_size, emb_dim)
            if self.use_down_scale:
                return self.profile_transform(user_profile_emb)  # => [batch_size, hidden_units]
            else:
                return user_profile_emb.detach().clone()

        # Иначе multi-profile => [batch_size, K, emb_dim]
        bsz, K, edim = user_profile_emb.shape

        # Сначала down_scale (если нужно)
        if self.use_down_scale:
            # Применим линейно к каждому из K профилей
            user_profile_emb = user_profile_emb.view(bsz * K, edim)
            user_profile_emb = self.profile_transform(user_profile_emb)  # => [bsz*K, hidden_units]
            user_profile_emb = user_profile_emb.view(bsz, K, self.hidden_size)

        # Теперь агрегируем
        aggregated = self.profile_aggregator(user_profile_emb,
                                             *self.multi_profile_weighting_kwargs)  # => [bsz, emb_dim_now])
        return aggregated
