# src/utils.py

import random
import numpy as np
import torch
import json

from torch import nn
import torch.nn.functional as F


def set_seed(seed):
    """Устанавливает зерно для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_user_profile_embeddings(file_path, user_id_mapping):
    """
    Старый метод: загружает эмбеддинги профилей пользователей из ОДНОГО JSON-файла,
    результат: [num_users, emb_dim], [num_users]
    """
    with open(file_path, 'r') as f:
        user_profiles_data = json.load(f)

    embedding_dim = len(next(iter(user_profiles_data.values())))
    # num_users = len(user_id_mapping)
    max_idx = max(user_id_mapping.values()) + 1  # Ensure list can accommodate the highest index
    print('Seq Stats:', max_idx, len(user_id_mapping))
    user_profiles_list = [[0.0] * embedding_dim for _ in range(max_idx)]
    null_profile_binary_mask = [False for _ in range(max_idx)]

    not_found_profiles_cnt = 0
    for original_id, idx in user_id_mapping.items():
        embedding = user_profiles_data.get(str(original_id))
        if embedding is not None:
            user_profiles_list[idx] = embedding
        else:
            # Если эмбеддинг не найден, инициализируем нулями
            # user_profiles_list[idx] = [0.0] * embedding_dim
            null_profile_binary_mask[idx] = True
    print(f"Number of users without profiles: {not_found_profiles_cnt}")

    user_profiles_tensor = torch.tensor(user_profiles_list, dtype=torch.float)
    null_profile_binary_mask_tensor = torch.BoolTensor(null_profile_binary_mask)
    return user_profiles_tensor, null_profile_binary_mask_tensor


def init_criterion_reconstruct(criterion_name):
    if criterion_name == 'MSE':
        return lambda x,y: nn.MSELoss()(x,y)
    if criterion_name == 'RMSE':
        return lambda x,y: torch.sqrt(nn.MSELoss()(x,y))
    if criterion_name == 'CosSim':
        return lambda x,y: 1 - torch.mean(nn.CosineSimilarity(dim=1, eps=1e-6)(x,y))
    raise Exception('Not existing reconstruction loss')


def calculate_recsys_loss(target_seq, outputs, criterion):
    # Проверяем, если outputs является кортежем (на всякий случай)
    if isinstance(outputs, tuple):
        outputs = outputs[0]

    logits = outputs.view(-1, outputs.size(-1))
    targets = target_seq.view(-1)

    loss = criterion(logits, targets)
    return loss


def calculate_guide_loss(model,
                         user_profile_emb,
                         hidden_for_reconstruction,
                         null_profile_binary_mask_batch,
                         criterion_reconstruct_fn):
    # if model.use_down_scale:
    #     user_profile_emb_transformed = model.profile_transform(user_profile_emb)
    # else:
    #     user_profile_emb_transformed = user_profile_emb.detach().clone().to(device)
    # if model.use_upscale:
    #     hidden_for_reconstruction = model.hidden_layer_transform(hidden_for_reconstruction)
    # user_profile_emb_transformed[null_profile_binary_mask_batch] = hidden_for_reconstruction[
    #     null_profile_binary_mask_batch]
    #
    # loss_guide = criterion_reconstruct_fn(hidden_for_reconstruction, user_profile_emb_transformed)
    # return loss_guide

    # pass
    user_profile_emb_transformed = model.aggregate_profile(user_profile_emb)
    if model.use_upscale:
        hidden_for_reconstruction = model.hidden_layer_transform(hidden_for_reconstruction)

    user_profile_emb_transformed[null_profile_binary_mask_batch] = \
        hidden_for_reconstruction[null_profile_binary_mask_batch]

    loss_guide = criterion_reconstruct_fn(hidden_for_reconstruction, user_profile_emb_transformed)
    return loss_guide

def mean_weightening(hidden_states):
    """hidden_states: [batch_size, seq_len, hidden_size]"""
    return hidden_states.mean(dim=1)


def exponential_weightening(hidden_states, weight_scale):
    """hidden_states: [batch_size, seq_len, hidden_size]"""
    device = hidden_states.device

    indices = torch.arange(hidden_states.shape[1]).float().to(device)  # [0, 1, 2, ..., seq_len-1]
    weights = torch.exp(weight_scale * indices)  # Shape: [seq_len]

    # Normalize weights (optional, for scale invariance)
    weights = weights / weights.sum()

    # Reshape weights to [1, n_items, 1] for broadcasting
    weights = weights.view(1, hidden_states.shape[1], 1)

    # Apply weights and aggregate
    weighted_tensor = hidden_states * weights
    result = weighted_tensor.sum(dim=1)  # Aggregated tensor, shape: [batch_size, hidden_units]
    return result


class SimpleAttentionAggregator(nn.Module):
    def __init__(self, hidden_units):
        super(SimpleAttentionAggregator, self).__init__()
        self.attention = nn.Linear(hidden_units, 1)  # Learnable attention weights

    def forward(self, x):
        """
        x: Input tensor of shape [batch_size, n_items, hidden_units]
        Returns:
        Aggregated tensor of shape [batch_size, hidden_units]
        """
        # Compute attention scores (shape: [batch_size, n_items, 1])
        scores = self.attention(x)

        # Normalize scores with softmax over the 2nd dimension
        weights = F.softmax(scores, dim=1)  # Shape: [batch_size, n_items, 1]

        # Weighted sum of the input tensor
        weighted_sum = (x * weights).sum(dim=1)  # Shape: [batch_size, hidden_units]
        return weighted_sum