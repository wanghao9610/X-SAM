import math

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ...utils import farthest_point_sample, index_points, knn_point, point_sample, rand_sample_repeat
from .configuration_sampler import SamplerConfig


class ConvReLULN1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(ConvReLULN1D, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias), self.act
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # (B, C, N) -> (B, C_1, N)
        x = self.net(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)

        return x


class GeoSamplerModel(nn.Module):
    def __init__(self, config: SamplerConfig):
        super(GeoSamplerModel, self).__init__()
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.num_init_point = config.num_init_point
        self.num_sub_point = config.num_sub_point
        self.num_neighbor = config.num_neighbor
        self.pooler_mode = config.pooler_mode

        self.diff_projectors = nn.ModuleList()
        self.agg_projectors = nn.ModuleList()
        self.poolers = nn.ModuleList()

        for i in range(len(self.num_sub_point)):
            self.diff_projectors.append(nn.Linear(self.input_dim + 2, self.input_dim + 2))
            self.agg_projectors.append(
                ConvReLULN1D(
                    in_channels=2 * (self.input_dim + 2),
                    out_channels=self.input_dim,
                )
            )
            if self.pooler_mode == "mean":
                self.poolers.append(nn.AvgPool1d(kernel_size=self.num_neighbor[i]))
            elif self.pooler_mode == "max":
                self.poolers.append(nn.AdaptiveMaxPool1d(output_size=1))
            else:
                raise NotImplementedError(f"{self.pooler_mode} is not supported.")

        self.flatten_projector = nn.Linear(self.input_dim * self.num_sub_point[-1], self.input_dim)
        self.dim_projector = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, input_feats, vprompt_masks, original_dtype=None, return_dtype=None):
        assert len(input_feats) == len(vprompt_masks)
        if original_dtype is None:
            original_dtype = input_feats[0].dtype
        if return_dtype is None:
            return_dtype = input_feats[0].dtype

        all_points = []
        all_feats = []
        all_img_ids = []

        # Sample points and their features
        for img_idx, (feat, mask) in enumerate(zip(input_feats, vprompt_masks)):
            if len(mask) == 0:
                continue

            # Get image dimensions (w, h)
            img_wh = torch.tensor(
                [mask[0].shape[-2], mask[0].shape[-1]],
                device=mask[0].device,
            )[
                None,
            ]

            # Sample points from each mask
            points_pos = [rand_sample_repeat((m.nonzero() / img_wh), self.num_init_point) for m in mask]
            points_pos = torch.stack(points_pos)  # [num_mask, num_sample_point, 2]

            # Reshape feature map
            h = w = int(math.sqrt(feat.shape[0]))
            c = feat.shape[-1]
            feat_map = feat.reshape(h, w, c).permute(2, 0, 1)
            feat_map = feat_map.unsqueeze(0).repeat(points_pos.shape[0], 1, 1, 1)

            # Sample features at points
            feat_map_orig = feat_map.to(original_dtype)
            sampled_feats = point_sample(
                feat_map_orig,
                points_pos.flip(dims=(2,)).type(original_dtype),
                return_dtype,
                align_corners=True,
            )
            sampled_feats = sampled_feats.transpose(-2, -1)

            # Track image indices
            cur_img_ids = [img_idx] * len(points_pos)

            # Save to global lists
            all_points.append(points_pos)
            all_feats.append(sampled_feats)
            all_img_ids.extend(cur_img_ids)

        # No cond found, return list of None
        if len(all_points) == 0:
            return [None] * len(vprompt_masks)

        # Concatenate all points and features
        all_points = torch.cat(all_points, dim=0).to(return_dtype)  # [B*num_mask, num_sample_point, 2]
        all_feats = torch.cat(all_feats, dim=0)  # [B*num_mask, num_sample_point, C]
        all_img_ids = torch.tensor(all_img_ids, device=all_feats.device)

        assert all_points.shape[:-1] == all_feats.shape[:-1]

        # Process points through multiple stages
        for stage_idx in range(len(self.num_sub_point)):
            cur_sub_points = self.num_sub_point[stage_idx]
            cur_neighbors = self.num_neighbor[stage_idx]

            # Sample subset of points using farthest point sampling
            all_points = all_points.contiguous()
            fps_idx = farthest_point_sample(all_points, cur_sub_points).long()
            new_points = index_points(all_points, fps_idx)  # [B, npoint, 2]
            new_feats = index_points(all_feats, fps_idx)  # [B, npoint, d]

            # Find k nearest neighbors for each sampled point
            nn_idx = knn_point(cur_neighbors, all_points, new_points)
            group_points = index_points(all_points, nn_idx)  # [B, npoint, k, 2]
            group_feats = index_points(all_feats, nn_idx)  # [B, npoint, k, d]

            # Combine point features with coordinates
            local_feats = torch.cat([group_feats, group_points], dim=-1)  # [B, npoint, k, d+2]
            anchor_feats = torch.cat([new_feats, new_points], dim=-1).unsqueeze(-2)
            diff_feats = local_feats - anchor_feats

            # Project and aggregate features
            diff_feats = self.diff_projectors[stage_idx](diff_feats)
            gather_feats = torch.cat(
                [diff_feats, anchor_feats.repeat(1, 1, cur_neighbors, 1)], dim=-1
            )  # [B, npoint, k, 2(d+2)]

            # Reshape and project features
            b, n, s, d = gather_feats.size()
            gather_feats = gather_feats.permute(0, 1, 3, 2)  # [B, npoint, 2(d+2), k]
            gather_feats = gather_feats.reshape(-1, d, s)  # [B*npoint, 2(d+2), k]
            gather_feats = self.agg_projectors[stage_idx](gather_feats)  # [B*npoint, d, k]

            # Pool features
            batch_size, feat_dim, _ = gather_feats.size()
            gather_feats = self.poolers[stage_idx](gather_feats).view(batch_size, feat_dim)  # [B*npoint, d]
            gather_feats = gather_feats.reshape(b, n, -1)  # [B, npoint, d]

            # Update points and features for next stage
            all_points = new_points
            all_feats = gather_feats

        # Final projection
        flat_feats = all_feats.flatten(1, -1)  # [B, npoint x d]
        flat_feats = self.flatten_projector(flat_feats)
        vprompt_feats = self.dim_projector(flat_feats)  # [B, d]

        # Group features by image
        output_feats = []
        for img_idx in range(len(vprompt_masks)):
            mask = all_img_ids == img_idx
            if not mask.any():
                output_feats.append(None)
            else:
                output_feats.append(vprompt_feats[mask])

        return output_feats


class NaiveSamplerModel(nn.Module):
    def __init__(self, config: SamplerConfig):
        super(NaiveSamplerModel, self).__init__()
        self.num_sample_point = config.num_sample_point
        self.pooler = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, input_feats, vprompt_masks, original_dtype=None, return_dtype=None):
        assert len(input_feats) == len(vprompt_masks)
        if original_dtype is None:
            original_dtype = input_feats[0].dtype
        if return_dtype is None:
            return_dtype = input_feats[0].dtype

        all_points = []
        all_feats = []
        all_img_ids = []
        # input_feats: [B, H*W, C]
        # vprompt_masks: ([N, H, W], [N, H, W], ...), len(vprompt_masks) = B, N = num_masks
        for img_idx, (feat, mask) in enumerate(zip(input_feats, vprompt_masks)):
            # [H*W, C]
            if len(mask) != 0:
                img_wh = torch.tensor(
                    [mask[0].shape[-2], mask[0].shape[-1]],
                    device=mask[0].device,
                )[
                    None,
                ]
                # [num_sample_point, 2]
                for m in mask:
                    if m.nonzero().shape[0] <= 0:
                        print("error")
                points_pos = [rand_sample_repeat((m.nonzero() / img_wh), self.num_sample_point) for m in mask]
                # [num_mask, num_sample_point, 2]
                points_pos = torch.stack(points_pos)

                h = w = int(math.sqrt(feat.shape[0]))
                c = feat.shape[-1]

                feat_map = feat.reshape(h, w, c).permute(2, 0, 1)
                feat_map = feat_map.unsqueeze(0).repeat(points_pos.shape[0], 1, 1, 1)
                feat_map_orig = feat_map.to(original_dtype)
                sampled_feats = point_sample(
                    feat_map_orig,
                    points_pos.flip(dims=(2,)).type(original_dtype),
                    return_dtype,
                    align_corners=True,
                )
                # [num_mask, num_sample_point, C]
                sampled_feats = sampled_feats.transpose(-2, -1)

                cur_img_ids = [img_idx] * len(points_pos)

                all_points.append(points_pos)
                all_feats.append(sampled_feats)
                all_img_ids.extend(cur_img_ids)

        if len(all_points) == 0:
            return [None] * len(vprompt_masks)

        all_points = torch.cat(all_points, dim=0).to(return_dtype)
        all_feats = torch.cat(all_feats, dim=0).to(return_dtype)
        all_img_ids = torch.tensor(all_img_ids, device=all_feats.device)
        vprompt_feats = self.pooler(all_feats.transpose(-2, -1)).transpose(-2, -1).squeeze(1)

        output_feats = []
        for img_idx in range(len(vprompt_masks)):
            mask = all_img_ids == img_idx
            if not mask.any():
                output_feats.append(None)
            else:
                output_feats.append(vprompt_feats[mask])

        return output_feats


# Reference: https://github.com/apple/ml-ferret/blob/main/ferret/model/ferret_arch.py
class SamplerModel(PreTrainedModel):
    _auto_class = "AutoModel"
    config_class = SamplerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight, mean=0.0, std=std)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, 0.0)

    def __init__(self, config: SamplerConfig):
        super(SamplerModel, self).__init__(config)

        if config.sampler_type == "naive":
            self.model = NaiveSamplerModel(config)
        elif config.sampler_type == "geo":
            self.model = GeoSamplerModel(config)
        else:
            raise NotImplementedError(f"{config.sampler_type} is not supported.")

        self.post_init()

    def forward(self, vprompt_feats, vprompt_masks, original_dtype=None, return_dtype=None):
        return self.model(vprompt_feats, vprompt_masks, original_dtype, return_dtype)
