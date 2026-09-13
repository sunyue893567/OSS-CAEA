"""Pixel-Semantic Alignment Head (PSAH) for OSS-CAEA."""
import torch
from torch.nn import functional as F
from mmseg.registry import MODELS
from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
from .alignment import UnifiedEmbeddingAlignment
from .coarse_segmentation_head import coarse_segmentation_loss


@MODELS.register_module()
class PSAH(Mask2FormerHead):
    """Pixel-Semantic Alignment Head (PSAH) with query-to-image alignment."""
    def __init__(self, clip_dim=768, alignment_heads=8, temperature=50.,
                 coarse_loss_weight=1., coarse_alpha=0.4, **kwargs):
        super().__init__(**kwargs)
        del self.cls_embed
        self.alignment = UnifiedEmbeddingAlignment(kwargs['feat_channels'], clip_dim,
                                                    alignment_heads, temperature)
        self.coarse_loss_weight = coarse_loss_weight
        self.coarse_alpha = coarse_alpha

    def _prediction(self, query, mask_features, memory, text, target_size):
        query = self.transformer_decoder.post_norm(query)
        classes = self.alignment(query, memory, text)
        masks = torch.einsum('bqc,bchw->bqhw', self.mask_embed(query), mask_features)
        attention = F.interpolate(masks, target_size, mode='bilinear', align_corners=False)
        attention = (attention.sigmoid().flatten(2).unsqueeze(1)
                     .expand(-1, self.num_heads, -1, -1).flatten(0, 1) < .5).detach()
        return classes, masks, attention

    def forward(self, inputs, batch_data_samples):
        text = inputs['text_features']
        if len(text) != self.num_classes:
            raise ValueError(f'Vocabulary has {len(text)} classes, head expects {self.num_classes}')
        mask_features, memories = self.pixel_decoder(inputs['features'])
        batch = mask_features.shape[0]
        decoder_inputs, positions = [], []
        for i in range(self.num_transformer_feat_level):
            feat = self.decoder_input_projs[i](memories[i])
            decoder_inputs.append(feat.flatten(2).transpose(1, 2) + self.level_embed.weight[i])
            mask = feat.new_zeros(batch, *feat.shape[-2:], dtype=torch.bool)
            positions.append(self.decoder_positional_encoding(mask).flatten(2).transpose(1, 2))
        query_pos = self.query_embed.weight[None].expand(batch, -1, -1)
        query = self.query_feat.weight[None].expand(batch, -1, -1)
        classes, masks, attention = self._prediction(query, mask_features, inputs['clip_memory'], text,
                                                    memories[0].shape[-2:])
        all_classes, all_masks = [classes], [masks]
        for i, layer in enumerate(self.transformer_decoder.layers):
            level = i % self.num_transformer_feat_level
            attention = attention.clone()
            attention[attention.all(-1)] = False
            query = layer(query=query, key=decoder_inputs[level], value=decoder_inputs[level],
                          query_pos=query_pos, key_pos=positions[level], cross_attn_mask=attention,
                          query_key_padding_mask=None, key_padding_mask=None)
            classes, masks, attention = self._prediction(
                query, mask_features, inputs['clip_memory'], text,
                memories[(i+1) % self.num_transformer_feat_level].shape[-2:])
            all_classes.append(classes)
            all_masks.append(masks)
        return all_classes, all_masks

    def loss(self, inputs, batch_data_samples, train_cfg):
        # Final masks/classes and auxiliary decoder layers retain Hungarian matching,
        # CE/BCE/Dice losses. Coarse CE+Dice alone cannot train downstream PSAH.
        losses = super().loss(inputs, batch_data_samples, train_cfg)
        target = torch.stack([sample.gt_sem_seg.data for sample in batch_data_samples])
        losses['loss_coarse'] = self.coarse_loss_weight * coarse_segmentation_loss(
            inputs['coarse_logits'], target, self.ignore_index, self.coarse_alpha)
        return losses
