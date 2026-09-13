"""Frozen image/text encoders and the trainable OSS-CAEA CAM pathway."""
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from ..models.backbones.dino_v2 import DinoVisionTransformer
from ..models.heads.third_party.model import CLIP, build_model
from ..models.heads.third_party.clip import tokenize
from .cam import CAM
from .coarse_segmentation_head import CoarseSegmentationHead


@MODELS.register_module()
class OSSCAEABackbone(BaseModule):
    """Encoder-side implementation of the OSS-CAEA framework (Figure 2).

    Paper names -> registered attributes:
    VFM Image Encoder -> vfm; Depth Anything V2 -> depth_encoder;
    CLIP Image Encoder -> clip.visual; CLIP Text Encoder -> clip.encode_text;
    CAM -> cams; Coarse Segmentation Head -> coarse_head.
    """
    def __init__(self, class_json, vfm_checkpoint=None, depth_checkpoint=None,
                 clip_checkpoint=None, channels=256, out_indices=(21, 22, 23),
                 image_size=518, patch_size=14, embed_dim=1024, depth=24,
                 num_heads=16, clip_dim=768, prompt_templates=None,
                 allow_random_init=False, tiny_clip=False, init_cfg=None):
        super().__init__(init_cfg)
        self.checkpoints = (vfm_checkpoint, depth_checkpoint, clip_checkpoint)
        self.allow_random_init = allow_random_init
        if not allow_random_init:
            for name, path in zip(('DINOv2', 'Depth Anything V2', 'CLIP'), self.checkpoints):
                if not path or not Path(path).is_file():
                    raise FileNotFoundError(f'{name} checkpoint not found: {path}. See docs/OSS-CAEA.md')
        if not out_indices or min(out_indices) < 0 or max(out_indices) >= depth:
            raise ValueError('out_indices must select existing encoder layers')
        args = dict(img_size=image_size, patch_size=patch_size, embed_dim=embed_dim,
                    depth=depth, num_heads=num_heads, init_values=1e-5,
                    block_chunks=0, out_indices=list(out_indices))
        # DA V2-Large uses a DINOv2-Large encoder: load its pretrained.* weights.
        # Its depth prediction decoder is unused; no PromptDA/submodule import.
        self.vfm = DinoVisionTransformer(**args)
        self.depth_encoder = DinoVisionTransformer(**args)
        if clip_checkpoint:
            try:
                state = torch.jit.load(clip_checkpoint, map_location='cpu').state_dict()
            except RuntimeError:
                state = torch.load(clip_checkpoint, map_location='cpu')
                state = state.get('state_dict', state)
            state = dict(state)
            # Repository builder expects these metadata entries even for plain weights.
            for key in ('input_resolution', 'context_length', 'vocab_size'):
                state.setdefault(key, torch.tensor(0))
            self.clip = build_model(state).float()
        elif allow_random_init and tiny_clip:
            self.clip = CLIP(clip_dim, 28, 2, 64, 14, 77, 49408, 64, 1, 2).float()
            # Existing CLIP class expects loaded weights for these empty tensors.
            nn.init.normal_(self.clip.positional_embedding, std=.01)
            nn.init.normal_(self.clip.text_projection, std=64**-.5)
        else:
            raise ValueError('A CLIP checkpoint is required; tiny_clip is only for smoke tests')
        if not hasattr(self.clip.visual, 'conv1') or not hasattr(self.clip.visual, 'transformer'):
            raise ValueError('OSS-CAEA requires a ViT CLIP image encoder')
        actual_clip_dim = self.clip.text_projection.shape[1]
        if actual_clip_dim != clip_dim:
            raise ValueError(f'clip_dim={clip_dim}, checkpoint embedding dimension={actual_clip_dim}')
        self.patch_size = patch_size
        self.clip_patch_size = self.clip.visual.conv1.kernel_size[0]
        attn = self.clip.visual.transformer.resblocks[-1].attn
        self.clip_heads = attn.num_heads
        width = attn.embed_dim
        head_dim = width // self.clip_heads
        self.cams = nn.ModuleList([
            CAM(embed_dim, embed_dim, clip_dim, head_dim)
            for _ in out_indices])
        self.modulation_proj = nn.Conv2d(width, channels, 1)
        self.coarse_head = CoarseSegmentationHead(channels, clip_dim)
        self.pyramid_proj = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(4)])
        self.prompt_templates = prompt_templates or ['a photo of a {}.', 'a {} in the scene.']
        # Vocabulary is derived from current config and deliberately not checkpoint-persistent.
        self.register_buffer('text_features', torch.empty(0, clip_dim), persistent=False)
        self.register_buffer('imagenet_mean', torch.tensor([.485, .456, .406])[None, :, None, None], persistent=False)
        self.register_buffer('imagenet_std', torch.tensor([.229, .224, .225])[None, :, None, None], persistent=False)
        self.register_buffer('clip_mean', torch.tensor([.48145466, .4578275, .40821073])[None, :, None, None], persistent=False)
        self.register_buffer('clip_std', torch.tensor([.26862954, .26130258, .27577711])[None, :, None, None], persistent=False)
        for module in (self.vfm, self.depth_encoder, self.clip):
            module.requires_grad_(False).eval()
        self.set_vocabulary(class_json)

    @torch.no_grad()
    def set_vocabulary(self, class_json):
        classes = json.loads(Path(class_json).read_text()) if isinstance(class_json, (str, Path)) else list(class_json)
        if classes and classes[-1].lower() in ('background', 'no object', 'no-object'):
            classes = classes[:-1]
        if not classes:
            raise ValueError('Vocabulary must contain foreground categories')
        features = []
        for name in classes:
            prompts = [template.format(alias.strip()) for template in self.prompt_templates
                       for alias in name.split(',') if alias.strip()]
            encoded = self.clip.encode_text(tokenize(prompts).to(self.clip.positional_embedding.device)).float()
            features.append(F.normalize(F.normalize(encoded, dim=-1).mean(0), dim=-1))
        self.text_features = torch.stack(features)
        if not torch.isfinite(self.text_features).all():
            raise ValueError('CLIP text encoder produced non-finite embeddings')
        self.class_names = classes

    def init_weights(self):
        if self._is_init:
            return
        for module, path, prefix in ((self.vfm, self.checkpoints[0], ''),
                                     (self.depth_encoder, self.checkpoints[1], 'pretrained.')):
            if path:
                state = torch.load(path, map_location='cpu')
                state = state.get('state_dict', state)
                if prefix:
                    state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
                    if not state:
                        raise ValueError('DA V2 checkpoint must contain pretrained.* encoder weights')
                # DINO classification head is Identity here; checkpoint norm is retained.
                module.load_state_dict(state, strict=True)
        self._is_init = True

    def train(self, mode=True):
        super().train(mode)
        for module in (self.vfm, self.depth_encoder, self.clip):
            module.eval()
        return self

    @torch.no_grad()
    def _clip_features(self, rgb):
        visual = self.clip.visual
        x = visual.conv1((rgb - self.clip_mean) / self.clip_std)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((visual.class_embedding[None, None].expand(b, 1, c), x), dim=1)
        pos = visual.positional_embedding
        side = math.isqrt(pos.shape[0] - 1)
        grid = pos[1:].T.reshape(1, c, side, side)
        grid = F.interpolate(grid, (h, w), mode='bicubic', align_corners=False)
        pos = torch.cat((pos[:1], grid.flatten(2).squeeze(0).T))
        x = visual.ln_pre(x + pos).transpose(0, 1)
        blocks = visual.transformer.resblocks
        for block in blocks[:-1]:
            x = block(x)
        last = blocks[-1]
        normed = last.ln_1(x)
        qkv = F.linear(normed, last.attn.in_proj_weight, last.attn.in_proj_bias)
        qkv = qkv.permute(1, 0, 2).reshape(b, h*w+1, 3, self.clip_heads, c//self.clip_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        per_head = F.scaled_dot_product_attention(q, k, v)
        # Preserve genuine last-layer multihead outputs for Eq. (8).
        head_maps = per_head[:, :, 1:].permute(0, 1, 3, 2).reshape(b, self.clip_heads, c//self.clip_heads, h, w)
        final = last(x).transpose(0, 1)
        memory = visual.ln_post(final[:, 1:]) @ visual.proj
        return head_maps.float(), memory.float()

    def forward(self, x):
        # MMSeg preprocessor provides ImageNet-normalized RGB, not raw BGR.
        rgb = (x * self.imagenet_std + self.imagenet_mean).clamp(0, 1)
        multiple = math.lcm(self.patch_size, self.clip_patch_size)
        h, w = x.shape[-2:]
        padded = (math.ceil(h/multiple)*multiple, math.ceil(w/multiple)*multiple)
        rgb = F.pad(rgb, (0, padded[1]-w, 0, padded[0]-h), mode='replicate')
        with torch.no_grad():
            normalized = (rgb - self.imagenet_mean) / self.imagenet_std
            visual = self.vfm.forward_features(normalized)
            geometry = self.depth_encoder.forward_features(normalized)
            clip_heads, memory = self._clip_features(rgb)
        gates = [cam(v, d, self.text_features) for cam, v, d in zip(self.cams, visual, geometry)]
        gates = [F.interpolate(g, clip_heads.shape[-2:], mode='bilinear', align_corners=False) for g in gates]
        # Equal cross-layer mean: paper does not specify its layer reduction.
        gate = torch.stack(gates).mean(0)
        modulated = clip_heads * gate[:, None]
        b, heads, channels, gh, gw = modulated.shape
        fused = self.modulation_proj(modulated.reshape(b, heads*channels, gh, gw))
        # Crop padded area before creating the stride 4/8/16/32 pyramid.
        coarse_size = (max(1, math.ceil(h/4)), max(1, math.ceil(w/4)))
        fused = F.interpolate(fused, (math.ceil(padded[0]/4), math.ceil(padded[1]/4)),
                              mode='bilinear', align_corners=False)[..., :coarse_size[0], :coarse_size[1]]
        coarse_features, coarse_logits = self.coarse_head(fused, self.text_features, coarse_size)
        pyramid = [proj(F.interpolate(coarse_features, (max(1, math.ceil(h/s)), max(1, math.ceil(w/s))),
                                     mode='bilinear', align_corners=False))
                   for proj, s in zip(self.pyramid_proj, (4, 8, 16, 32))]
        return dict(features=pyramid, clip_memory=memory, text_features=self.text_features,
                    coarse_logits=coarse_logits)
