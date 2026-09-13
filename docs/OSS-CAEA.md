# OSS-CAEA

The complete OSS-CAEA implementation lives in this repository.
It uses MMSegmentation and has no sibling-repository or Detectron2 runtime dependency.
The configs/oss_caea configurations select the OSSCAEA segmentor.

## Implemented computation graph

RGB -> frozen DINOv2 + frozen DA V2 encoder -> per-layer CAM -> mean over layers
                                                       |
RGB -> frozen CLIP image encoder -> final attention head outputs * CAM weights
                                                       |
                                  learned head merge -> coarse head + coarse loss
                                                       |
                                coarse semantic prior + latent feature pyramid
                                                       |
                                  Pixel Decoder -> Transformer Decoder queries
                                                       |
CLIP image patch embeddings -> PSAH cross-attention (K,V), queries (Q)
                                                       |
shared CLIP text embeddings -> cosine classification + predicted masks -> segmentation

The CLIP text encoder is run once per configured vocabulary. The same cached
embeddings are reused by CAM, the coarse head and PSAH during training/inference.
DINOv2, DA V2 and CLIP are frozen with requires_grad=False, eval() and no_grad().
All CAM projections, coarse modules, the feature pyramid, pixel/transformer
decoders and the query alignment module are trainable.

## Source mapping

- oss_caea/framework/cam.py: CAM implements PDF Eq. (4)-(7).
  Spatial tokens retain their ordering; cosine weights are signed, with no added
  sigmoid, softmax or weighted-sum normalization.
- oss_caea/framework/coarse_segmentation_head.py: CoarseSegmentationHead and coarse loss.
- oss_caea/framework/alignment.py: UnifiedEmbeddingAlignment inside PSAH.
- oss_caea/framework/backbone.py: frozen encoders, shared text cache, true per-head
  final CLIP self-attention outputs, Eq. (8) elementwise modulation, head merge,
  coarse head and a stride 4/8/16/32 feature pyramid.
- oss_caea/framework/psah.py: MMSeg Mask2Former pixel decoder, masked transformer
  query decoder, CLIP-memory alignment, category/mask outputs and training loss.
- oss_caea/framework/oss_caea.py: EncoderDecoder integration, crop-local sizes for
  sliding inference, and an eval-only set_vocabulary API.
- configs/_base_/models/oss_caea.py: production model configuration.
- configs/oss_caea/: Cityscapes/GTA source training and DELIVER evaluation.
- tests/test_oss_caea.py: numerical and complete-model checks without downloads.

## Explicit resolutions of inconsistencies in the PDF

This is a runnable interpretation of the supplied manuscript, not a claim of
bit-for-bit equivalence to an unavailable OSS-CAEA reference implementation.

1. CAM inputs follow Figure 2 and Section 3: DINOv2 VFM and DA V2 features. The
   CLIPI label used for the VFM stream in Eq. (4) is interpreted as a notation
   inconsistency. We use the last three transformer blocks (zero-based 21,22,23
   for Large); they have the same native spatial grid, not different image scales.
2. A trainable 1x1 projection aligns each encoder stream to the CLIP attention
   head dimension before addition/LayerNorm. Text uses a learned linear mapping.
   The global textual vector averages foreground category embeddings; background
   is excluded. This is a global semantic gate, not per-class token attention.
3. Eq. (7) and Figure 3 take precedence over weighted-average prose: each token is
   multiplied by its cosine similarity and retained. Mean reduction over the
   selected layers is an explicit implementation choice (not specified by PDF).
4. Eq. (8) takes precedence over the Figure 2 MatMul label. The shared CAM gate
   is broadcast over actual CLIP heads and multiplied elementwise. A learned
   1x1 projection merges the heads. Spatial resizing is explicit and supports
   rectangular inputs. Frozen CLIP patch embeddings are reused by PSAH as memory.
5. The coarse head follows Eq. (10): 3x3 Conv -> BN -> ReLU -> bilinear upsampling
   -> 1x1 semantic projection. Because a learned fixed-K convolution cannot
   change categories at evaluation, its category kernels are normalized shared
   text embeddings. This dynamic classifier is an explicit open-vocabulary
   extension of the fixed-class formula. Coarse logits have K foreground channels.
6. Coarse probabilities pool text embeddings into a dense semantic prior. Its
   projected features are added to the coarse latent map, which supplies the PSAH
   feature pyramid. We do not send a K-channel score map into an RGB-pretrained
   CLIP/backbone. The original frozen CLIP branch is reused; there is no second
   image encoding of segmentation probabilities. This resolves the incompatible
   x_clip/coarse-output interface in Figure 4 and Section 3.2.
7. PSAH uses learned queries, a two-layer
   query MLP, eight-head CLIP-memory attention, residual + LayerNorm, L2-normalized
   classification and temperature 50. Q/K/V projections are inside MultiheadAttention.
   We do not apply the PDF Eq. (14) softmax to text embeddings. The no-object
   classifier is a zero vector. Video tracking/association is
   omitted because this model performs image semantic segmentation.
8. Coarse loss is 0.4 CE + 0.6 multiclass soft Dice with ignored pixels excluded.
   The complete loss additionally retains Mask2Former Hungarian query matching,
   final/auxiliary class CE and mask BCE/Dice. Without downstream losses, coarse
   supervision alone would not train PSAH. The coarse multiplier is configurable.
9. The 80k schedule drops LR by 0.1 at 64k and 68k iterations. Its last LR is
   1e-6; the PDF statement that these two drops reach zero is not mathematically
   consistent and is not implemented as an extra undocumented scheduler.

## Environment

A validation environment was created at .venv-oss-caea inside this repository.
It uses PyTorch 2.1.2/cu121, torchvision 0.16.2, MMCV 2.1.0, MMEngine 0.10.7,
MMSegmentation 1.2.2 and MMDetection 3.3.0. MMCV must include its compiled ops.
This is separate from the server's original PyTorch environment.

For a fresh environment (Python 3.11):

```bash
cd /home/featurize/Code/OSS-CAEA
python -m venv .venv-oss-caea
.venv-oss-caea/bin/python -m pip install -r requirements-oss-caea.txt \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```

## Pretrained weights and datasets

Place the following original checkpoints in checkpoints/:

- dinov2_vitl14_pretrain.pth: raw DINOv2 ViT-L/14 encoder state dict.
- depth_anything_v2_vitl.pth: original DA V2-Large state dict containing pretrained.*.
- ViT-L-14.pt: original CLIP ViT-L/14 TorchScript checkpoint or compatible state dict.

Use original patch-14 checkpoints rather than converted patch-16 checkpoints. This model uses native patch 14
and pads RGB to complete patches. DA V2's pretrained encoder is implemented with
the shared DINOv2 class; its depth decoder is not used.
Missing or incompatible weights fail explicitly; production never silently freezes
random encoders. CLIP is loaded on CPU, then follows the model device. Text caches
are buffers that move with the model but are regenerated from the chosen vocabulary.

Use the original repository dataset preparation/layout for Cityscapes, GTA,
BDD100K, Mapillary, ACDC and DELIVER. Dataset paths can be overridden in configs.
No dataset conversion, downloads or full training are performed by model creation.

## Training and evaluation

```bash
cd /home/featurize/Code/OSS-CAEA
# One GPU; effective batch is 2. Four GPUs with this config give total batch 8.
.venv-oss-caea/bin/python tools/train.py configs/oss_caea/oss_caea_cityscapes.py
# GTA source training
.venv-oss-caea/bin/python tools/train.py configs/oss_caea/oss_caea_gta.py
# Four GPUs
.venv-oss-caea/bin/python -m torch.distributed.run --nproc_per_node=4 \
  tools/train.py configs/oss_caea/oss_caea_cityscapes.py --launcher pytorch
# Open-vocabulary DELIVER evaluation with an OSS-CAEA checkpoint.
.venv-oss-caea/bin/python tools/test.py configs/oss_caea/oss_caea_deliver.py \
  work_dirs/oss_caea_cityscapes/iter_80000.pth
# Validation (small randomly initialized encoders, no data/checkpoint download)
PYTHONDONTWRITEBYTECODE=1 .venv-oss-caea/bin/python -m pytest tests/test_oss_caea.py -q
```

Training checkpoints include frozen encoder weights under normal MMEngine saving;
use the complete model checkpoint. The external
pretrained files are still required for initial model construction. Changing the
configured class JSON and head num_classes permits checkpoint loading with a new
vocabulary because there is no trainable category-count-dependent classifier matrix.
For an already loaded model in eval mode, use model.set_vocabulary(class_names).

## Validation scope

Recorded validation: 8 tests passed on 2026-09-13, including CPU and CUDA
on the server RTX 2080 Ti. Both tools/train.py --help and tools/test.py --help
load successfully in the isolated environment.

Tests cover signed-cosine CAM behavior, ignored coarse labels, full CPU/CUDA
forward/backward and optimizer updates, frozen parameters, CLIP-memory influence,
strict checkpoint round trips, vocabulary changes, rectangular and sliding inference,
production config resolution, encoder checkpoint formats, MMEngine train_step and AMP.
Small random encoders exercise the same implementation with reduced dimensions.
These checks do not establish segmentation accuracy or reproduce the PDF metrics.
Production pretrained weights and datasets are required for that evaluation.

## Third-party notices

Research attribution and preserved license terms are recorded in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

## Project naming

The repository root is /home/featurize/Code/OSS-CAEA and the Python package is
oss_caea. Public names are OSSCAEA, OSSCAEABackbone, CAM, CoarseSegmentationHead and PSAH.
Parameter names/state_dict keys are unchanged, so existing OSS-CAEA weights load.
Only the current model implementation and shared foundation-model utilities are
retained in this working tree. Superseded model implementations and configs were
removed. See NAMING-AUDIT.md for the current Figure 2/3/4 name-to-code mapping.
