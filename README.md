# OSS-CAEA

Open-vocabulary domain-generalized semantic segmentation with Collaborative
Attention Module (**CAM**) and Pixel-Semantic Alignment Head (**PSAH**).
The project directory is `OSS-CAEA`; the importable Python package is `oss_caea`.

## Model components

| Component | Python class | Source |
| --- | --- | --- |
| Model | `OSSCAEA` | `oss_caea/framework/oss_caea.py` |
| Frozen encoders and CAM pathway | `OSSCAEABackbone` | `oss_caea/framework/backbone.py` |
| Collaborative Attention Module | `CAM` | `oss_caea/framework/cam.py` |
| Coarse Segmentation Head | `CoarseSegmentationHead` | `oss_caea/framework/coarse_segmentation_head.py` |
| Pixel-Semantic Alignment Head | `PSAH` | `oss_caea/framework/psah.py` |
| PSAH embedding alignment | `UnifiedEmbeddingAlignment` | `oss_caea/framework/alignment.py` |

## Training and evaluation

Prepare the checkpoints and datasets described in [the setup guide](docs/OSS-CAEA.md).

```bash
cd /home/featurize/Code/OSS-CAEA
.venv-oss-caea/bin/python tools/train.py configs/oss_caea/oss_caea_cityscapes.py
.venv-oss-caea/bin/python tools/train.py configs/oss_caea/oss_caea_gta.py
.venv-oss-caea/bin/python tools/test.py configs/oss_caea/oss_caea_deliver.py \
  work_dirs/oss_caea_cityscapes/iter_80000.pth
```

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-oss-caea/bin/python -m pytest tests/test_oss_caea.py -q
```

See [architecture, assumptions and checkpoint requirements](docs/OSS-CAEA.md).
Third-party license and research credits are recorded in [the notices](docs/THIRD-PARTY-NOTICES.md).

See [the PDF-to-code naming audit](docs/NAMING-AUDIT.md) for all module names.
