# Third-party licenses and research attribution

The displayed project/model names are OSS-CAEA, CAM and PSAH. Naming does not
change third-party ownership or claim that referenced architectures are original
contributions of this repository.

## Query-to-image embedding alignment

The PSAH alignment implementation follows the architecture described in
**Unified Embedding Alignment for Open-Vocabulary Video Instance Segmentation**
(ECCV 2024), by Hao Fang, Peng Wu, Yawei Li, Xinxin Zhang and Xiankai Lu.
Paper: https://arxiv.org/abs/2407.07427.

The source reference supplied ZeroShotClassifier/CrossAttentionLayer under
Apache-2.0. The implementation here was adapted for MMSegmentation. The original
license text is preserved verbatim in [THIRD-PARTY-LICENSE](THIRD-PARTY-LICENSE).
Modified/adapted files: oss_caea/framework/alignment.py and the PSAH integration.

## Base implementation and dependencies

The base implementation is associated with **Leveraging Depth and Language for
Open-Vocabulary Domain-Generalized Semantic Segmentation**. Its original
Apache-2.0 license remains in the repository-root LICENSE file.

Foundation models and libraries, including CLIP, DINOv2, Depth Anything V2,
MMSegmentation and MMDetection, retain their respective licenses and attribution.
Existing copyright notices in their retained source files are not removed.
