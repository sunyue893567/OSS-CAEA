"""EncoderDecoder integration for feature bundles and crop-local mask sizes."""
from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder


@MODELS.register_module()
class OSSCAEA(EncoderDecoder):
    """OSS-CAEA: Collaborative Attention and Embedding Alignment framework."""
    def encode_decode(self, inputs, batch_img_metas):
        # MMSeg slide_inference updates img_shape but leaves full-image pad_shape.
        # Mask2Former predict prefers pad_shape: pass the actual crop dimensions.
        metadata = [dict(meta, pad_shape=inputs.shape[-2:]) for meta in batch_img_metas]
        return self.decode_head.predict(self.extract_feat(inputs), metadata, self.test_cfg)

    def set_vocabulary(self, classes):
        """Change inference vocabulary without replacing learned classifier weights."""
        if self.training:
            raise RuntimeError('Switch vocabularies in eval mode; training also needs matching dataset/loss config')
        self.backbone.set_vocabulary(classes)
        count = len(self.backbone.class_names)
        self.decode_head.num_classes = self.decode_head.out_channels = count
        self.num_classes = self.out_channels = count
