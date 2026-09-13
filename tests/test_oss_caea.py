"""Executable numerical, integration and checkpoint checks, without downloads."""
import copy
import io
from pathlib import Path
import pytest
import torch
from mmengine.config import Config
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules
import oss_caea
from oss_caea.framework.cam import CAM
from oss_caea.framework.coarse_segmentation_head import coarse_segmentation_loss

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
register_all_modules()


def tiny_model(device='cpu'):
    cfg = Config.fromfile(str(ROOT/'configs/_base_/models/oss_caea.py')).model
    cfg.backbone.update(class_json=['road','car','background'], vfm_checkpoint=None,
                        depth_checkpoint=None, clip_checkpoint=None, allow_random_init=True,
                        tiny_clip=True, channels=32, clip_dim=64, image_size=28,
                        patch_size=14, embed_dim=32, depth=2, num_heads=4, out_indices=(0,1))
    cfg.data_preprocessor.size=(56,70)
    cfg.decode_head.update(num_classes=2, clip_dim=64, alignment_heads=1,
                           in_channels=[32]*4, feat_channels=32, out_channels=32,
                           num_queries=5)
    cfg.decode_head.loss_cls.class_weight=[1.,1.,.1]
    p=cfg.decode_head.pixel_decoder
    p.norm_cfg.num_groups=4
    p.encoder.num_layers=1
    p.encoder.layer_cfg.self_attn_cfg.update(embed_dims=32,num_heads=4)
    p.encoder.layer_cfg.ffn_cfg.update(embed_dims=32,feedforward_channels=64)
    p.positional_encoding.num_feats=16
    d=cfg.decode_head.transformer_decoder
    d.num_layers=2
    d.layer_cfg.self_attn_cfg.update(embed_dims=32,num_heads=4)
    d.layer_cfg.cross_attn_cfg.update(embed_dims=32,num_heads=4)
    d.layer_cfg.ffn_cfg.update(embed_dims=32,feedforward_channels=64)
    cfg.decode_head.positional_encoding.num_feats=16
    cfg.decode_head.train_cfg.num_points=32
    cfg.test_cfg.update(mode='whole')
    model=MODELS.build(cfg).to(device)
    model.init_weights()
    return model


def samples(device='cpu', h=56, w=70):
    result=[]
    for i in range(2):
        sample=SegDataSample(metainfo=dict(img_shape=(h,w),ori_shape=(h,w),pad_shape=(h,w),
                                         padding_size=[0,0,0,0]))
        target=torch.zeros(1,h,w,dtype=torch.long,device=device)
        target[:,:,w//2:]=1
        target[:,:3,:3]=255
        sample.gt_sem_seg=PixelData(data=target)
        result.append(sample)
    return result


def test_cam_signed_cosine_and_spatial_order():
    cam=CAM(2,2,2,2)
    with torch.no_grad():
        cam.visual_proj.weight.copy_(torch.eye(2).reshape(2,2,1,1))
        cam.visual_proj.bias.zero_()
        cam.depth_proj.weight.zero_(); cam.depth_proj.bias.zero_()
        cam.text_proj.weight.copy_(torch.eye(2)); cam.text_proj.bias.zero_()
    visual=torch.tensor([[[[1.,-1.,0.]], [[-1.,1.,0.]]]])
    result=cam(visual, torch.zeros_like(visual), torch.tensor([[1.,-1.]]))
    # Negative cosine flips the opposite token; zero stays finite/zero.
    torch.testing.assert_close(result[:,:,:,0],result[:,:,:,1])
    assert result[0,0,0,0] > .99 and result[0,1,0,0] < -.99
    assert torch.equal(result[:,:,:,2],torch.zeros_like(result[:,:,:,2]))
    assert result.shape == visual.shape


def test_coarse_loss_ignore_and_correct_prediction():
    truth=torch.zeros(1,8,8,dtype=torch.long)
    good=torch.stack((torch.full((1,8,8),10.),torch.full((1,8,8),-10.)),1).requires_grad_()
    bad=-good
    assert coarse_segmentation_loss(good,truth)<coarse_segmentation_loss(bad,truth)
    ignored=coarse_segmentation_loss(good,torch.full_like(truth,255))
    ignored.backward()
    assert ignored.item()==0 and torch.equal(good.grad,torch.zeros_like(good))


@pytest.mark.parametrize('device',['cpu']+(['cuda'] if torch.cuda.is_available() else []))
def test_full_model_training_inference_checkpoint_and_vocabulary(device):
    torch.manual_seed(12)
    model=tiny_model(device).train()
    image=torch.randn(2,3,56,70,device=device)
    data=samples(device)
    losses=model.loss(image,data)
    assert 'decode.loss_coarse' in losses
    loss=sum(v for k,v in losses.items() if 'loss' in k)
    assert torch.isfinite(loss)
    loss.backward()
    for frozen in (model.backbone.vfm,model.backbone.depth_encoder,model.backbone.clip):
        assert not frozen.training
        assert all(not p.requires_grad and p.grad is None for p in frozen.parameters())
    for module in (model.backbone.cams,model.backbone.coarse_head,model.decode_head.alignment,
                   model.decode_head.pixel_decoder,model.decode_head.transformer_decoder):
        grads=[p.grad for p in module.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum()>0 for g in grads)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    optimizer.step(); optimizer.zero_grad()
    model.eval()
    with torch.no_grad():
        features=model.extract_feat(image)
        a,masks=model.decode_head(features,data)
        changed=dict(features,clip_memory=features['clip_memory'].flip(2))
        b,masks2=model.decode_head(changed,data)
        assert not torch.allclose(a[-1],b[-1])
        torch.testing.assert_close(masks[-1],masks2[-1])
        predictions=model.predict(image,data)
        assert predictions[0].seg_logits.data.shape==(2,56,70)
        # Checkpoint reload must preserve deterministic output and omit vocabulary caches.
        state=copy.deepcopy(model.state_dict())
        assert not any(k.endswith('text_features') for k in state)
        expected=model.encode_decode(image,[s.metainfo for s in data])
        model.load_state_dict(state,strict=True)
        torch.testing.assert_close(expected,model.encode_decode(image,[s.metainfo for s in data]))
        model.set_vocabulary(['road','car','tree','background'])
        model.load_state_dict(state,strict=True)
        output=model.encode_decode(image,[s.metainfo for s in data])
        assert output.shape==(2,3,56,70) and torch.isfinite(output).all()
        # Non-square, non-patch-divisible input through full segmentor.
        odd=torch.randn(1,3,53,67,device=device)
        odd_data=samples(device,53,67)[:1]
        assert model.predict(odd,odd_data)[0].seg_logits.data.shape==(3,53,67)
        # Sliding-window inference must retain selected vocabulary.
        model.test_cfg.update(mode='slide',crop_size=(42,42),stride=(28,28))
        assert model.predict(odd,odd_data)[0].seg_logits.data.shape==(3,53,67)


def test_production_config_resolution():
    for name in ('cityscapes','gta','deliver'):
        cfg=Config.fromfile(str(ROOT/f'configs/oss_caea/oss_caea_{name}.py'))
        assert cfg.model.backbone.type=='OSSCAEABackbone'
        assert cfg.model.decode_head.type=='PSAH'
        assert cfg.model.decode_head.num_classes==(25 if name=='deliver' else 19)
        if name!='deliver':
            assert cfg.train_cfg.max_iters==80000
    with pytest.raises(FileNotFoundError,match='checkpoint'):
        MODELS.build(dict(type='OSSCAEABackbone',class_json=['road']))


def test_encoder_checkpoint_formats_and_strict_loading(tmp_path):
    model=tiny_model().eval()
    vfm=tmp_path/'dino.pth'; depth=tmp_path/'depth.pth'; clip=tmp_path/'clip.pt'
    torch.save(model.backbone.vfm.state_dict(),vfm)
    depth_state={'pretrained.'+k:v for k,v in model.backbone.depth_encoder.state_dict().items()}
    depth_state['depth_head.unused']=torch.ones(1)
    torch.save(depth_state,depth)
    torch.save(model.backbone.clip.state_dict(),clip)
    cfg=dict(type='OSSCAEABackbone',class_json=['road','car'],vfm_checkpoint=str(vfm),
             depth_checkpoint=str(depth),clip_checkpoint=str(clip),channels=32,clip_dim=64,
             image_size=28,patch_size=14,embed_dim=32,depth=2,num_heads=4,out_indices=(0,1))
    restored=MODELS.build(cfg); restored.init_weights()
    for key,value in model.backbone.vfm.state_dict().items():
        torch.testing.assert_close(value,restored.vfm.state_dict()[key])
    torch.testing.assert_close(model.backbone.text_features,restored.text_features,rtol=2e-3,atol=2e-3)
    depth_state.pop('pretrained.patch_embed.proj.weight')
    torch.save(depth_state,depth)
    with pytest.raises(RuntimeError,match='Missing key'):
        MODELS.build(cfg).init_weights()


def test_mmengine_train_step_and_optimizer_filter():
    from mmengine.optim import build_optim_wrapper
    model=tiny_model()
    config=Config.fromfile(str(ROOT/'configs/oss_caea/oss_caea_cityscapes.py'))
    wrapper=build_optim_wrapper(model,config.optim_wrapper)
    selected={id(p) for group in wrapper.optimizer.param_groups for p in group['params']}
    assert selected=={id(p) for p in model.parameters() if p.requires_grad}
    before=model.backbone.cams[0].text_proj.weight.detach().clone()
    batch=dict(inputs=[torch.randint(0,256,(3,56,70),dtype=torch.uint8) for _ in range(2)],
               data_samples=samples())
    logs=model.train_step(batch,wrapper)
    assert torch.isfinite(logs['loss'])
    assert not torch.equal(before,model.backbone.cams[0].text_proj.weight)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA is required for AMP')
def test_cuda_amp_forward_backward():
    model=tiny_model('cuda').train()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    scaler=torch.cuda.amp.GradScaler(init_scale=16.)
    with torch.autocast('cuda',dtype=torch.float16):
        losses=model.loss(torch.randn(2,3,56,70,device='cuda'),samples('cuda'))
        total=sum(value for key,value in losses.items() if 'loss' in key)
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    assert torch.isfinite(total)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    scaler.step(optimizer); scaler.update()
