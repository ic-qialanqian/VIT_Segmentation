from .metrics import averageMeter, runningScore
from .log import get_logger
from .optim import Ranger

from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, \
    compute_speed, setup_seed, group_weight_decay


def get_dataset(cfg):
    assert cfg['dataset'] in ['nyuv2', 'nyuv2_new', 'sunrgbd', 'cityscapes', 'camvid', 'irseg', 'pst900', 'irseg_msv', 'irseg_samcl']

    if cfg['dataset'] == 'irseg':
        from .datasets.irseg import IRSeg
        # return IRSeg(cfg, mode='trainval'), IRSeg(cfg, mode='test')
        return IRSeg(cfg, mode='train'), IRSeg(cfg, mode='val'), IRSeg(cfg, mode='test')
        
    elif cfg['dataset'] == 'irseg_samcl':
        from .datasets.irseg_samcl import IRSeg
        
        return IRSeg(cfg, mode='train'), IRSeg(cfg, mode='val'), IRSeg(cfg, mode='test')
        
    elif cfg['dataset'] == 'pst900':
        from .datasets.pst900 import PSTSeg
        # return IRSeg(cfg, mode='trainval'), IRSeg(cfg, mode='test')
        return PSTSeg(cfg, mode='train'), PSTSeg(cfg, mode='val'), PSTSeg(cfg, mode='test')


def get_model(cfg):

    if cfg['model_name'] == 'EGFNet':
        from .models.EGFNet import EGFNet
        return EGFNet(n_classes=cfg['n_classes'])
     
    elif cfg['model_name']== 'LASNet_base':
        from .models.LASNet_base import LASNet
        return LASNet(n_classes=cfg['n_classes'])

    elif cfg['model_name']== 'LASNet_base_4ch':
        from .models.LASNet_base_4ch import LASNet
        return LASNet(n_classes=cfg['n_classes'])

    elif cfg['model_name']== 'LASNet_pretrain':
        from .models.LASNet_pretrain import LASNet
        return LASNet(n_classes=cfg['n_classes'])
    
    elif cfg['model_name']== 'LASNet_base_samcl':
        from .models.LASNet_base_samcl import LASNet
        return LASNet(n_classes=cfg['n_classes'])
        
    else:
        from .models.LASNet import LASNet
        return LASNet(n_classes=cfg['n_classes'])
        #return LASNet(n_classes=5)
