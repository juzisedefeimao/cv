
__sets = {}

# 目标检测
from .Resnet50_train_ohem import Resnet50_train_ohem

from .Resnet50_train import Resnet50_train
from .Resnet50_train_rpn import Resnet50_train_rpn
from .Resnet50_train_detect import Resnet50_train_detect
from .Resnet50_train_shared_conv import Resnet50_train_shared_conv
from .Resnet50_test_rpn import Resnet50_test_rpn
from .Resnet50_test_detect import Resnet50_test_detect

from .FPN_Resnet50_train import FPN_Resnet50_train
from .FPN_Resnet50_train_shared_conv import FPN_Resnet50_train_shared_conv
from .FPN_Resnet50_train_detect import FPN_Resnet50_train_detect
from .FPN_Resnet50_train_rpn import FPN_Resnet50_train_rpn
from .FPN_Resnet50_test import FPN_Resnet50_test

# 小样本目标检测与分类
from .Siammask_train import Siammask_train
from .Siammask_test import Siammask_test

# 分类器
from .Resnet18_classifier_train import Resnet18_classifier_train
from .Resnet18_classifier_test import Resnet18_classifier_test
from .Resnet18_fcn_classifier_train import Resnet18_fcn_classifier_train
from .Resnet18_fcn_classifier_test import Resnet18_fcn_classifier_test
from .Triple_classifier_train import Triple_classifier_train
from .Triple_classifier_test import Triple_classifier_test
# 小样本分类器
from .Relation_net import Relation_Network
from .Relation_net_test import Relation_Network_test
from .Relation_Resnet_net import Relation_Resnet_Network
# 对抗分类器
from .Gan_classifier_train import Gan_classifier_train
from .Triplegan_train import Triplegan_train
from .Triple import TripleGAN
# 元对抗生产
from .MetaGan_train import MeatGan_Network


def get_network(name):
    """Get a networks by name."""

    if name == 'Resnet50_train_ohem':
        return Resnet50_train_ohem

    elif name == 'Resnet50_train':
        return Resnet50_train()
    elif name == 'Resnet50_train_rpn':
        return Resnet50_train_rpn()
    elif name == 'Resnet50_train_detect':
        return Resnet50_train_detect()
    elif name == 'Resnet50_train_shared_conv':
        return Resnet50_train_shared_conv()
    elif name == 'Resnet50_test_rpn':
        return Resnet50_test_rpn()
    elif name == 'Resnet50_test_detect':
        return Resnet50_test_detect()

    elif name == 'FPN_Resnet50_train':
        return FPN_Resnet50_train()
    elif name == 'FPN_Resnet50_train_shared_conv':
        return FPN_Resnet50_train_shared_conv()
    elif name == 'FPN_Resnet50_train_rpn':
        return FPN_Resnet50_train_rpn()
    elif name == 'FPN_Resnet50_train_detect':
        return FPN_Resnet50_train_detect()
    elif name == 'FPN_Resnet50_test':
        return FPN_Resnet50_test()

    elif name == 'Siammask_train':
        return Siammask_train()
    elif name == 'Siammask_test':
        return Siammask_test()

    elif name == 'Resnet18_classifier_train':
        return Resnet18_classifier_train()
    elif name == 'Resnet18_classifier_test':
        return Resnet18_classifier_test()
    elif name == 'Resnet18_fcn_classifier_train':
        return Resnet18_fcn_classifier_train()
    elif name == 'Resnet18_fcn_classifier_test':
        return Resnet18_fcn_classifier_test()
    elif name == 'Triple_classifier_train':
        return Triple_classifier_train()
    elif name == 'Triple_classifier_test':
        return Triple_classifier_test()

    elif name == 'Relation_net_train':
        return Relation_Network()
    elif name == 'Relation_net_test':
        return Relation_Network_test()
    elif name == 'Relation_Resnet_net_train':
        return Relation_Resnet_Network()

    elif name == 'Gan_classifier_train':
        return Gan_classifier_train()
    elif name == 'Triplegan_train':
        return Triplegan_train()
    elif name == 'Triplegan':
        return TripleGAN()

    elif name == 'MetaGan_train':
        return MeatGan_Network()
    else:
        raise KeyError('Unknown dataset: {}'.format(name))

def list_networks():
    """List all registered imdbs."""
    return __sets.keys()