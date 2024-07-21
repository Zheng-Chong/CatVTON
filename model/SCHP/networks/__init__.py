from __future__ import absolute_import

from model.SCHP.networks.AugmentCE2P import resnet101

__factory = {
    'resnet101': resnet101,
}


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model arch: {}".format(name))
    return __factory[name](*args, **kwargs)