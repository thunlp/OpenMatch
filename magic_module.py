import copy
import operator
import torch
import torch.nn as nn
from transformers.modeling_bert import BertSelfAttention


class MagicModule(nn.Module):
    def __init__(self, module):
        nn.Module.__init__(self)
        self._type = type(module)

        ## Copy Original Model's parameter and buffer to all buffer
        for key, value in module._parameters.items():
            self.register_parameter('_origin_' + key, value)
            self.register_buffer(key, nn.Parameter(value.data))

        for key, value in module._buffers.items():
            self.register_buffer(key, copy.deepcopy(value))

        for key, value in module._modules.items():
            self.add_module(key, MagicModule(value))

        for key, value in module.__dict__.items():
            if (not key in self.__dict__) and\
                    (not key in self._buffers) and\
                    (not key in self._modules):
                self.__setattr__(key, value)

    def forward(self, *args, **kwargs):
        return self._type.forward(self, *args, **kwargs)

    ## delta in SGD is -lr*grad
    def update_params(self, deltas):
        sub_params = {}
        for key, delta in deltas.items():
            if not ('.' in key):
                self._buffers[key] = self._buffers[key] + delta
            else:
                attr = key.split('.')[0]
                if not (attr in sub_params):
                    sub_params[attr] = {}
                sub_params[attr]['.'.join(key.split('.')[1:])] = delta
        for key, value in sub_params.items():
            self._modules[key].update_params(value)

    def check_forward_args(self, *args, **kwargs):
        assert issubclass(self._type, nn.RNNBase)
        return nn.RNNBase.check_forward_args(self, *args, **kwargs)

    @property
    def _flat_weights(self):
        assert issubclass(self._type, nn.RNNBase)
        return [p for layerparams in self.all_weights for p in layerparams]

    @property
    def all_weights(self):
        assert issubclass(self._type, nn.RNNBase)
        return [[getattr(self, weight) for weight in weights] for weights in
                self._all_weights]

    def _get_abs_string_index(self, idx):
        assert issubclass(self._type, nn.ModuleList)
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        assert issubclass(self._type, nn.ModuleList)
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __len__(self):
        assert issubclass(self._type, nn.ModuleList)
        return len(self._modules)

    def transpose_for_scores(self, x):
        assert issubclass(self._type, BertSelfAttention)
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
