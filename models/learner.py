import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np

class Learner(nn.Module):
    """

    """

    def __init__(self, backbone, heads=None, parameters=None, imgc=3, imgsz=417):
        """

        Parameters:
        ----------
            backbone: The backbone of network. expected type: list of (string, list)
            heads: The heads of network. expected type: lists of list of (string, list)
        """
        super(Learner, self).__init__()


        self.backbone = backbone
        self.heads = heads

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()
        parameters_idx = 0
        for i, (name, param) in enumerate(self.backbone):
            if name == 'conv2d':
                assert parameters[parameters_idx].shape == torch.ones(*param[:4]).shape
                if parameters and parameters_idx < len(parameters):
                    w = nn.Parameter(parameters[parameters_idx])
                    b = nn.Parameter(parameters[parameters_idx+1])
                else:
                    # [ch_out, ch_in, kernelsz, kernelsz]
                    w = nn.Parameter(torch.ones(*param[:4]))
                    # gain=1 according to cbfin's implementation
                    torch.nn.init.kaiming_normal_(w)
                    b = nn.Parameter(torch.zeros(param[0]))
                self.vars.append(w)
                # [ch_out] bias
                self.vars.append(b)

            elif name == 'convt2d':
                if parameters and parameters_idx < len(parameters):
                    w = nn.Parameter(parameters[parameters_idx])
                    b = nn.Parameter(parameters[parameters_idx+1])
                else:
                    # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                    w = nn.Parameter(torch.ones(*param[:4]))
                    # gain=1 according to cbfin's implementation
                    torch.nn.init.kaiming_normal_(w)
                    b = nn.Parameter(torch.zeros(param[1]))
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(b)

            elif name == 'linear':
                if parameters and parameters_idx < len(parameters):
                    w = nn.Parameter(parameters[parameters_idx])
                    b = nn.Parameter(parameters[parameters_idx+1])
                else:
                    # [ch_out, ch_in]
                    w = nn.Parameter(torch.ones(*param))
                    # gain=1 according to cbfinn's implementation
                    torch.nn.init.kaiming_normal_(w)
                    b = nn.Parameter(torch.zeros(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(b)

            elif name == 'bn':
                if parameters and parameters_idx < len(parameters):
                    w = nn.Parameter(parameters[parameters_idx])
                    b = nn.Parameter(parameters[parameters_idx+1])
                else:
                    # [ch_out]
                    w = nn.Parameter(torch.ones(param[0]))
                    b = nn.Parameter(torch.zeros(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(b)

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                # parameters_idx -= 2
                continue
            else:
                raise NotImplementedError
            parameters_idx += 2
        if heads:
            for head in self.heads:
                for (name, param) in head:
                    if name == 'conv2d':
                        # [ch_out, ch_in, kernelsz, kernelsz]
                        w = nn.Parameter(torch.ones(*param[:4]))
                        # gain=1 according to cbfin's implementation
                        torch.nn.init.kaiming_normal_(w)
                        b = nn.Parameter(torch.zeros(param[0]))
                        self.vars.append(w)
                        # [ch_out] bias
                        self.vars.append(b)

                    elif name == 'convt2d':
                        # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                        w = nn.Parameter(torch.ones(*param[:4]))
                        # gain=1 according to cbfin's implementation
                        torch.nn.init.kaiming_normal_(w)
                        b = nn.Parameter(torch.zeros(param[1]))
                        self.vars.append(w)
                        # [ch_in, ch_out]
                        self.vars.append(b)

                    elif name == 'linear':
                        # [ch_out, ch_in]
                        w = nn.Parameter(torch.ones(*param))
                        # gain=1 according to cbfinn's implementation
                        torch.nn.init.kaiming_normal_(w)
                        b = nn.Parameter(torch.zeros(param[0]))
                        self.vars.append(w)
                        # [ch_out]
                        self.vars.append(b)

                    elif name == 'bn':
                        # [ch_out]
                        w = nn.Parameter(torch.ones(param[0]))
                        b = nn.Parameter(torch.zeros(param[0]))
                        self.vars.append(w)
                        # [ch_out]
                        self.vars.append(b)

                        # must set requires_grad=False
                        running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                        running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                        self.vars_bn.extend([running_mean, running_var])


                    elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                                'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                        continue
                    else:
                        raise NotImplementedError





    def extra_repr(self):
        info = ''

        for name, param in self.backbone:
            if name == 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name == 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name == 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name == 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info



    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.backbone:
            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5], dilation=param[6])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name == 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name == 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name == 'tanh':
                x = F.tanh(x)
            elif name == 'sigmoid':
                x = torch.sigmoid(x)
            elif name == 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name == 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError
            # print("x_", name, ": ", x[0, 0, 0, 0])
        if not self.heads: 
            return x
        outputs = []
        for i, head in enumerate(self.heads):
            outputs.append(x)
            for (name, param) in head:
                if name == 'conv2d':
                    w, b = vars[idx], vars[idx + 1]
                    # remember to keep synchrozied of forward_encoder and forward_decoder!
                    outputs[i] = F.conv2d(outputs[i], w, b, stride=param[4], padding=param[5], dilation=param[6])
                    idx += 2
                    # print(name, param, '\tout:', x.shape)
                elif name == 'convt2d':
                    w, b = vars[idx], vars[idx + 1]
                    # remember to keep synchrozied of forward_encoder and forward_decoder!
                    outputs[i] = F.conv_transpose2d(outputs[i], w, b, stride=param[4], padding=param[5])
                    idx += 2
                    # print(name, param, '\tout:', x.shape)
                elif name == 'linear':
                    w, b = vars[idx], vars[idx + 1]
                    outputs[i] = F.linear(outputs[i], w, b)
                    idx += 2
                    # print('forward:', idx, x.norm().item())
                elif name == 'bn':
                    w, b = vars[idx], vars[idx + 1]
                    running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                    outputs[i] = F.batch_norm(outputs[i], running_mean, running_var, weight=w, bias=b, training=bn_training)
                    idx += 2
                    bn_idx += 2

                elif name == 'flatten':
                    # print(x.shape)
                    outputs[i] = x.view(outputs[i].size(0), -1)
                elif name == 'reshape':
                    # [b, 8] => [b, 2, 2, 2]
                    outputs[i] = x.view(outputs[i].size(0), *param)
                elif name == 'relu':
                    outputs[i] = F.relu(outputs[i], inplace=param[0])
                elif name == 'leakyrelu':
                    outputs[i] = F.leaky_relu(outputs[i], negative_slope=param[0], inplace=param[1])
                elif name == 'tanh':
                    outputs[i] = F.tanh(outputs[i])
                elif name == 'sigmoid':
                    outputs[i] = torch.sigmoid(outputs[i])
                elif name == 'upsample':
                    outputs[i] = F.upsample_nearest(outputs[i], scale_factor=param[0])
                elif name == 'max_pool2d':
                    outputs[i] = F.max_pool2d(outputs[i], param[0], param[1], param[2])
                elif name == 'avg_pool2d':
                    outputs[i] = F.avg_pool2d(outputs[i], param[0], param[1], param[2])

                else:
                    raise NotImplementedError
        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)


        return outputs


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars