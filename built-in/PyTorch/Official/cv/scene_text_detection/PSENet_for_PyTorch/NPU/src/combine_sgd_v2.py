import torch
from change_data_ptr import change_data_ptr


def combine_tensor(list_of_tensor, copy_back=True):
    total_numel = 0
    for tensor in list_of_tensor:
        total_numel += tensor.storage().size()
    combine_tensor = torch.zeros(total_numel).npu().to(list_of_tensor[0].dtype)

    idx = 0

    if copy_back:
        for tensor in list_of_tensor:
            temp = tensor.clone()
            temp.copy_(tensor)
            change_data_ptr(tensor, combine_tensor, idx)
            temp_data = tensor.data
            temp_data.copy_(temp)
            idx += temp.storage().size()
    else:
        for tensor in list_of_tensor:
            change_data_ptr(tensor, combine_tensor, idx)
            idx += temp.storage().size()
    return combine_tensor

def recombine_tensor(size, combined_tensor, index=0):
    temp_grad = torch.zeros(size).npu().to(combined_tensor.dtype)
    change_data_ptr(temp_grad, combined_tensor, index)
    return temp_grad


class CombinedSGD(torch.optim.SGD):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, combine_grad=True):
        self.combined = combine_grad
        self.init_combine = False
        self.first_init = True
        self.opt_level_O2_has_bn = False
        self.combined_grad = []
        self.combined_weight = []
        self.combined_momentum = []
        super(CombinedSGD, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def split_combined_tensors(self, input_combined_grad_1, input_combined_grad_2=None):
        if len(self.combined_weight) > 0:
            for tensor in self.combined_weight:
                tensor = None
            self.first_init = False
            self.combined_grad = []
            self.combined_weight = []
            self.combined_momentum = []

        index_ops, index_bn = 0, 0
        for param_group in self.param_groups:
            size_ops, size_bn = 0, 0
            ord_param_list = []
            spe_param_list = []
            for param in param_group["params"]:
                if param.requires_grad and param.grad is not None:
                    temp_size = param.grad.storage().size()
                    if input_combined_grad_1.data_ptr() <= param.grad.data_ptr() < input_combined_grad_1.data_ptr() + input_combined_grad_1.numel() * input_combined_grad_1.element_size():
                        size_ops += temp_size
                        ord_param_list.append(param)
                    else:
                        size_bn += temp_size
                        spe_param_list.append(param)
            self.combined_grad.append(recombine_tensor(size_ops, input_combined_grad_1, index_ops))
            self.combined_weight.append(combine_tensor(ord_param_list, copy_back=True))

            if self.first_init:
                self,combined_momentum.append(tensor.zeros_like(self.combined_grad[-1]))
            else:
                self,combined_momentum.append(self.combined_grad[-1].clone())

            index_ops += size_ops

            if input_combined_grad_2 is not None:
                self.combined_grad.append(recombine_tensor(size_bn, input_combined_grad_2, index_bn))
                self.combined_weight.append(combine_tensor(spe_param_list, copy_back=True))

                if self.first_init:
                    self,combined_momentum.append(tensor.zeros_like(self.combined_grad[-1]))
                else:
                    self,combined_momentum.append(self.combined_grad[-1].clone())
                index_bn += size_bn

    def _init_combined(self):
        if not self.init_combine:
            if hasattr(self, "_amp_stash"):
                stash = self._amp_stash
                if hasattr(stash, "all_fp32_params"):
                    if len(stash.grads_combined_list) == 0:
                        raise RuntimeError("When use CombinedSGD, Apex O1 need to use combine_grad=True module!")
                    self.split_combined_tensors(stash.grads_combined_list[-1])
                    self.init_combine = True
                elif hasattr(stash, "all_fp32_from_fp16_params"):
                    if len(stash.grads_combined_list) == 0:
                        raise RuntimeError("When use CombinedSGD, Apex O2 need to usecombine_grad=True module!")
                    if stash.grads_combined_list[1] is not None:
                        if stash.grads_combined_list[2] is None:
                            self.split_combined_tensors(stash.grads_combined_list[1])
                        else:
                            self.split_combined_tensors(stash.grads_combined_list[1], self.grads_combined_list[2])
                            self.opt_level_O2_has_bn = True
                    else:
                        raise RuntimeError("Inapproperiate network which only have batchnorm layers!")
                    self.init_combine = True
            else:
                for param_group in self.param_groups:
                    lst_grad = []
                    lst_weight = []
                    for param in param_group["params"]:
                        if param.requires_grad and param.grad is not None:
                            lst_grad.append(param.grad)
                            lst_weight.append(param)
                    if len(lst_grad) > 0:
                        self.combine_grad.append(combine_tensor(lst_grad, True))
                        self.combine_weight.append(combine_tensor(lst_weight, True))
                        self.combined_momentum.append(torch.zeros_like(self.combine_grad[-1]))
                        self.init_combine = True

    def step_combined(self, weight_decay, momentum, dampening, nesterov, lr, combined_grad, combined_weight, combined_momentum):
        if weight_decay != 0:
            combined_grad.add_(combined_weight, alpha=weight_decay)
        if momentum != 0:
            combined_momentum.mul_(momentum).add_(combined_grad, alpha=1 - dampening)
            if nesterov:
                combined_grad.add_(combined_momentum, alpha=momentum)
            else:
                combined_grad.copy_(combined_momentum)
        combined_weight.add_(combined_grad, alpha=lr)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        idx = 0
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            if self.combined:
                self._init_combined()
                self.step_combined(weight_decay, momentum, dampening, nesterov, -group["lr"],
                    self.combined_grad[idx], self.combined_weight[idx], self.combined_momentum[idx])
                if self.opt_level_O2_has_bn:
                    idx += 1
                    self.step_combined(weight_decay, momentum, dampening, nesterov, -group["lr"],
                        self.combined_grad[idx], self.combined_weight[idx], self.combined_momentum[idx])
            else:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad
                    if weight_decay != 0:
                        d_p = d_p.add(p, alpha=weight_decay)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf
    
                    p.add_(d_p, alpha=-group['lr'])

        return loss