from .optimizer import Optimizer
from nnlearn import backend


class Adam(Optimizer):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        super().__init__(
            parameters,
            defaults={
                'lr': lr,
                'betas': betas,
                'eps': eps,
                'weight_decay': weight_decay,
            },
        )
        self._refresh_attributes_from_first_group()
        self.t = 0
        self.m = {}
        self.v = {}
        # 每个参数只按自身实际更新次数做偏差修正；t 保留全局调用次数供旧状态兼容。
        self.steps = {}

    def step(self):
        self.t += 1
        super().step()

    def _refresh_attributes_from_first_group(self):
        group = self.param_groups[0]
        self.lr = group.get('lr', getattr(self, 'lr', 1e-3))
        self.beta1, self.beta2 = group.get(
            'betas', (getattr(self, 'beta1', 0.9), getattr(self, 'beta2', 0.999))
        )
        self.eps = group.get('eps', getattr(self, 'eps', 1e-8))
        self.weight_decay = group.get(
            'weight_decay', getattr(self, 'weight_decay', 0.0)
        )

    def update_one(self, parameter, group):
        key = id(parameter)
        xp = backend.get_array_module(parameter.data)
        if key not in self.m:
            self.m[key] = xp.zeros_like(parameter.data)
            self.v[key] = xp.zeros_like(parameter.data)
            self.steps[key] = 0

        lr = self._scalar_like(group['lr'], parameter)
        beta1, beta2 = group['betas']
        eps = self._scalar_like(group['eps'], parameter)
        weight_decay = group['weight_decay']
        grad = parameter.grad.data
        if weight_decay:
            grad = grad + self._scalar_like(weight_decay, parameter) * parameter.data

        self.m[key] = (
            self._scalar_like(beta1, parameter) * self.m[key]
            + self._scalar_like(1 - beta1, parameter) * grad
        )
        self.v[key] = (
            self._scalar_like(beta2, parameter) * self.v[key]
            + self._scalar_like(1 - beta2, parameter) * (grad * grad)
        )
        self.steps[key] += 1

        m_hat = self.m[key] / self._scalar_like(1 - beta1 ** self.steps[key], parameter)
        v_hat = self.v[key] / self._scalar_like(1 - beta2 ** self.steps[key], parameter)
        parameter.data = parameter.data - lr * m_hat / (xp.sqrt(v_hat) + eps)

    def state_dict(self):
        indices = self._param_indices()
        state = {}
        for key, m_value in self.m.items():
            if key in indices:
                state[indices[key]] = {
                    'm': self._state_tensor(m_value),
                    'v': self._state_tensor(self.v[key]),
                    'step': self.steps[key],
                }
        return {
            'state': state,
            'param_groups': self._state_param_groups(),
            't': self.t,
        }

    def load_state_dict(self, state_dict):
        self._load_param_groups(state_dict)
        self.t = state_dict.get('t', self.t)
        self.m = {}
        self.v = {}
        self.steps = {}
        for index, values in state_dict.get('state', {}).items():
            parameter = self.parameters[int(index)]
            key = id(parameter)
            self.m[key] = self._state_array(values['m'], parameter)
            self.v[key] = self._state_array(values['v'], parameter)
            self.steps[key] = int(values.get('step', self.t))


class AdamW(Adam):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2):
        super().__init__(parameters, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay)
        self._refresh_attributes_from_first_group()

    def _refresh_attributes_from_first_group(self):
        super()._refresh_attributes_from_first_group()
        self.decoupled_weight_decay = self.weight_decay
        self.weight_decay = 0.0

    def update_one(self, parameter, group):
        weight_decay = group.get('weight_decay', self.decoupled_weight_decay)
        if weight_decay:
            factor = self._scalar_like(1 - group['lr'] * weight_decay, parameter)
            parameter.data = parameter.data * factor
        adam_group = dict(group)
        adam_group['weight_decay'] = 0.0
        super().update_one(parameter, adam_group)
