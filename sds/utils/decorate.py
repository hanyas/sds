from functools import wraps

import numpy as np
import torch


def to_float(arr, device=torch.device('cpu')):
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr).float().to(device)
    elif isinstance(arr, torch.FloatTensor):
        return arr.to(device)
    else:
        raise TypeError


def np_float(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().double().cpu().numpy()
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        raise TypeError


def ensure_args_are_viable_lists(f):
    def wrapper(self, obs, act=None, **kwargs):
        assert obs is not None
        obs = [np.atleast_2d(obs)] if not isinstance(obs, (list, tuple)) else obs

        if act is None:
            act = []
            for _obs in obs:
                act.append(np.zeros((_obs.shape[0], self.act_dim)))

        act = [np.atleast_2d(act)] if not isinstance(act, (list, tuple)) else act

        return f(self, obs, act, **kwargs)
    return wrapper


def init_empty_logctl_to_zero(f):
    def wrapper(self, *args,  logctl=None, **kwargs):
        if f.__name__ == 'forward' or f.__name__ == 'backward':
            if len(args) == 3:
                logobs = args[-1]
                logctl = [np.zeros_like(_logobs) for _logobs in logobs]
                return f(self, *args, logctl, **kwargs)
            else:
                return f(self, *args, **kwargs)
        if f.__name__ == 'joint_posterior':
            if len(args) == 5:
                logobs = args[-1]
                logctl = [np.zeros_like(_logobs) for _logobs in logobs]
                return f(self, *args, logctl, **kwargs)
            else:
                return f(self, *args, **kwargs)
        else:
            raise NotImplementedError
    return wrapper


def ensure_args_torch_floats(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        wrapped_args = []
        for arg in args:
            if isinstance(arg, list):
                wrapped_args.append([to_float(arr, self.device) for arr in arg])
            elif isinstance(arg, np.ndarray):
                wrapped_args.append(to_float(arg, self.device))
            else:
                wrapped_args.append(arg)

        return f(self, *wrapped_args, **kwargs)
    return wrapper


def ensure_return_numpy_floats(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        func_outputs = f(self, *args, **kwargs)
        if isinstance(func_outputs, list):
            wrapper_outputs = []
            for fout in func_outputs:
                if isinstance(fout, torch.Tensor):
                    wrapper_outputs.append(np_float(fout))
                elif isinstance(fout, list):
                    wrapper_outputs.append([np_float(arr) for arr in fout])
        elif isinstance(func_outputs, torch.Tensor):
            wrapper_outputs = np_float(func_outputs)
        else:
            raise NotImplementedError

        return wrapper_outputs
    return wrapper


def parse_init_values(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        wrapped_args = []
        for arg in args:
            if arg is None:
                wrapped_args.append(arg)
            else:
                wrapped_args.append([np.atleast_2d(arr[:self.nb_lags]) for arr in arg])

        return f(self, *wrapped_args, **kwargs)
    return wrapper


def ensure_ar_stack(f):
    @wraps(f)
    def wrapper(self, z, x, *args):
        xr = np.reshape(x, (-1, self.obs_dim * self.nb_lags))
        return f(self, z, xr, *args)
    return wrapper
