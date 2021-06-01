from functools import wraps

import numpy as np
import torch


def ensure_args_are_viable(f):
    @wraps(f)
    def wrapper(self, obs, act=None, **kwargs):
        assert obs is not None

        if isinstance(obs, list):
            obs = [np.atleast_2d(_obs) for _obs in obs]
            if act is None:
                act = [np.zeros((_obs.shape[0], self.act_dim)) for _obs in obs]
            else:
                act = [np.atleast_2d(_act) for _act in act]
        elif isinstance(obs, np.ndarray):
            obs = np.atleast_2d(obs)
            if act is None:
                act = np.zeros((obs.shape[0], self.act_dim))
            else:
                act = np.atleast_2d(act)
        else:
            raise NotImplementedError

        return f(self, obs, act, **kwargs)
    return wrapper


def init_empty_logact_to_zero(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if f.__name__ == 'forward' or f.__name__ == 'backward':
            if len(args) == 3:
                loginit, logtrans, logobs = args
                logact = np.zeros_like(logobs) if isinstance(logobs, np.ndarray) \
                         else [np.zeros_like(_logobs) for _logobs in logobs]
                return f(self, loginit, logtrans, logobs, logact, **kwargs)
            else:
                return f(self, *args, **kwargs)
        if f.__name__ == 'joint_posterior':
            if len(args) == 5:
                alpha, beta, loginit, logtrans, logobs = args
                logact = np.zeros_like(logobs) if isinstance(logobs, np.ndarray) \
                         else [np.zeros_like(_logobs) for _logobs in logobs]
                return f(self, alpha, beta, loginit, logtrans, logobs, logact, **kwargs)
            else:
                return f(self, *args, **kwargs)
        else:
            raise NotImplementedError
    return wrapper


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
