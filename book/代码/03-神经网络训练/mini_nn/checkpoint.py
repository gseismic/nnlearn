"""通过 NumPy NPZ 文件保存和恢复模块参数。"""

import numpy as np


def save_state_dict(module, path):
    np.savez(path, **module.state_dict())


def load_state_dict(module, path):
    with np.load(path, allow_pickle=False) as archive:
        state = {name: archive[name] for name in archive.files}
    return module.load_state_dict(state)
