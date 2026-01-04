import numpy as np 
import h5py, threading
import queue as Queue
import torch

# bkgdGen 类（后台数据生成器）
class bkgdGen(threading.Thread):
    def __init__(self, data_generator, max_prefetch=1):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = data_generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item, block=True, timeout=None)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get(block=True, timeout=None)
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

def gen_train_batch_bg(mb_size, in_depth, dev='cuda', trans=True, cvars=[]):
    train_mask = np.arange(0, 365, dtype=np.uint) <  292

    rain_12km = h5py.File(r'E:\文章2\实验\2013\NDVI\WRF_precip_2005_12km-mask-clip0p05-99p5.hdf5', "r")['SD_adjusted'][:][train_mask]

    vars_all = {}
    with h5py.File(r'E:\文章2\实验\2013\NDVI\WRF_50km_vars-mask-clip0p05-99p5.hdf5', "r") as h5fd:
        vars_all['prec'] = h5fd['SD_adjusted'][:][train_mask] # precipitation as always
        for _var in cvars:
            vars_all[_var] = h5fd[_var][:][train_mask]

    if trans:
        rain_12km = np.log(1 + rain_12km)

    while True:
        idx = np.random.randint(0, rain_12km.shape[0] - in_depth, mb_size)
        vars4stack = []
        batch_prec = np.array([vars_all['prec'][s_idx: (s_idx + in_depth)] for s_idx in idx], dtype=np.float32)

        for _var in cvars:
            batch_cvar = np.array([vars_all[_var][s_idx: (s_idx + in_depth)] for s_idx in idx], dtype=np.float32)
            vars4stack.append(batch_cvar)

        batch_Y = np.expand_dims([rain_12km[s_idx + in_depth - 1] for s_idx in idx], 1)

        yield torch.from_numpy(batch_prec).to(dev), [torch.from_numpy(_d).to(dev) for _d in
                                                     vars4stack], torch.from_numpy(batch_Y).to(dev)


def get1batch4test(in_depth, idx=None, dev='cuda', trans=True, cvars=[]):
    test_mask = np.arange(0, 365, dtype=np.uint) >= 292
    test_days = np.where(test_mask)[0]  # 获取测试集原始日期索引

    rain_12km = h5py.File(r'E:\文章2\实验\2013\NDVI\WRF_precip_2005_12km-mask-clip0p05-99p5.hdf5',"r")['SD_adjusted'][:][test_mask]

    vars_all = {}
    with h5py.File(r'E:\文章2\实验\2013\NDVI\WRF_50km_vars-mask-clip0p05-99p5.hdf5',"r") as h5fd:
        vars_all['prec'] = h5fd['SD_adjusted'][:][test_mask]
        for _var in cvars:
            vars_all[_var] = h5fd[_var][:][test_mask]

    if trans:
        rain_12km = np.log(1 + rain_12km)

    if idx is None:
        idx = np.random.randint(0, rain_12km.shape[0] - in_depth + 1, 128)
    # 计算每个样本对应的原始日期
    else:
        idx = np.array(idx)  # 确保idx是NumPy数组

    batch_prec = np.array([vars_all['prec'][s_idx: (s_idx + in_depth)] for s_idx in idx], dtype=np.float32)

    vars4stack = []
    for _var in cvars:
        batch_cvar = np.array([vars_all[_var][s_idx: (s_idx + in_depth)] for s_idx in idx], dtype=np.float32)
        vars4stack.append(batch_cvar)

    batch_Y = np.expand_dims([rain_12km[s_idx + in_depth - 1] for s_idx in idx], 1)
    original_days = test_days[idx + in_depth - 1]

    return (
        torch.from_numpy(batch_prec).to(dev),
        [torch.from_numpy(_d).to(dev) for _d in vars4stack],
        torch.from_numpy(batch_Y).to(dev),
        original_days
    )
