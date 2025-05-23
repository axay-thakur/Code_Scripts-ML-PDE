import numpy as np
import scipy.io
import h5py
import torch
#initial condition generator
# See - Modeling the dynamics of PDE systems with physics-constrained deep auto-regressive networks,Geneva, JCP 
def initcond(order,x_f):
    lam = np.random.randn(2, 2*order+1)
    c = np.random.rand() - 0.5
    k = np.arange(-order, order+1)
    kx = np.outer(k, x_f)*2*np.pi

    # vector field
    f = np.dot(lam[[0]], np.cos(kx)) + np.dot(lam[[1]], np.sin(kx))
    f = 2 * f / np.amax(np.abs(f)) + 2.0*c
    return f

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float