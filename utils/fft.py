import numpy


def _DFT_cst(N, fft_length, trunc=True):
    n = numpy.arange(N)
    k = n.reshape((N, 1)).astype(numpy.float64)
    M = numpy.exp(-2j * numpy.pi * k * n / fft_length)
    return M[:fft_length // 2 + 1] if trunc else M

def DFT(x, fft_length=None, axis=1):
    if axis == 1:
        x = x.T
    if fft_length is None:
        fft_length = x.shape[0]
    cst = _DFT_cst(x.shape[0], fft_length, trunc=axis==1)
    if axis == 1:
        return numpy.matmul(cst, x).T
    return numpy.matmul(cst, x)

def fft2d_(mat, fft_length):
    mat = mat[:fft_length[0], :fft_length[1]]
    res = mat.copy()
    res = DFT(res, fft_length[1], axis=1)
    res = DFT(res, fft_length[0], axis=0)
    return res[:fft_length[0], :fft_length[1]//2 + 1]