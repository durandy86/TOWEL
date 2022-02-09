import numpy as np
from itertools import chain

def splitter2D(img, nblocks, dims):
    """
    nblock : tuple number of block along dimensions 2 and 3.
    """
    split1 = np.array_split(img, nblocks[0], axis=dims[0])
    split2 = [np.array_split(im, nblocks[1], axis=dims[1]) for im in split1]
    olist = [np.copy(a) for a in list(chain.from_iterable(split2))]
    return olist


def stitcher2D(outputs, nblocks,dims):
    stitched = []
    if np.array(nblocks).size == 1:
        nblocks = np.array([nblocks, nblocks])
    for i in range(nblocks[0]):
        stitched.append(outputs[i * nblocks[1]])
        for j in range(1, nblocks[1]):
            outind = j + i * nblocks[1]
            stitched[i] = np.concatenate((stitched[i], outputs[outind]), axis=dims[1])
    return np.concatenate(tuple(stitched), axis=dims[0])



def splitter1D(img, nblock, dim):
    """
    nblock : tuple number of block along dimensions 2 and 3.
    """
    s = list(np.shape(img))

    split1 = np.array_split(img, nblock, axis=dim)
    if nblock==s[0]:
        s[dim]=1
        olist = [np.copy(np.reshape(a,tuple(s))) for a in list(chain.from_iterable(split1))]
    else:
        olist = [np.copy(a) for a in list(chain.from_iterable(split1))]

    return olist


def stitcher1D(outputs,dim):
    return np.concatenate(outputs, axis=dim)







