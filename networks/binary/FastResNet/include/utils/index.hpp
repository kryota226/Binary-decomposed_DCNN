#pragma once
#include <cassert>

/**
    std::vector<double> をテンソルとして扱うときの配列添字の計算

    :param dim0N: 最上位階の次元数
    :param dim1N: 上から2番目の階位の次元数
    :param dim2N: 上から3番目の階位の次元数
    :param dim3N: 最下位階の次元数。[i] と [i+1] はメモリ上でも隣。
*/
template <int dim0N, int dim1N=1, int dim2N=1, int dim3N=1>
class Index
{
public:
    static int indexOf(const int i0)
    {
        assert(0 <= i0);
        assert(i0 < dim0N);
        return i0;
    }

    static int indexOf(const int i0, const int i1)
    {
        assert(0 <= i0);
        assert(i0 < dim0N);
        assert(0 <= i1);
        assert(i1 < dim1N);
        return i0 * dim1N + i1;
    }

    static int indexOf(const int i0, const int i1, const int i2)
    {
        assert(0 <= i0);
        assert(i0 < dim0N);
        assert(0 <= i1);
        assert(i1 < dim1N);
        assert(0 <= i2);
        assert(i2 < dim2N);
        return (i0 * dim1N + i1) * dim2N + i2;
    }

    static int indexOf(const int i0, const int i1, const int i2, const int i3)
    {
        assert(0 <= i0);
        assert(i0 < dim0N);
        assert(0 <= i1);
        assert(i1 < dim1N);
        assert(0 <= i2);
        assert(i2 < dim2N);
        assert(0 <= i3);
        assert(i3 < dim3N);
        return ((i0 * dim1N + i1) * dim2N + i2) * dim3N + i3;
    }
};
