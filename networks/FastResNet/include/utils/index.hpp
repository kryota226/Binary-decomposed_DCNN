#pragma once
#include <cassert>

/**
    std::vector<double> ‚ğƒeƒ“ƒ\ƒ‹‚Æ‚µ‚Äˆµ‚¤‚Æ‚«‚Ì”z—ñ“Yš‚ÌŒvZ

    :param dim0N: ÅãˆÊŠK‚ÌŸŒ³”
    :param dim1N: ã‚©‚ç2”Ô–Ú‚ÌŠKˆÊ‚ÌŸŒ³”
    :param dim2N: ã‚©‚ç3”Ô–Ú‚ÌŠKˆÊ‚ÌŸŒ³”
    :param dim3N: Å‰ºˆÊŠK‚ÌŸŒ³”B[i] ‚Æ [i+1] ‚Íƒƒ‚ƒŠã‚Å‚à—×B
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
