#pragma once
#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>
#include "index.hpp"


template <typename T, int d1, int d2=1, int d3=1, int d4=1>
class Tensor
{
public:
    enum {
        DIM_1 = d1,
        DIM_2 = d2,
        DIM_3 = d3,
        DIM_4 = d4,
        SIZE = DIM_1 * DIM_2 * DIM_3 * DIM_4,
    };
    static_assert(0 < DIM_1, "");
    static_assert(0 < DIM_2, "");
    static_assert(0 < DIM_3, "");
    static_assert(0 < DIM_4, "");

    typedef Index<DIM_1, DIM_2, DIM_3, DIM_4> IndexT;
    typedef Tensor<T, DIM_1, DIM_2, DIM_3, DIM_4> SameT;

    std::vector<T> data;

    Tensor(const T v=0)
        : data(SIZE, v)
    {
        assert(data.size() == SIZE);
    }

    Tensor(const std::vector<T> & v)
        : data(v)
    {
        assert(data.size() == SIZE);
    }

    // ** 4�K�e���\��(rank 4 tensor)�̊e�v�f�ւ̃A�N�Z�X(read/write) * //
    T & at(const int i1, const int i2=0, const int i3=0, const int i4=0)
    {
        return data[IndexT::indexOf(i1, i2, i3, i4)];
    }

    // ** 4�K�e���\��(rank 4 tensor)�̊e�v�f�ւ̃A�N�Z�X(READ ONLY) * //
    const T & at(const int i1, const int i2=0, const int i3=0, const int i4=0) const
    {
        return data[IndexT::indexOf(i1, i2, i3, i4)];
    }

    /** �e���\�� -> �e���\���ւ̉��炩�̕ϊ��E�v�Z */
    template<typename OpT>
    typename OpT::OutT operator >> (OpT & next_layer)
    {
        return next_layer.forward(*this);
    }


    /*
    �I�y���[�^
    --------------------------------------------------
    ��{�I�ȉ��Z�q�͂��ׂĂ̗v�f�ɍ�p����B
    */
    const SameT operator + (const T value) const
    {
        std::vector<T> dst(SIZE, value);
        std::transform(
            data.begin(),
            data.end(),
            dst.begin(),
            dst.begin(),
            std::plus<T>()
        );
        return SameT(dst);
    }

    const SameT operator - (const T value) const
    {
        std::vector<T> dst(SIZE, value);
        std::transform(
            data.begin(),
            data.end(),
            dst.begin(),
            dst.begin(),
            std::minus<T>()
        );
        return SameT(dst);
    }

    const SameT operator * (const T value) const
    {
        std::vector<T> dst(SIZE, value);
        std::transform(
            data.begin(),
            data.end(),
            dst.begin(),
            dst.begin(),
            std::multiplies<T>()
        );
        return SameT(dst);
    }

    const SameT operator / (const T value) const
    {
        std::vector<T> dst(SIZE, value);
        std::transform(
            data.begin(),
            data.end(),
            dst.begin(),
            dst.begin(),
            std::divides<T>()
        );
        return SameT(dst);
    }

    const SameT operator + (const SameT & v) const
    {
        std::vector<T> dst = v.data;
        std::transform(
            data.begin(), data.end(), dst.begin(), dst.begin(),
            std::plus<T>()
        );
        return SameT(dst);
    }

    const SameT operator - (const SameT & v) const
    {
        std::vector<T> dst = v.data;
        std::transform(
            data.begin(), data.end(), dst.begin(), dst.begin(),
            std::minus<T>()
        );
        return SameT(dst);
    }

    const SameT operator * (const SameT & v) const
    {
        std::vector<T> dst = v.data;
        std::transform(
            data.begin(), data.end(), dst.begin(), dst.begin(),
            std::multiplies<T>()
        );
        return SameT(dst);
    }

    const SameT operator / (const SameT & v) const
    {
        std::vector<T> dst = v.data;
        std::transform(
            data.begin(), data.end(), dst.begin(), dst.begin(),
            std::divides<T>()
        );
        return SameT(dst);
    }
};