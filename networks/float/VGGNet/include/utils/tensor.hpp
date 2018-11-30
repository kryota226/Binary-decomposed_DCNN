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
    typedef Tensor<T, DIM_1, DIM_2, DIM_3, DIM_4> ValueT;

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

    ~Tensor(void) {}

    // ** 4階テンソル(rank 4 tensor)の各要素へのアクセス(read/write) * //
    T & at(const int i1, const int i2=0, const int i3=0, const int i4=0)
    {
        return data[IndexT::indexOf(i1, i2, i3, i4)];
    }

    // ** 4階テンソル(rank 4 tensor)の各要素へのアクセス(READ ONLY) * //
    const T & at(const int i1, const int i2=0, const int i3=0, const int i4=0) const
    {
        return data[IndexT::indexOf(i1, i2, i3, i4)];
    }

    /** テンソル -> テンソルへの何らかの変換・計算 */
    template<typename OpT>
    typename OpT::OutT operator >> (OpT & next_layer)
    {
        return next_layer.forward(*this);
    }


    /*
    オペレータ
    --------------------------------------------------
    基本的な演算子はすべての要素に作用する。
    */
    const ValueT operator + (const T value) const
    {
        std::vector<T> dst(SIZE, value);
        std::transform(
            data.begin(),
            data.end(),
            dst.begin(),
            dst.begin(),
            std::plus<T>()
        );
        return ValueT(dst);
    }

    void operator += (const T value)
    {
        std::vector<T> dst(SIZE, value);
        std::transform(
            data.begin(),
            data.end(),
            data.begin(),
            std::plus<T>()
        );
    }

    const ValueT operator - (const T value) const
    {
        std::vector<T> dst(SIZE, value);
        std::transform(
            data.begin(),
            data.end(),
            dst.begin(),
            dst.begin(),
            std::minus<T>()
        );
        return ValueT(dst);
    }

    const ValueT operator * (const T value) const
    {
        std::vector<T> dst(SIZE, value);
        std::transform(
            data.begin(),
            data.end(),
            dst.begin(),
            dst.begin(),
            std::multiplies<T>()
        );
        return ValueT(dst);
    }

    void operator *= (const T value)
    {
        std::vector<T> v_vector(SIZE, value);
        std::transform(
            data.begin(),
            data.end(),
            v_vector.begin(),
            data.begin(),
            std::multiplies<T>()
        );
    }

    const ValueT operator / (const T value) const
    {
        std::vector<T> dst(SIZE, value);
        std::transform(
            data.begin(),
            data.end(),
            dst.begin(),
            dst.begin(),
            std::divides<T>()
        );
        return ValueT(dst);
    }


    const T max(void) const
    {
        return *max_element(data.begin(), data.end());
    }

    const T min(void) const
    {
        return *min_element(data.begin(), data.end());
    }

    const T sum(void) const
    {
        return std::accumulate(data.begin(), data.end(), 0.0);
    }

    const T mean(void) const
    {
        return std::accumulate(data.begin(), data.end(), 0.0) / SIZE;
    }

    const ValueT square(void) const
    {
        std::vector<T> dst(SIZE);
        std::transform(
            data.begin(),
            data.end(),
            dst.begin(),
            [](const T v) {
                return v * v;
            }
        );
        return ValueT(dst);
    }
};