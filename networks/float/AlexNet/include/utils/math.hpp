#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>


namespace math
{


template <typename T>
const double max(const T & src)
{
    return *max_element(src.data.begin(), src.data.end());
}

template <typename T>
const double min(const T & src)
{
    return *min_element(src.data.begin(), src.data.end());
}

template <typename T>
const double sum(const T & src)
{
    return std::accumulate(src.data.begin(), src.data.end(), 0.0);
}

template <typename T>
const double mean(const T & src)
{
    return std::accumulate(src.data.begin(), src.data.end(), 0.0) / T::SIZE;
}

template <typename T>
const T square(const T & src)
{
    T result;
    std::transform(
        src.data.begin(), src.data.end(), result.data.begin(),
        [](const double v) { return v * v; }
    );
    return result;
}

template <typename T>
const T exp(const T & src)
{
    T result;
    std::transform(
        src.data.begin(), src.data.end(), result.data.begin(),
        [](const double v) { return std::exp(v); }
    );
    return result;
}

template <typename T>
const T pow(const T & src, const double exp)
{
    T result;
    std::transform(
        src.data.begin(), src.data.end(), result.data.begin(),
        //[&exp](const double v) { return std::pow(v, exp); }
        std::bind2d(std::pow(), exp)
    );
    return result;
}

template <typename T>
const T sqrt(const T & src)
{
    T result;
    std::transform(
        src.data.begin(), src.data.end(), result.data.begin(),
        std::sqrt()
    );
    return result;
}


}