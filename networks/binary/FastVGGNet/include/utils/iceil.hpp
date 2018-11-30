#pragma once


/*
定数型切り上げ計算機。
L が割られる数、Rが割る数。
*/
template <long L, long R>
struct iceil
{
    static const long value = (L + (R - 1)) / R;
};