#pragma once


/*
�萔�^�؂�グ�v�Z�@�B
L �������鐔�AR�����鐔�B
*/
template <long L, long R>
struct iceil
{
    static const long value = (L + (R - 1)) / R;
};