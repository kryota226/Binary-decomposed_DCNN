#pragma once
#include "opencv2/opencv.hpp"


class Timer
{
public:
    /*
    ���Ԍv���J�n
    */
    void start(void);

    /*
    ���Ԍv���I��
    */
    void stop(void);

    /*
    �v�����ʂ�Ԃ��B �P�ʂ̓~���b�B
    */
    double time(void);


private:
    double start_time;
    double stop_time;
};