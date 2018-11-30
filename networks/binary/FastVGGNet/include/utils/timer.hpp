#pragma once
#include "opencv2/opencv.hpp"


class Timer
{
public:
    /*
    時間計測開始
    */
    void start(void);

    /*
    時間計測終了
    */
    void stop(void);

    /*
    計測結果を返す。 単位はミリ秒。
    */
    double time(void);


private:
    double start_time;
    double stop_time;
};