#include <cassert>
#include "utils/timer.hpp"


void Timer::start(void)
{
    start_time = static_cast<double>(cv::getTickCount());
}

void Timer::stop(void)
{
    stop_time = static_cast<double>(cv::getTickCount());
}

double Timer::time(void)
{
    assert(start_time < stop_time);
    return (stop_time - start_time) * 1000 / cv::getTickFrequency();
}