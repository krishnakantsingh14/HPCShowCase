#include <unistd.h> // for usleep()
#include <iostream>
#include <ctime>

int main()
{
    struct timespec tstart, tstop, tresult;
    clock_gettime(CLOCK_MONOTONIC, &tstart);
    sleep(10);
    clock_gettime(CLOCK_MONOTONIC, &tstop);
    
    tresult.tv_sec = tstop.tv_sec - tstart.tv_sec;
    tresult.tv_nsec = tstop.tv_nsec - tstart.tv_nsec;
    
    std::cout << "Time taken: " << tresult.tv_sec << " seconds and " << tresult.tv_nsec << " nanoseconds" << std::endl;

    return 0;
}