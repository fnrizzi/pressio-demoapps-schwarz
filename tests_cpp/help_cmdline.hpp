
#ifndef HELP_PARSE_HPP_
#define HELP_PARSE_HPP_

#include <string>

int parse_num_threads(int argc, char *argv[])
{
    /*
        1. no args passed => 1 thread is used
        2. ./exe n        => n threads are used if n!=1
        3. ./exe -1       => max hardware concurrency
    */

    int numthreads = 1;
    if (argc >= 2){
        numthreads = std::stoi(argv[1]);
        if (numthreads == -1){ numthreads = std::thread::hardware_concurrency(); }
        std::cout << "Running with " << numthreads << " threads\n";
        assert(numthreads >= 1);
    }
    else {
        std::cout << "defaulting to 1 thread because not cmdline arg was found\n";
    }

    return numthreads;
}

#endif
