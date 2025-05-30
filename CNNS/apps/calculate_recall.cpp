// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "diskann_utils.h"

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::cout << argv[0] << " <ground_truth_bin> <our_results_bin>  <r> " << std::endl;
        return -1;
    }
    
    return 0;
}
