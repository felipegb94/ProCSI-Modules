/* Pi - CUDA version
 * Author: Aaron Weeden, Shodor, May 2015
 *
 * Approximate pi using a Left Riemann Sum under a quarter unit circle.
 *
 * When running the program, the number of rectangles can be passed using the
 * -r option, e.g. 'pi-cuda-1 -r X', where X is the number of rectangles.
 */

/*************
 * LIBRARIES *
 *************/

/* Pi - CUDA version 1 - uses integers for CUDA kernels
 * Author: Aaron Weeden, Shodor, May 2015
 */
#include <stdio.h> /* fprintf() */
#include <iostream> 

#include "pi-cuda-1.cuh"

/************************
 * FUNCTION DEFINITIONS *
 ************************/
int main(int argc, char **argv) 
{
    int numRects = 1e10;
    double area = 0.0;

    if(CUDA_ENABLED){
        printf("CUDA is enabled\n");
    }
    else{
        printf("Cuda is not enabled\n");
    }
    // {
    //     std::cout << "WORKING!" <<std::endl;
    // }
    calculateArea(numRects, &area);
    std::cout << "Pi = " << 4*area << std::endl;

  return 0;
}




