#include <stdio.h> /* fprintf() */
#include <iostream> 

#include "pi-cuda-unifiedMem.cuh"

/************************
 * FUNCTION DEFINITIONS *
 ************************/
int main(int argc, char **argv) 
{
    long numRects = 2e9;
    double area = 0.0;

    if(CUDA_ENABLED){
        printf("CUDA is enabled\n");
    }
    else if(OPENMP_ENABLED){
        printf("OMP is enabled\n");     
    }
    else{
        printf("Neitheer CUDA or OMP are enabled. Running serially\n");
    }
    // {
    //     std::cout << "WORKING!" <<std::endl;
    // }
    calculateArea(numRects, &area);

    std::cout << "NumRects used = " << numRects << std::endl;
    std::cout << "Pi = " << 4*area << std::endl;

  return 0;
}
