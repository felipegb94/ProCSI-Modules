/******************************************************************************
* FILE: pi-serial-rectangles.c
* 
* DESCRIPTION:
*	Computing Pi Example - Serial
*	This example demonstrates how to compute pi using left Riemann sums.
*
* INSTRUCTIONS:
*	Create a parallel program to compute pi using OpenMP's reduction clause.
*	You do not need to use this version as the basis for your parallel version,
*	but you are welcome to.
*
* AUTHOR: Phil List
* Modified By: Felipe Gutierrez
******************************************************************************/

#include <stdio.h>      /* to use printf() */
#include <math.h>	/* to use sqrt() */

int main (int argc, char **argv) {
	
	int whichRect, numRects;
	float rectHeight, totalArea, leftXPos;
    
	numRects = 1000000;		/* And each rectangle has a width of 1/numRects */
	totalArea = 0;

	if(CUDA_ENABLED){
		printf("CUDA is enabled\n");
	}
	else{
		printf("Cuda is not enabled\n");
	}
	
	        
	for (whichRect=0; whichRect < numRects; whichRect++){
	
		leftXPos = whichRect * ( 1.0 / numRects);	/* this rectangle's x-position */
		
		rectHeight = sqrt( 1 - pow( leftXPos, 2) );	/* height = y = sqrt( r^2 - x^2 ) */
		
		totalArea += rectHeight * ( 1.0 / numRects );	/* a running total of the areas */
	}

    printf("Pi is about %f\n", 4*totalArea);
    
    return 0;
}
