/******************************************************************************
* FILE: pi-example.c
* DESCRIPTION:
*   Pi - OpenMP
*   Approximate pi using a Left Riemann Sum under a quarter unit circle.
*
*   When running the program, the number of rectangles can be passed using the
*   -r option, e.g. 'pi-example -r X', where X is the number of rectangles.
* 
* ORIGINAL AUTHOR: Aaron Weeden
* MODIFIED BY: Phil List
******************************************************************************/

#include <omp.h>        /* to use OpenMP */
#include <stdio.h>      /* to use printf() */
#include <math.h>       /* to use sqrt() */
#include <stdlib.h>	/* to use malloc(), free() */
#include <unistd.h>	/* to use getopt() */


int main (int argc, char **argv) {

  int whichRect, numRects;
  float rectHeight, totalArea, leftXPos;

  numRects = 1000000;          /* Each rectangle has a width of 1/numRects */
  totalArea = 0;
  double *areas = (double*)malloc(numRects * sizeof(double));

  #pragma omp parallel for private(leftXPos, rectHeight)
  for (whichRect=0; whichRect < numRects; whichRect++)
  {
  //  printf("thread num: %i of %i\n", omp_get_thread_num(), omp_get_num_threads());

    /* this rectangle's x-position */
    leftXPos = whichRect * ( 1.0 / numRects);       
    /* height = y = sqrt( r^2 - x^2 ) */
    rectHeight = sqrt( 1 - pow( leftXPos, 2) );     
    /* A running total of the areas */
    areas[whichRect] = rectHeight * ( 1.0 / numRects );
  }

  /* Sum all rectangle areas*/
  for(whichRect=0; whichRect < numRects; whichRect++)
  {
    totalArea += areas[whichRect];
  }

  /* Since we only calculated the areas from 0 to 1 we were only looking a
   * at a quarter of a circle. So we have to multiply by four (symmetry)
   */
  printf("Pi is about %f\n", 4*totalArea);
  free(areas);
  return 0;
}


