/******************************************************************************
* FILE: pi_serial.c
* 
* DESCRIPTION:
*   Computing Pi Example - Serial
*   This example demonstrates how to compute pi using monte carlo method
*
*
* AUTHOR: Felipe Gutierrez
******************************************************************************/

#include <stdio.h>      /* to use printf() */
#include <math.h>   /* to use sqrt() */
#include <time.h> /* time() */
#include <stdlib.h> /* srand() and rand() */



/* generate a random floating point number from min to max */
double randfrom(double min, double max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}


int main (int argc, char **argv) 
{
    srand(time(NULL)); /* Give a seed to the rand() */
    int numPoints;
    int radius; /* Radius of circle */
    int i;
    double areaRatio;
    double pi;
    double squareArea;
    double circleArea;
    double randomX; /* Generate a random number between 0 and radius*/
    double randomY; /* Generate random number between 0 and radius */

    numPoints = 20e6;
    radius = 1;
    squareArea = numPoints;

    for(i = 0;i < numPoints;i++)
    {
        randomX = randfrom(0,radius);

        randomY = randfrom(0,radius);
        if(sqrt(randomX*randomX + randomY*randomY) <= radius)
        {
            circleArea = circleArea + 1;
        }

    }

    areaRatio = circleArea / squareArea;
    pi = 4 * (areaRatio);

    printf("Pi is about %f\n", pi);
    
    return 0;
}
