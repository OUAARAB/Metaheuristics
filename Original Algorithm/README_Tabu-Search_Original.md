# Tabu Search Algorithm

In this project we will implement Tabu-Search algorithm in Python to run on [TSPLib](https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) instances **[Berlin52](https://github.com/pdrozdowski/TSPLib.Net/blob/master/TSPLIB95/tsp/berlin52.tsp)** and **[CH130](https://github.com/pdrozdowski/TSPLib.Net/blob/master/TSPLIB95/tsp/ch130.tsp)**.


# Algorithm
The algorithm start by generating a new solution and an empty tabu list ,based on the generated solution it generates a new neighborhood of solutions,  it then finds the best neighbor and checks whether it is in the tabu list or not , if it is it checks the aspiration criteria , if it doesn't it makes it the new best solution , add it to the tabu list and iterates if the stopping criteria isn't already met.
 

![Tabu Search - an overview | ScienceDirect Topics](https://ars.els-cdn.com/content/image/3-s2.0-B9780128188804000119-f10-05-9780128188804.jpg)


# Script

The script has a Tabu search class which contains a constructor that initialize the parameters of the algorithm and 13 methods.
Parameters :

 - Number of iterations (iteration)
 - Tabu list length (tabu_length)

To generate a new neighborhood we use two opt and swap at the same time.
**Swap** will swap two indexes in a route.
**Two-opt** will take a segment of a route and reverse it.

# How to run the script

In order to run the script you will need to download the desired TSpLib instance either Berlin52 or Ch130 put it in the same location as the script, open the script , head down to main function and in file_path either put the full path or just change the name of the file and run the algorithm.
The algorithm will then perform 10 runs (to change in the main function aswell) and give you the best found solution , the worst and the average along with the spent time.
The higher the iterations the better the solution.
