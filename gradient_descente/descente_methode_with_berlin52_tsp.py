import random
import math
import time

# Function to calculate the distance between two cities
def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

# Function to find the nearest unvisited city from a given city
def find_nearest_city(city, unvisited_cities):
    min_distance = float('inf')
    nearest_city = None
    for next_city in unvisited_cities:
        dist = distance(city, next_city)
        if dist < min_distance:
            min_distance = dist
            nearest_city = next_city
    return nearest_city

# Function to calculate the total distance of a route
def total_distance(route, cities):
    total = 0
    for i in range(len(route) - 1):
        total += distance(cities[route[i]], cities[route[i + 1]])
    total += distance(cities[route[-1]], cities[route[0]])  # Return to the starting city
    return total

# Function to read the coordinates from the TSP file
def read_tsp(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        coord_section = False
        cities = []

        for line in lines:
            if "NODE_COORD_SECTION" in line:
                coord_section = True
                continue
            if "EOF" in line:
                coord_section = False

            if coord_section:
                line = line.strip()
                if line:
                    city_info = line.split()
                    cities.append((float(city_info[1]), float(city_info[2])))

    return cities

# Greedy algorithm to create an initial solution
def greedy_initial_solution(cities):
    current_city = cities[0]  # Start from the first city
    unvisited_cities = cities[1:]  # List of unvisited cities
    solution = [0]  # Initialize the solution with the index of the starting city

    while unvisited_cities:
        nearest_city = find_nearest_city(current_city, unvisited_cities)
        solution.append(cities.index(nearest_city))
        unvisited_cities.remove(nearest_city)
        current_city = nearest_city

    return solution

# Descent method for the Traveling Salesman Problem (TSP)
def descent_method(cities, max_iterations):
    num_cities = len(cities)
    initial_solution = greedy_initial_solution(cities)
    best_solution = initial_solution
    best_distance = total_distance(initial_solution, cities)

    start_time = time.time()

    for _ in range(max_iterations):
        i, j = random.sample(range(num_cities), 2)
        new_solution = best_solution[:]
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        new_distance = total_distance(new_solution, cities)

        if new_distance < best_distance:
            best_solution = new_solution
            best_distance = new_distance

    end_time = time.time()
    execution_time = end_time - start_time

    return best_solution, best_distance, execution_time

# File path for berlin52.tsp - change this to your file path
file_path = './berlin52.tsp'

# Read the coordinates from the file
cities = read_tsp(file_path)

max_iterations = 1000

best_route, best_distance, execution_time = descent_method(cities, max_iterations)

print("Best Route:", best_route)
print("Best Distance:", best_distance)
print("Execution Time:", execution_time, "seconds")
