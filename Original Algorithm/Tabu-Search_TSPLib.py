import random
import time
import copy
import math
import hashlib
from random import randint

class TabuSearch:

    def __init__(self, coordinates) -> None:
        self.coordinates = coordinates
        self.number_city = len(coordinates)
        self.distance_list = self.calculate_distances()
        self.iteration = 500
        self.tabu_length = 10
        self.tabu_list = []

    #Read node coordinates from the input file
    @staticmethod
    def read_coordinates(file_path):
        coordinates = []
        with open(file_path, "r") as f:
            in_node_coord_section = False
            for line in f:
                if in_node_coord_section:
                    if line.strip() == "EOF":
                        break
                    parts = line.split()
                    x = float(parts[1])
                    y = float(parts[2])
                    coordinates.append((x, y))
                elif line.strip() == "NODE_COORD_SECTION":
                    in_node_coord_section = True
        return coordinates


    #Calculate Euclidean distances between nodes
    def calculate_distances(self):
        distance_list = [[0.0] * self.number_city for _ in range(self.number_city)]
        for i in range(self.number_city):
            for j in range(self.number_city):
                x1, y1 = self.coordinates[i]
                x2, y2 = self.coordinates[j]
                distance = round(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
                distance_list[i][j] = distance
        return distance_list

    #Generate an initial route based on node coordinates
    def initial_route(self):
        route = list(range(self.number_city))
        random.shuffle(route)
        cost = self.calculate_cost(route)
        return route, cost

    #Calculate the total distance of a route
    def calculate_cost(self, route):
        cost = 0
        for i in range(len(route)):
            current_node = route[i]
            next_node = route[(i + 1) % self.number_city]
            cost += self.distance_list[current_node][next_node]
        return cost
    
    #Hash a solution to store it in tabu list
    def tsp_solution_hash(self,solution):
        solution_str = ','.join(map(str, solution))
        hash_object = hashlib.sha256(solution_str.encode())
        hash_value = hash_object.hexdigest()
        return hash_value

    #Check if a solution is in tabu list
    def is_in_tabu_list(self, tabu):
        return tabu in self.tabu_list

    #Add a solution to tabu list
    def add_tabu_list(self, tabu):
        self.tabu_list.append(tabu)
        if len(self.tabu_list) > self.tabu_length:
            self.tabu_list.pop(0)

    #Generate neighbourhood based on given solution by performing two-opt swap
    def two_opt_swap(self,route):
        neighbors = []
        
        for i in range(len(route)):
            node1 = 0
            node2 = 0
            
            while node1 == node2:
                node1 = randint(1, len(route)-1)
                node2 = randint(1, len(route)-1)
                
            if node1 > node2:
                swap = node1
                node1 = node2
                node2 = swap
                
            tmp = route[node1:node2]
            tmp_route = route[:node1] + tmp[::-1] +route[node2:]
            neighbors.append(tmp_route)
            
        return neighbors

    #Tabu search
    def tabu_search(self, route):
        results = []
        best_candidate = copy.deepcopy(route)
        best_candidate_cost = self.calculate_cost(best_candidate)
        results.append(route)

        while self.iteration > 0:
            
            #Instead of looking for all possible combinations (takes too much time) in a solution to generate a new neighbourhood then performing a swap or two-opt or or-opt
            #We will use a method that generate a neighbourhood while doing two opt and swap at the same time

            neighbors = self.two_opt_swap(route) 


            for sol in neighbors:
                current_candidate = sol
                current_candidate_cost = self.calculate_cost(current_candidate)
                current_candidate_hash = self.tsp_solution_hash(current_candidate)

                if current_candidate_cost < best_candidate_cost or not self.is_in_tabu_list(current_candidate_hash):
                        best_candidate = current_candidate
                        best_candidate_cost = current_candidate_cost
                        tabu = current_candidate_hash
                        self.add_tabu_list(tabu)
            if best_candidate_cost < self.calculate_cost(route):
                route = best_candidate

            results.append(best_candidate)

            self.iteration -= 1

        return route, results

if __name__ == "__main__":
    file_path = "berlin52.tsp"
    coordinates = TabuSearch.read_coordinates(file_path)
    print("Coordinates read from the file:")
    for i, (x, y) in enumerate(coordinates):
        print(f"Node {i + 1}: x={x}, y={y}")

    best_solution_cost = float('inf')
    worst_solution_cost = float('-inf')
    total_solution_cost = 0
    total_execution_time = 0
    all_solutions = []

    for _ in range(10):
        ts = TabuSearch(coordinates)
        route, initial_cost = ts.initial_route()
        print("Running ...")

        t = time.time()
        best_route, _ = ts.tabu_search(route)
        execution_time = time.time() - t
        cost = ts.calculate_cost(best_route)

        all_solutions.append((best_route, cost))
        total_solution_cost += cost
        total_execution_time += execution_time

        if cost < best_solution_cost:
            best_solution_cost = cost
            best_solution_route = best_route

        if cost > worst_solution_cost:
            worst_solution_cost = cost
            worst_solution_route = best_route

    average_solution_cost = total_solution_cost / 10
    average_execution_time = total_execution_time / 10

    print()
    print("Results:")
    print(f'Highest cost: {worst_solution_cost}')
    print(f'Worst solution: {worst_solution_route}')
    print()
    print(f'Lowest cost: {best_solution_cost}')
    print(f'Best solution: {best_solution_route}')
    print()
    print(f'Average cost: {average_solution_cost}')
    print()
    print(f'Average time spent: {average_execution_time} seconds')
    print(f'Total time spent: {total_execution_time} seconds')