import numpy as np
from numpy import inf

# Matrice des distances entre les villes
d = np.array([[0, 10, 12, 11, 14],
              [10, 0, 13, 15, 8],
              [12, 13, 0, 9, 14],
              [11, 15, 9, 0, 16],
              [14, 8, 14, 16, 0]])

# Paramètres du problème
iteration = 3
n_ants = 5
n_citys = 5

# Initialisation des paramètres de l'algorithme des fourmis
m = n_ants
n = n_citys
e = 0.5         # Taux d'évaporation des phéromones
alpha = 1       # Facteur de phéromones
beta = 2        # Facteur de visibilité

# Calcul de la visibilité de la prochaine ville (visibility(i, j) = 1/d(i, j))
visibility = 1 / (d + 1e-10)

visibility[visibility == inf] = 0

# Initialisation des phéromones sur les chemins entre les villes
pheromone = 0.1 * np.ones((m, n))

# Initialisation des routes des fourmis avec une taille de n+1
rute = np.ones((m, n+1))

# Initialisation de la meilleure route
best_route = np.zeros(n+1)
best_dist_cost = float('inf')

# Boucle principale de l'algorithme
for ite in range(iteration):
    rute[:, 0] = 1  # Position initiale et finale de chaque fourmi : ville 1
    
    # Boucle sur chaque fourmi
    for i in range(m):
        temp_visibility = np.array(visibility)  # Création d'une copie de la visibilité
        
        # Boucle sur chaque ville
        for j in range(n-1):
            cur_loc = int(rute[i, j] - 1)  # Ville actuelle de la fourmi
            temp_visibility[:, cur_loc] = 0  # Mise à zéro de la visibilité de la ville actuelle
            
            p_feature = np.power(pheromone[cur_loc, :], beta)
            v_feature = np.power(temp_visibility[cur_loc, :], alpha)
            
            p_feature = p_feature[:, np.newaxis]
            v_feature = v_feature[:, np.newaxis]
            
            combine_feature = np.multiply(p_feature, v_feature)
            total = np.sum(combine_feature)
            
            probs = combine_feature / total
            cum_prob = np.cumsum(probs)
            
            r = np.random.random_sample()
            city = np.nonzero(cum_prob > r)[0][0] + 1
            rute[i, j+1] = city
            
        left = list(set([i for i in range(1, n+1)]) - set(rute[i, :-2]))[0]
        rute[i, -2] = left
    
    rute_opt = np.array(rute)
    dist_cost = np.zeros((m, 1))
    
    # Calcul de la distance totale pour chaque fourmi
    for i in range(m):
        s = 0
        for j in range(n-1):
            s = s + d[int(rute_opt[i, j])-1, int(rute_opt[i, j+1])-1]
        dist_cost[i] = s
    
    # Sélection de la meilleure route
    dist_min_loc = np.argmin(dist_cost)
    dist_min_cost = dist_cost[dist_min_loc]
    
    # Mise à jour de la meilleure route si nécessaire
    if dist_min_cost < best_dist_cost:
        best_route = rute[dist_min_loc, :]
        best_dist_cost = dist_min_cost
    
    # Évaporation des phéromones
    pheromone = (1 - e) * pheromone
    
    # Mise à jour des phéromones sur les meilleures routes
    for i in range(m):
        for j in range(n-1):
            dt = 1 / dist_cost[i]
            pheromone[int(rute_opt[i, j])-1, int(rute_opt[i, j+1])-1] += dt

# Affichage des résultats
print('Routes de toutes les fourmis à la fin :')
print(rute_opt)
print()
print('Meilleur chemin :', best_route)
print('Coût du meilleur chemin :', int(best_dist_cost[0]) + d[int(best_route[-2])-1, 0])