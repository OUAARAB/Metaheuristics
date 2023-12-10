# Simulated Annealing pour le Problème du Voyageur de Commerce (TSP)

Ce script Python implémente l'algorithme du recuit simulé pour résoudre le Problème du Voyageur de Commerce (TSP). Le TSP est un problème d'optimisation classique où l'objectif est de trouver le trajet le plus court qui visite un ensemble de villes et retourne à la ville de départ.

## Utilisation

1. **Installation des dépendances :**
   ```bash
   pip install tsplib95 

tsplib95 : Un package Python pour travailler avec des instances du problème TSP.

matplotlib : Une bibliothèque de tracé 2D pour créer des visualisations.

Google Colab : Utilisé pour monter Google Drive et charger les instances du TSP.

2. **Algorithme :**
Le script utilise le recuit simulé pour améliorer itérativement une solution au TSP. Le recuit simulé est un algorithme d'optimisation probabiliste inspiré du processus de recuit en métallurgie. Il accepte de nouvelles solutions même si elles sont moins bonnes que la solution actuelle, avec une probabilité décroissante à mesure que l'algorithme progresse.
La représentation de la solution du TSP est une séquence de villes, et l'algorithme explore des solutions voisines par le biais d'opérations telles que l'inversion, l'insertion, le remplacement et le remplacement de routes.

3. **Entrée :**
Le script charge les instances du TSP à partir d'un répertoire Google Drive. Vous pouvez remplacer le chemin du fichier dans le script par le chemin de votre instance du TSP.

4. **Sortie :**
Le script produit en sortie le meilleur trajet trouvé, sa distance, ainsi que des informations liées à la convergence telles que les coûts maximum, minimum et moyens. Il trace également les trajets finaux à l'aide de matplotlib.

5. **Paramètres :**
   
Température initiale : 50

Alpha (facteur de réduction de la température) varie de 0.85 a 0.99

Conditions d'arrêt : 500 itérations avec la même solution, 8000 itérations avec la même différence de coût.


7. **Auteurs :**

BEN JAAFAR Chaima

OUATTOU Fatima

BOUSSELHAM Oumaima
