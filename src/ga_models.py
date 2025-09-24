# src/ga_models.py

import random
import math


class RouteGA:
    """
    Classe para implementar um Algoritmo Genético para o Problema do Caixeiro Viajante (TSP).
    """

    def __init__(self, cities):
        self.cities = cities
        self.city_names = list(cities.keys())
        self.num_cities = len(self.city_names)

    def calculate_distance(self, city1_name, city2_name):
        """Calcula a distância euclidiana entre duas cidades."""
        x1, y1 = self.cities[city1_name]
        x2, y2 = self.cities[city2_name]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calculate_fitness(self, route):
        """
        Calcula o fitness de uma rota. Rotas mais curtas são melhores.
        O fitness aqui é a distância total da rota.
        """
        total_distance = 0
        for i in range(self.num_cities - 1):
            total_distance += self.calculate_distance(route[i], route[i + 1])
        total_distance += self.calculate_distance(route[-1], route[0])  # Retorno ao início
        return total_distance

    def create_initial_population(self, pop_size):
        """Cria uma população inicial de rotas aleatórias."""
        population = []
        for _ in range(pop_size):
            route = random.sample(self.city_names, self.num_cities)
            population.append(route)
        return population

    def tournament_selection(self, population, tournament_size=3):
        """Seleciona o melhor cromossomo de um subconjunto aleatório."""
        tournament = random.sample(population, tournament_size)
        return min(tournament, key=self.calculate_fitness)

    def order_crossover(self, parent1, parent2):
        """Order Crossover (OX) para criar um novo filho a partir de dois pais."""
        start, end = sorted(random.sample(range(self.num_cities), 2))
        child = [None] * self.num_cities
        child[start:end] = parent1[start:end]

        parent2_cities_for_child = [city for city in parent2 if city not in child]

        p2_idx = 0
        for i in range(self.num_cities):
            if child[i] is None:
                child[i] = parent2_cities_for_child[p2_idx]
                p2_idx += 1
        return child

    def swap_mutation(self, chromosome, mutation_rate=0.05):
        """Troca a posição de duas cidades aleatoriamente com uma dada probabilidade."""
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(self.num_cities), 2)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        return chromosome