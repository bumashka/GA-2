import sys
import random
import time

import numpy as np
import matplotlib.pyplot as plt


class MDGA:

    def __init__(self, config):
        # Максимальное количество хромосом в популяции
        self.max_population = config['max_population']
        # Максимальное количество эпох
        self.max_epochs = config['max_epochs']
        # Актуальная эпоха
        self.current_epoch = 0
        # Вероятность кроссовера
        self.crossover_chance = config['crossover_chance']
        # Вероятность мутации
        self.mutation_chance = config['mutation_chance']
        # Нижняя граница
        self.lower_bound = float(config['lower_bound'])
        # Верхняя граница
        self.upper_bound = config['upper_bound']
        # Количество точек, на которые делим отрезок
        self.chromosome_length = 14
        # Размерность
        self.n = config["dimension"]
        # Функция, с которой работаем
        self.function = lambda x: np.sum([5 * (i + 1) * x[i] ** 2 for i in range(0, self.n)])
        # Актуальная популяция хромосом

        self.population = [[random.uniform(self.lower_bound, self.upper_bound) for i in
                            range(0, self.n)] for x in range(0, self.max_population)]
        # Потомки актуальной популяции
        self.children = []
        # Лучшее решение актуальной популяции
        self.current_best_solution = sys.maxsize - 1

    def f(self, x, y):
        return np.sum(5 * 1 * x ** 2 + 5 * 2 * y ** 2)

    def selection(self):
        reselected_population = []
        f = [self.function(x) for x in self.population]
        sum_f = sum(f)
        fitness_function = [self.function(x) / sum_f for x in self.population]
        for i in range(0, self.max_population):
            for ff in range(0, int(abs(fitness_function[i]))):
                reselected_population.append(self.population[i])
            if random.uniform(0, 1) <= int(abs(fitness_function[i]) % 1 * 1000):
                reselected_population.append(self.population[i])
        self.population = reselected_population

    def crossover(self):
        # Арифметический кроссинговер
        for i in range(0, len(self.population)):
            if random.uniform(0, 1) <= self.crossover_chance:
                chrom_a = self.population[i]

                chrom_b = self.population[random.randint(0, len(self.population) - 1)]

                w = 0.3

                chrom_a_ = [w * chrom_a[i] + (1 - w) * chrom_b[i] for i in range(self.n)]
                chrom_b_ = [(1 - w) * chrom_a[i] + w * chrom_b[i] for i in range(self.n)]

                self.children.append(chrom_a_)
                self.children.append(chrom_b_)

    def mutation(self):
        for x in range(len(self.children)):
            if round(random.uniform(0, 1), 3) <= self.mutation_chance:
                for i in range(self.n):
                    self.children[x][i] = random.uniform(self.lower_bound, self.upper_bound)

    def sort(self):
        return lambda x: self.function(x)

    def reduction(self):
        self.population = self.population + self.children
        self.population.sort(key=self.sort(), reverse=False)
        self.population = self.population[:self.max_population]

    def plot_graph(self):
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection='3d')

        x = np.arange(self.lower_bound, self.upper_bound, 0.01)
        y = np.arange(self.lower_bound, self.upper_bound, 0.01)
        x, y = np.meshgrid(x, y)

        a = 5

        z = a * x * x + a * 2 * y * y

        X = np.array([self.population[x][0] for x in range(len(self.population))])
        Y = np.array([self.population[y][1] for y in range(len(self.population))])
        Z = np.array([self.function(x) for x in self.population])

        ax.plot_wireframe(x, y, z, cmap='Blues')
        ax.grid(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        fig.show()
        ax.scatter(X, Y, Z, c='red', linewidth=0.5)
        ax.set_title(f'График функции для {self.current_epoch} эпохи')
        ax.view_init(-90, 90)
        fig.show()

    def run(self):
        epochs_without_changes = 0
        # self.plot_graph()
        execution_time = 0
        while self.current_epoch != self.max_epochs and epochs_without_changes != 10:
            self.current_epoch += 1
            start = time.time()
            self.selection()
            self.crossover()
            self.mutation()
            self.reduction()
            end = time.time()
            execution_time += (end - start)
            best_solution = min([self.function(x) for x in self.population])
            print(
                f"""
++++++++++
Эпоха: {self.current_epoch}
Лучшее решение эпохи: {best_solution}
Лучшее решение за все эпохи: {self.current_best_solution}
Количество эпох без изменения результата: {epochs_without_changes}
Время вычисления эпохи в секундах: {end - start}
Время вычисления для всех эпох в секундах: {execution_time}
++++++++++
"""
            )
            if best_solution < self.current_best_solution:
                self.current_best_solution = best_solution
                epochs_without_changes = 0
            elif best_solution == self.current_best_solution:
                epochs_without_changes += 1
            else:
                epochs_without_changes = 0
            # self.plot_graph()

        with open("result.txt", "a") as file:
            file.write(
                f"\n{self.mutation_chance},{self.crossover_chance},{self.max_population},{execution_time},"
                f"{self.current_epoch},{self.current_best_solution}")


def main():
    for i in [0.25, 0.5, 0.7, 0.9]:
        for j in [0.001, 0.01, 0.1, 0.5, 0.8]:
            for k in [30, 60, 90]:
                config = {
                    "crossover_chance": i,
                    "mutation_chance": j,
                    "max_population": k,
                    "max_epochs": 20,
                    "lower_bound": -5.12,
                    "upper_bound": 5.12,
                    "dimension": 2
                }
                GA = MDGA(config)
                GA.run()


if __name__ == '__main__':
    random.seed(round(time.time()))
    main()
