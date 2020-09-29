import numpy as np


class ExperienceHolder:
    def __init__(self, capacity, cells):
        self.capacity = capacity
        self.cells = cells
        self.mem = np.empty(shape=(cells, capacity), dtype=object)
        self.index_head = 0
        self.index_tail = 0

        self.full = False

    def size(self):
        if self.full:
            return self.capacity

        direct = self.index_tail - self.index_head
        if direct <= 0:
            direct += self.capacity
        return direct

    def save(self, entry):
        for e in range(len(entry)):
            self.mem[e, self.index_tail] = entry[e]

        self.index_tail = (self.index_tail+1) % self.capacity
        if self.full:
            self.index_head = self.index_tail
        elif self.index_tail == self.index_head:
            self.full = True

    def replay(self, size):
        experience_numbers = np.random.choice(self.size(), size=size, replace=False)
        experience_numbers = (experience_numbers + self.index_head) % self.capacity
        return [np.vstack(e) for e in self.mem[:, experience_numbers]]
