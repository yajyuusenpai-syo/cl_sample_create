import csv
import itertools
import numpy as np
import random

def generate(x, y, z, absorption):
    # 組み合わせ
    combinations = list(itertools.product(x, y, z, absorption))

    random.seed(0)
    random.shuffle(combinations)

    with open('room_list.csv', mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'x', 'y', 'z', 'absorption'])

        for idx, combo in enumerate(combinations):
            writer.writerow([str(idx).zfill(4)] + list(combo))

if __name__ == '__main__':
    # room_size: np.ndarray = np.array([9., 7.5, 3.5])
    X = np.arange(7.0, 10.5, 0.5)
    Y = np.arange(6.0, 8.5, 0.5)
    Z = np.arange(2.5, 4.5, 0.5)

    # absorption: float = 0.5
    A = [0.4, 0.5, 0.6]

    generate(X, Y, Z, A)