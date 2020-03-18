# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import random

position = 0
walk = [position]
steps = 1000
for i in range(100):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)

plt.plot(walk[:100])
plt.savefig('numpy-3.png')


