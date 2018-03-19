import numpy as np
import matplotlib.pyplot as plt

data = dict()
with open("convergence.txt", "r") as f:
    for rows in f:
        columns = rows.split(' ')

        resolution = int(columns[0])
        order      = int(columns[1])

        if not order in data:
            data[order] = dict()

        if not resolution in data[order]:
            data[order][resolution] = dict()

        data[order][resolution]['e_abs']  = columns[3]
        data[order][resolution]['e_rel1'] = columns[4]
        data[order][resolution]['e_rel2'] = columns[5]
        data[order][resolution]['hmin']   = columns[6]
        data[order][resolution]['hmax']   = columns[7]

for order in data.keys():
    res = data[order].keys()
    x = np.array([data[order][r]['hmax'] for r in res], dtype=float)
    y = np.array([data[order][r]['e_rel1'] for r in res], dtype=float)
    plt.loglog(x, y, '-o', label="Order %d"%order)

    C  = y[0]/x[0]**order
    xr = np.linspace(np.min(x), np.max(x), 100)
    yr = C*xr**order
    plt.loglog(xr, yr, 'k--')

plt.grid()
plt.xlabel('hmin')
plt.ylabel('error (relative)')
plt.title('Convergence')
plt.legend(loc='upper left')
plt.show()
