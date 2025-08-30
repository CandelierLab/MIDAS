import os
import matplotlib.pyplot as plt

import MIDAS

os.system('clear')

# ═══ Parameters ═══════════════════════════════════════════════════════════

dataDir = '/home/raphael/Bureau/data/'

# ══════════════════════════════════════════════════════════════════════════

S = MIDAS.storage(dataDir + 'test.db')

# ─── Time

T = S.get_time(0)

# plt.style.use('dark_background')
# fig, ax = plt.subplots()
# ax.scatter(T[:,2], T[:,3])
# plt.show()

# ─── Trajectory

T = S.get_trajectory(0)

plt.style.use('dark_background')
fig, ax = plt.subplots()
ax.plot(T[:,2], T[:,3], '.-')
plt.show()