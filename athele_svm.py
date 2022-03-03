# for data
import pandas as pd
import numpy as np
# for plotting
import matplotlib.pyplot as plt
# for training
from sklearn.preprocessing import normalize

# read athleteEvents
df = pd.read_csv("athleteEvents.csv")

df = df[['Team','Medal','Sport','Year']]
# drop rows that contain 'N/A' in 'Medal'
df = df.dropna()
# drop uninterst year interval
df = df[df['Year']>=1980]
df = df[df['Year']<=2016]

# group based on team and medal
gtm = df.groupby(['Team']).count().filter(['Medal'])
gtm = gtm[gtm['Medal'] > 100]

# group based on team and sport
gts = df.groupby(['Sport']).count().filter(['Medal'])

# group based on team, medal and sport
gtsm = df.groupby(['Team','Sport']).count().filter(['Medal'])
gtsm = gtsm[gtsm.index.get_level_values('Team').isin(gtm.index)]

# create matrix
M = np.zeros((len(gtm),len(gts)))
for i in range(len(gtm)):
    for j in range(len(gts)):
        sport = gts.iloc[j].name
        team = gtm.iloc[i].name
        if (team,sport) in gtsm.index:
            M[i,j] = gtsm.loc[(team, sport)]

# pre-process the data
mean = M.mean(axis=0)
M = M - mean[np.newaxis,:]
M = normalize(M, axis=0, norm='l1')

# svd
u, s, vh = np.linalg.svd(M, full_matrices=True)

# find top 10 singular values' corresponding sport and team
list = []
for i in range(10):
    # find ui for sport
    U1 = np.zeros(len(gtm))
    inner_max = 0
    u1 = u.T[i]
    idx_U1 = np.argmax(u1)
    team = gtm.iloc[idx_U1].name
    # find vi for team
    V1 = np.zeros(len(gtm))
    inner_max = 0
    v1 = vh[i]
    idx_V1 = np.argmax(v1)
    sport = gts.iloc[idx_V1].name
    list.append((team,sport))

# rank teams based on certain left singular vector
rank = 1
u_rank = u.T[rank]
idx_u = np.argsort(u_rank)
team_sorted = gtm.iloc[idx_u].index.tolist()

# rank sports based on certain right singular vector
rank = 1
vh_rank = vh[rank]
idx_vh = np.argsort(vh_rank)
sport_sorted = gts.iloc[idx_vh].index.tolist()

# visualize team rank table
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
list = []
for i in range(len(team_sorted)):
    team = team_sorted[i]
    ui = np.sort(u_rank)[i]
    list.append([team,ui])

rcolors = plt.cm.BuPu(np.full(len(list), 0.1))
ccolors = plt.cm.BuPu(np.full(2, 0.1))
ax.table(cellText=list, loc='center')

fig.tight_layout()

# visualize sport rank table
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
list = []
for i in range(len(sport_sorted)):
    sport = sport_sorted[i]
    vi = np.sort(vh_rank)[i]
    list.append([sport,vi])

rcolors = plt.cm.BuPu(np.full(len(list), 0.1))
ccolors = plt.cm.BuPu(np.full(2, 0.1))
ax.table(cellText=list, loc='center')

fig.tight_layout()

plt.show()