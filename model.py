import numpy as np
import matplotlib.pyplot as plt

# Randomly group individuals into pairs.
# Return a tuple of 2 arrays, the first item of each array are a pair etc...
def pairs(N):
    assert N % 2 == 0
    arr = np.arange(N)
    np.random.shuffle(arr)
    return (arr[:-N//2], arr[N//2:])

# Simulate an economic exchange between pairs of individuals.
# The arguments, a and b, represent the current wealth of each pair.
# Return a tuple of 2 arrays, representing the new wealth of each pair.
def kinetic_exchange(a, b):
    assert len(a) == len(b)
    N = len(a)
    rand = np.random.uniform(0, 1, N)
    anew = rand * (a + b)
    bnew = (1 - rand) * (a + b)
    return anew, bnew

# Compute the gini coefficient of the array of individuals wealth, w.
def gini(w):
    N = len(w)
    w = np.sort(w)
    i = np.arange(1, N + 1)
    G = 2*(np.sum(i * w) / (N * np.sum(w))) - (1 + 1/N)
    return G

# Compute one timestep on w, the array of individuals wealth.
# Let gs be the income of each individual, and r be the return on capital.
def timestep(w, gs, r):
    assert len(w) == len(gs)
    w *= 1 + r
    ai,bi = pairs(len(w))
    w[ai],w[bi] = kinetic_exchange(w[ai], w[bi])
    w += gs
    # Normalize wealth.
    total = np.sum(w)
    w *= len(w) / total

# Decide the income for each individual.
# N is the individual count, g is the rate of economic growth.
# Return an array of income for each individual.
def income_distribution(N, g):
    gs = np.full(N, g, dtype=float)
    ai,bi = pairs(N)
    gs[ai],gs[bi] = kinetic_exchange(gs[ai], gs[bi])
    return gs

# Run the economic model of N individuals with return on capital, r, and economic
# growth, g.
# Return the average gini coefficient over the 50 timestamps after T timestamps
# have passed.
def sim(N, T, r, g):
    # Each individual starts with 1 unit wealth.
    w = np.full(N, 1, dtype=float)
    gs = income_distribution(N, g)
    for t in range(T):
        timestep(w, gs, r)
    gini_sum = 0
    for t in range(50):
        timestep(w, gs, r)
        gini_sum += gini(w)
    return gini_sum / 50;

individual_count = 50000
timesteps = 100
income = 5
return_on_capital = np.linspace(0, 19, 20)

gini_results = np.zeros(len(return_on_capital))
for i,r in enumerate(return_on_capital):
    gini_results[i] = sim(individual_count, timesteps, r / 100, income / 100)

fix,ax = plt.subplots()
ax.plot(return_on_capital, gini_results)
ax.axvline(x = income, label='Fixed 5% economic growth')
ax.set_xlabel('Return on capital (%)')
ax.set_ylabel('Gini coefficient')
plt.savefig('plot.png', bbox_inches='tight', pad_inches=0)
