from numpy.linalg import cond, norm
import numpy as np
from matplotlib import pyplot as plt
import glob, re
import argparse


for step in [5, 8]:
    f = 'ring-data'
    folder = f'julia_ring-data_res=20_step={step}'
    p = 'jpade-data'

    ts = sorted([int(re.split('=|\.', x)[-2]) for x in glob.glob(f'{folder}/jextrapolation*')])
    print(ts)
    # print([re.split('=|\.', x) for x in glob.glob(f'{folder}/extrapolation*')])
    # quit()
    last = sorted([int(re.split('=|\.', x)[-2]) for x in glob.glob(f'{folder}/{f}*')])[-1]

    with np.load(f'{folder}/{f}_t={last}.npz') as data:
        dft = data['dft']

    ground_truth = np.log10(np.abs(dft)**2 / max(np.abs(dft)**2))
    # plt.plot(freqs, ground_truth)
    # plt.show()

    norms = []
    norms2 = []
    pnorms = []
    qnorms = []

    for t in ts:
        with np.load(f'{folder}/{p}_t={t}.npz') as data:
            pade_approx = data['pade_approx']
            pn = data['p']
            qn = data['q']

        # print(len(pn), len(qn))
        # plt.plot(pn, label=f'{t}')

        with np.load(f'{folder}/{f}_t={t}.npz') as data:
            dft = data['dft']
            # fcen, df, dt = data['domain']
            # freqs = data['freqs']

        meep = np.log10(np.abs(dft)**2 / max(np.abs(dft)**2))

        loggy = np.log10(np.abs(pade_approx)**2 / max(np.abs(pade_approx)**2))
        normy = norm(loggy - ground_truth) / norm(ground_truth)

        pnorms.append(np.log10(norm(pn)))
        qnorms.append(np.log10(norm(qn)))

        norms.append(normy)
        norms2.append(norm(meep - ground_truth) / norm(ground_truth)) 

    # plt.legend()
    # plt.show()

    pnormdiff = []
    qnormdiff = []

    for i in range(len(pnorms)-1):
        pnormdiff.append(pnorms[i+1] - pnorms[i])
        qnormdiff.append(qnorms[i+1] - qnorms[i])

    # plt.plot(ts[1:], pnormdiff, marker='o')
    # plt.plot(ts[1:], qnormdiff, marker='o')
    # plt.show()
    # quit()

    plt.semilogy(ts, norms, '-o', label=f'|pade - ground|, step={step}')
    # plt.semilogy(ts, norms2, '-o', label='|meep - ground|')
    # plt.xlabel('Meep Time Steps')
    # plt.ylabel('Error')
    # plt.legend()
    # plt.show()



    meeps = []

    ts = sorted([int(re.split('=|\.', x)[-2]) for x in glob.glob(f'{folder}/{f}*')])[1:]
    print(ts)
    for t in ts:
        with np.load(f'{folder}/{f}_t={t}.npz') as data:
            dft = data['dft']

        meep = np.log10(np.abs(dft)**2 / max(np.abs(dft)**2))
        meeps.append(norm(meep - ground_truth)/norm(ground_truth))

    print(meeps[-1] - meeps[-2])

    plt.loglog(ts, meeps, '-o', label=f'|meep - ground|')

plt.xlabel('Meep Time Steps')
plt.ylabel('Error')
plt.legend()
plt.show()