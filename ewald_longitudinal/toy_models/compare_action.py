#!/usr/bin/env python3
"""Identify which friction model a fast Gamma-action code implements.

Step 1 (write a probe):  python compare_action.py make-probe probe.npz --kernel A
        stores Q (M,N,3), V (M,N,3), box and kernel parameters.
Step 2: evaluate Gamma(Q) @ V with the code under test and add the result to the
        probe file as 'GammaV' with shape (M,N,3) (friction force is -GammaV).
Step 3: python compare_action.py check probe.npz

The check reports the relative error against three dense candidate models:
  minimum   - minimum-image pairs only (prl_toy_models.friction_matrix)
  lattice   - full periodic-image sum, k=0 retained (LatticeFriction)
  lattice_no_k0 - lattice with the k=0 coefficient ghat(0)/(3V) I per pair removed
A small error against exactly one candidate identifies the convention. A residual that
changes with the Ewald splitting parameter can come from real-space or Fourier truncation,
mesh (alias/transfer) error or the split implementation as well as from a mishandled k=0
term; decompose it before attributing it. Differences between candidates are model
differences, not Ewald errors.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import prl_toy_models as model


def candidate_actions(Q, V, box, **fric):
    lattice = model.LatticeFriction(box, **fric)
    out = {'minimum': [], 'lattice': [], 'lattice_no_k0': []}
    for q, v in zip(Q, V):
        n = len(q)
        G_lat = lattice.matrix(q)
        uniform = lattice.zero_mode*np.kron(n*np.eye(n)-np.ones((n, n)), np.eye(3))
        out['minimum'].append((model.friction_matrix(q, box, **fric)@v.ravel()).reshape(n, 3))
        out['lattice'].append((G_lat@v.ravel()).reshape(n, 3))
        out['lattice_no_k0'].append(((G_lat-uniform)@v.ravel()).reshape(n, 3))
    return {k: np.asarray(v) for k, v in out.items()}, lattice.record()


def make_probe(path, kernel, gamma, kappa, r_ref, n, box, configs, seed):
    rng = np.random.default_rng(seed)
    tr = model.simulate('lj', kernel, n=n, box=box, burn=400, steps=40*configs, stride=40,
                        gamma=gamma, kappa=kappa, r_ref=r_ref, seed=seed)
    Q = tr['Q'][1:configs+1]
    V = rng.normal(size=Q.shape)
    np.savez(path, Q=Q, V=V, box=np.repeat(box, 3), kernel=np.array(kernel),
             gamma=gamma, kappa=kappa, r_ref=r_ref)
    print(f'Wrote {path}: {len(Q)} configurations of N={n}. Add GammaV (M,N,3) from the code under test.')


def check(path):
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    if 'GammaV' not in data:
        raise SystemExit('Probe file has no GammaV array yet.')
    fric = dict(kernel=str(data['kernel']), gamma=float(data['gamma']),
                kappa=float(data['kappa']), r_ref=float(data['r_ref']))
    test = np.asarray(data['GammaV'], float)
    cands, record = candidate_actions(data['Q'], data['V'], data['box'], **fric)
    rel = {k: float(np.linalg.norm(test-v)/np.linalg.norm(v)) for k, v in cands.items()}
    flipped = {k: float(np.linalg.norm(test+v)/np.linalg.norm(v)) for k, v in cands.items()}
    gaps = {f'{a}_vs_{b}': float(np.linalg.norm(cands[a]-cands[b])/np.linalg.norm(cands[b]))
            for a, b in (('minimum', 'lattice'), ('lattice_no_k0', 'lattice'))}
    report = dict(relative_error=rel, relative_error_if_sign_flipped=flipped,
                  model_gaps=gaps, best_match=min(rel, key=rel.get), lattice_record=record,
                  total_momentum_of_GammaV=np.abs(test.sum(axis=1)).max().item())
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    mk = sub.add_parser('make-probe')
    mk.add_argument('path'); mk.add_argument('--kernel', choices=['A', 'B'], default='A')
    mk.add_argument('--gamma', type=float, default=.5); mk.add_argument('--kappa', type=float, default=.7)
    mk.add_argument('--r-ref', type=float, default=1.3); mk.add_argument('--n', type=int, default=64)
    mk.add_argument('--box', type=float, default=5.5); mk.add_argument('--configs', type=int, default=3)
    mk.add_argument('--seed', type=int, default=7)
    ck = sub.add_parser('check'); ck.add_argument('path')
    args = p.parse_args()
    if args.command == 'make-probe':
        make_probe(args.path, args.kernel, args.gamma, args.kappa, args.r_ref, args.n, args.box,
                   args.configs, args.seed)
    else:
        check(args.path)


if __name__ == '__main__':
    main()
