"""Read-only bridge to PR 7 Candidate 1/S1L and independent frozen matrices.

The near edge tensors include every retained periodic image.  Dense matrices
are assembled from pair tensors, never from calls to the production action.
All new configurations and physical reference caches stay in this directory.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time

import protocol as P
np = P.np
from scipy import linalg, special

HERE = Path(__file__).resolve().parent
CLUSTER = HERE.parent
VALIDATION = CLUSTER / 'validation_exp1'
GRID = VALIDATION / 'deconv_psd_comparison'
FINAL = GRID / 'final_exp2_selection'
for folder in (CLUSTER, VALIDATION, GRID):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
import ewald_exp_lib as E
from grid_psd import GridPSDGamma, ProjectedMesh

EPS_PARAMETERS = {
    1e-3: dict(xi=.8, sR=2.293671942361165, sF=3.195712452071397, M=12, p=3),
    1e-4: dict(xi=1.24875, sR=2.9, sF=2.9, M=20, p=4),
    1e-5: dict(xi=1.05, sR=3.111895674293868, sF=3.7596714427763667, M=20, p=6),
}


def position_hash(q, L):
    return hashlib.sha256(np.concatenate((np.asarray(q, float).ravel(), [float(L)])).tobytes()).hexdigest()


def configuration(N=16, family='uniform', seed=0):
    """Box-scaled nested subsets; N=256 is the exact historical geometry.

    This also matches PR 7's existing N=64/128 uniform seed 2 scaling
    references bit for bit.  The smaller subsets are not a regenerated RSA
    ensemble; their actual minimum separations are reported explicitly.
    """
    if int(N) != N or not 2 <= N <= 256:
        raise ValueError('Require integer 2 <= N <= 256')
    if family not in E.FAMS:
        raise ValueError(f'Unknown configuration family {family}')
    historical_seed = int(seed) if int(seed) in E.SEEDS else 2
    base = E.config(family, historical_seed, .75)
    L = float((int(N)/.75)**(1/3))
    q = base['q'].copy() if N == 256 else np.mod(base['q'][:N] * (L/base['L']), L)
    generation = ('exact committed PR 7 configuration' if N == 256 else
                  'nested subset of committed configuration, box-scaled at fixed density')
    cfg = dict(name=base['name'] if N == 256 else f'{family}_N{N}_s{seed}_nested_rho0.75',
               family=family, seed=int(seed), source_seed=historical_seed, rho=.75,
               N=int(N), L=L, q=q, hash=position_hash(q, L)[:12],
               position_sha256=position_hash(q, L), d_min=E.min_periodic_separation(q, L),
               generation=generation)
    path = P.INPUTS / f'q_N{N}_{family}_s{seed}.npz'
    if path.exists():
        with np.load(path) as data:
            if float(data['L']) != L or not np.array_equal(data['q'], q):
                raise RuntimeError(f'Saved configuration differs: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, q=q, L=L)
        P.save(path.with_suffix('.json'), {k:v for k,v in cfg.items() if k != 'q'})
    cfg['position_file'] = str(path)
    cfg['position_file_sha256'] = P.sha(path)
    return cfg


def smooth_at_least(n):
    n = max(2, int(np.ceil(n)))
    while True:
        value = n
        for factor in (2, 3, 5):
            while value % factor == 0:
                value //= factor
        if value == 1:
            return n
        n += 1


def near_edges(op):
    """Aggregate all production images of each unordered pair, in sorted order."""
    keys = op.i.astype(np.int64)*op.N + op.j
    unique, inverse = np.unique(keys, return_inverse=True)
    tensors = np.zeros((len(unique), 3, 3))
    for a in range(3):
        for b in range(a, 3):
            tensors[:, a, b] = tensors[:, b, a] = np.bincount(
                inverse, op.w*op.d[:, a]*op.d[:, b], minlength=len(unique))
    return [(int(key//op.N), int(key % op.N), tensor)
            for key, tensor in zip(unique, tensors)]


def edges_dense(edges, N):
    blocks = np.zeros((N, N, 3, 3))
    for i, j, K in edges:
        blocks[i, i] += K
        blocks[j, j] += K
        blocks[i, j] -= K
        blocks[j, i] -= K
    return blocks.transpose(0, 2, 1, 3).reshape(3*N, 3*N)


def pair_graph_dense(W):
    """Graph conversion with explicit self cancellation, independent of action."""
    W = np.asarray(W, float)
    N = len(W)
    blocks = -W.copy()
    diag = np.arange(N)
    blocks[diag, diag] += W.sum(axis=1)
    return blocks.transpose(0, 2, 1, 3).reshape(3*N, 3*N)


@dataclass
class FrozenSystem:
    q: np.ndarray
    L: float
    op: object
    near_dense: np.ndarray
    long_dense: np.ndarray
    full_dense: np.ndarray
    edges: list
    metadata: dict
    cfg: dict

    def full_action(self, v):
        return self.op.matvec(np.asarray(v, float).ravel())

    def long_action(self, v):
        X = np.asarray(v, float).reshape(self.op.N, 3)
        return (np.einsum('nab,nb->na', self.op.Dm, X) -
                self.op.mesh.apply(self.op.idx, self.op.wt, X)).ravel()


def make_system(N=16, family='uniform', seed=0, epsilon=1e-4, M=None):
    started = time.perf_counter()
    cfg = configuration(N, family, seed)
    q, L = cfg['q'], cfg['L']
    key = next((e for e in EPS_PARAMETERS if np.isclose(e, epsilon, rtol=1e-12, atol=0.)), None)
    if key is None:
        raise ValueError(f'epsilon must be one of {tuple(EPS_PARAMETERS)}')
    par = dict(EPS_PARAMETERS[key], kappa=.2, rho=.75, theta=0.)
    baseM = par['M']
    par['M'] = smooth_at_least(max(par['p'], baseM*L/E.config('uniform', 2, .75)['L'])) if M is None else int(M)
    K = E.kernel(.2, par['xi'], 0.)
    rc, kc, shell = E.cutoffs(par, L)
    cache = E.MeshCache()
    table_started = time.perf_counter()
    for _ in range(32):
        original, info = cache.get(L, K, kc, par['M'], par['p'])
        if original is not None and kc*(L/par['M'])/np.pi <= .95:
            break
        if M is not None:
            raise ValueError(f'Explicit illegal M={M}: {info}')
        par['M'] = smooth_at_least(par['M']+1)
    else:
        raise RuntimeError('No legal scaled grid within 32 refinements')
    mesh = ProjectedMesh(original, audit_eigenvalues=False)
    t_table = time.perf_counter()-table_started
    op = GridPSDGamma(q, L, K, rc, mesh)
    dense_started = time.perf_counter()
    edges_started = time.perf_counter()
    edges = near_edges(op)
    t_edges = time.perf_counter()-edges_started
    GS = edges_dense(edges, N)
    W = mesh.pair_matrix(q)
    if isinstance(W, tuple):
        W = W[0]
    GL = pair_graph_dense(W)
    GH = GS+GL
    t_dense = time.perf_counter()-dense_started
    images_per_edge = np.bincount(np.unique(op.i*N+op.j, return_inverse=True)[1])
    metadata = dict({k:v for k,v in cfg.items() if k != 'q'}, **par,
        epsilon=float(epsilon), amplitude=1., candidate='Candidate 1 / S1L',
        kernel_theta=0., thermostat_theta=1., thermostat_mass=1.,
        spatial_parameter_source='PR 7 best_by_epsilon.csv Grid PSD; M scaled at fixed h then legal refinement',
        base_M=baseM, h=mesh.h, eta=kc*mesh.h/np.pi, kc=kc, rc=rc,
        kc_shell=shell, n_modes=original.n_modes, excluded_modes=original.n_excluded,
        W2min=original.W2min, legal_grid=bool(1 <= par['p'] <= par['M'] and original.n_excluded == 0 and original.W2min > 1e-10),
        edge_count=len(edges), n_images=op.n_images,
        max_images_per_edge=int(images_per_edge.max(initial=0)),
        full_rank_near_edges=int(sum(np.linalg.matrix_rank(edge[2], tol=1e-12) == 3 for edge in edges)),
        t_table_setup=t_table, t_config_setup=op.t_setup, t_near_edge_build=t_edges,
        t_independent_dense_setup=t_dense, t_total_spatial_setup=time.perf_counter()-started,
        dense_assembly='near image-summed pair tensors; long pair_matrix plus graph conversion',
        physical_spatial_validation='independent_reference must be evaluated; grid legality alone is insufficient',
        projection_active=bool(mesh.diagnostics['negative_blocks']))
    return FrozenSystem(q, L, op, GS, GL, GH, edges, metadata, cfg)


def physical_tail_bound(N, L, R, kappa=.2):
    """Conservative analytic unsplit graph-norm tail, including all images.

    The disjoint lattice cubes imply n(r) <= (N-1) 4*pi/3*(r+a)^3/L^3,
    a=sqrt(3)*L/2.  Integration by parts drops the negative boundary term;
    integrating this count against -g'(r) is a rigorous positive upper bound.
    The graph norm is at most twice the maximum omitted weighted degree.
    """
    a = np.sqrt(3)*L/2
    k = float(kappa)
    e = np.exp(-k*R)
    integral = e*(R*R+2*R/k+2/k**2)
    integral += (3*k*a+1)*e*(R/k+1/k**2)
    integral += (3*k*a*a+3*a)*e/k
    integral += 3*a*a*special.exp1(k*R)+a**3*e/R
    return float(2*(N-1)*4*np.pi/3/L**3*integral)


def direct_unsplit_dense(q, L, R, budget=1 << 18):
    """Literal physical lattice sum, streaming shifts without Ewald or a mesh."""
    N = len(q)
    i, j = np.triu_indices(N, 1)
    d0 = q[i]-q[j]
    d0 -= L*np.round(d0/L)
    tensors = np.zeros((len(i), 3, 3))
    m = int(np.ceil(R/L+.5))
    width = 2*m+1
    step = max(1, int(budget)//len(i))
    images = 0
    for start in range(0, width**3, step):
        flat = np.arange(start, min(start+step, width**3), dtype=np.int64)
        shifts = np.stack((flat//(width*width), (flat//width) % width, flat % width), axis=1)-m
        # A minimum-image pair displacement has norm <= sqrt(3)*L/2.
        shifts = shifts[np.einsum('sa,sa->s', shifts, shifts)*L**2 <= (R+np.sqrt(3)*L/2)**2]
        if not len(shifts):
            continue
        d = d0[None, :, :]+L*shifts[:, None, :]
        r2 = np.einsum('spa,spa->sp', d, d)
        keep = r2 <= R**2
        r = np.sqrt(r2[keep])
        dd = d[keep]
        pair = np.broadcast_to(np.arange(len(i)), keep.shape)[keep]
        w = np.exp(-.2*r)/(r*r*r)
        images += len(r)
        for a in range(3):
            for b in range(a, 3):
                tensors[:, a, b] += np.bincount(pair, w*dd[:, a]*dd[:, b], minlength=len(i))
                if a != b:
                    tensors[:, b, a] = tensors[:, a, b]
    return edges_dense(list(zip(i, j, tensors)), N), images


def independent_reference(system, relative_bound=1e-8):
    """Return an independently summed physical Gamma_D and checked provenance.

    Existing certified caches are read only and require matching geometry.
    New direct sums use an analytic graph-norm tail <= relative_bound; this
    budget is far below the tested 1e-4 spatial tolerance, not an exact sum.
    """
    started = time.perf_counter()
    cfg, N = system.cfg, len(system.q)
    source = None
    if N == 256:
        base = E.config(cfg['family'], cfg['source_seed'], .75)
        if not np.array_equal(base['q'], system.q) or base['L'] != system.L:
            raise RuntimeError('Historical reference geometry mismatch')
        source = E.ref_path(base, .2)
        if not source.exists():
            source = E.C.RAW/'positive_mesh_b2_refs'/source.name
    elif N in (64, 128) and cfg['family'] == 'uniform' and cfg['source_seed'] == 2:
        source = FINAL/'scaling_references'/f'N{N}_uniform_s2.npz'
    if source is not None and source.exists():
        info = json.loads(source.with_suffix('.json').read_text())
        actual_hash = cfg['hash']
        expected_hash = info.get('position_hash', info.get('hash'))
        if expected_hash != actual_hash:
            raise RuntimeError(f'Independent reference hash mismatch: {expected_hash} != {actual_hash}')
        if info.get('kappa') != .2 or info.get('amplitude', info.get('c', 1.)) != 1.:
            raise RuntimeError('Independent reference physics mismatch')
        with np.load(source) as data:
            G = data['G_D'].copy()
            if 'q' in data and not np.array_equal(data['q'], system.q):
                raise RuntimeError('Independent reference saved q mismatch')
        return G, dict(reference_source=str(source), reference_sha256=P.sha(source),
            reference_reused=True, reference_R=info.get('R_D'),
            reference_tail_bound_rel=info['cert_bound_rel'],
            t_reference_load=time.perf_counter()-started,
            reference_original_generation_s=info.get('t_direct_s'),
            reference_kind='independent unsplit direct periodic physical kernel')
    path = HERE/'references'/f'{cfg["name"]}_{cfg["hash"]}.npz'
    if path.exists():
        info = json.loads(path.with_suffix('.json').read_text())
        if info['position_sha256'] != cfg['position_sha256']:
            raise RuntimeError('Local physical reference geometry mismatch')
        with np.load(path) as data:
            G = data['G_D'].copy()
        if info['reference_tail_bound_rel'] <= relative_bound:
            return G, dict(info, reference_reused=True, reference_source=str(path),
                           reference_sha256=P.sha(path), t_reference_load=time.perf_counter()-started)
    # The minimum-image physical graph is PSD and <= the full physical graph.
    # Its norm is consequently a valid lower bound for choosing a radius.
    pi, pj = np.triu_indices(N, 1)
    d = system.q[pi]-system.q[pj]
    d -= system.L*np.round(d/system.L)
    r = np.linalg.norm(d, axis=1)
    minimum_K = (np.exp(-.2*r)/r**3)[:, None, None]*d[:, :, None]*d[:, None, :]
    lower_norm = P.norm2(edges_dense(list(zip(pi, pj, minimum_K)), N))
    R = 40.
    while physical_tail_bound(N, system.L, R)/lower_norm > relative_bound:
        R += 2.5
    G, images = direct_unsplit_dense(system.q, system.L, R)
    norm = P.norm2(G)
    bound = physical_tail_bound(N, system.L, R)/norm
    if bound > relative_bound:
        raise RuntimeError('Independent physical reference tail exceeds requested budget')
    info = dict(position_sha256=cfg['position_sha256'], reference_source=str(path),
        reference_reused=False, reference_R=R, reference_norm2=norm,
        reference_tail_bound_rel=bound, reference_tail_bound_abs=physical_tail_bound(N, system.L, R),
        reference_requested_bound_rel=relative_bound, reference_images=images,
        reference_kind='independent literal unsplit direct periodic physical kernel',
        t_reference_generate=time.perf_counter()-started, kappa=.2, amplitude=1.)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, G_D=G, q=system.q, L=system.L)
    P.save(path.with_suffix('.json'), info)
    return G, dict(info, reference_sha256=P.sha(path))
