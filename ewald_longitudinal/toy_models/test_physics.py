"""Physical identity tests, independent finite differences and statistics checks."""
import unittest
import numpy as np
import prl_toy_models as model


class PhysicsTests(unittest.TestCase):
    def test_conservative_force_is_negative_gradient(self):
        q = np.array([[.1,.2,.3], [1.32,.41,.37], [3.72,1.1,.8]])
        for kind in ('double_well', 'lj'):
            force, _, _ = model.conservative_force(q, 6., kind)
            fd = np.zeros_like(q)
            for i in range(3):
                for c in range(3):
                    plus, minus = q.copy(), q.copy()
                    plus[i,c] += 1e-6; minus[i,c] -= 1e-6
                    ep = model.conservative_force(plus, 6., kind)[1]
                    em = model.conservative_force(minus, 6., kind)[1]
                    fd[i,c] = -(ep-em)/(2e-6)
            np.testing.assert_allclose(force, fd, rtol=2e-6, atol=2e-7)
            np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-13)

    def test_double_well_has_two_minima_and_smooth_cutoff(self):
        r = np.linspace(.9, 1.95, 5000)
        _, d = model.pair_potential(r, 'double_well')
        minima = r[:-1][(d[:-1] < 0) & (d[1:] >= 0)]
        self.assertEqual(len(minima), 2)
        np.testing.assert_allclose(minima, [1.12,1.60], atol=.015)
        for kind in ('double_well','lj'):
            u, du = model.pair_potential(np.array([2.5,2.6]), kind)
            np.testing.assert_allclose(u, 0, atol=1e-14)
            np.testing.assert_allclose(du, 0, atol=1e-14)

    def test_graph_dissipation_nullspace_and_fdt(self):
        rng = np.random.default_rng(4)
        q = rng.uniform(0,6,(9,3))
        x = rng.normal(size=(9,3))
        n = len(q)
        translations = np.kron(np.ones((n,1)), np.eye(3))
        proj = np.eye(3*n) - translations@translations.T/n
        for kernel in ('A','B'):
            G = model.friction_matrix(q,6,kernel=kernel)
            np.testing.assert_allclose(G, G.T, atol=1e-14)
            np.testing.assert_allclose(G@translations, 0, atol=1e-13)
            self.assertGreater(np.linalg.eigvalsh(G).min(), -1e-12)
            i,j,dr,r = model.pair_geometry(q,6)
            pair_energy = np.sum(model.radial_friction(r,kernel=kernel)*
                                 (np.sum((x[i]-x[j])*dr,axis=1)/r)**2)
            self.assertAlmostEqual(x.ravel()@G@x.ravel(), pair_energy, places=11)
            vec, damp, noise = model.thermostat_factors(G,.02,1.4,.7)
            R = (vec*damp)@vec.T
            S = (vec*noise)@vec.T
            # Stationary covariance in the fixed-total-momentum subspace.
            expected = 1.4*.7*proj
            actual = R@expected@R.T + proj@S@S.T@proj
            np.testing.assert_allclose(actual, expected, atol=2e-13)
            p = rng.normal(size=(n,3))+[.2,.3,.4]
            out = model.thermostat_step(p,G,.02,1.4,.7,rng)
            np.testing.assert_allclose(out.sum(axis=0),p.sum(axis=0),atol=1e-13)

    def test_static_correlations_and_periodic_wrapping(self):
        rng = np.random.default_rng(6)
        q = rng.uniform(0,8,(12,3)); v = rng.normal(size=(12,3))
        v -= v.mean(axis=0)
        Q = np.repeat(q[None],12,axis=0); V=np.repeat(v[None],12,axis=0)
        t = np.arange(12)*.1
        obs = model.compute_observables(Q,V,t,8,r_bins=12,vccf_edges=[0,4,8])
        np.testing.assert_allclose(obs['vacf'],1,atol=1e-14)
        np.testing.assert_allclose(obs['hydro_L'],1,atol=1e-14)
        np.testing.assert_allclose(obs['hydro_T'],1,atol=1e-14)
        np.testing.assert_allclose(obs['van_hove_distinct'],
                                   np.broadcast_to(obs['rdf'],obs['van_hove_distinct'].shape))
        pairs=~np.eye(len(q),dtype=bool)
        dist=np.linalg.norm(model.minimum_image(q[:,None]-q[None,:],8),axis=-1)
        direct=(v@v.T)[pairs & (dist<4)].mean()
        self.assertAlmostEqual(obs['vccf'][0,0],direct,places=13)
        # Integer box shifts may vary independently by particle and saved frame.
        shifted=Q+8*rng.integers(-3,4,Q.shape)
        other=model.compute_observables(shifted,V,t,8,r_bins=12,vccf_edges=[0,4,8])
        for key in ('vacf','vccf','hydro_L','hydro_T','van_hove_distinct'):
            np.testing.assert_allclose(obs[key],other[key],atol=1e-12)

    def test_uniform_boost_and_distinct_pair_normalization(self):
        rng=np.random.default_rng(9)
        Q=rng.uniform(0,8,(40,50,3)); V=rng.normal(size=Q.shape)
        t=np.arange(40)*.1
        base=model.compute_observables(Q,V,t,8,r_bins=8,origins=20,remove_com=True)
        boost=np.array([.3,-.4,.2])
        moved=model.compute_observables(Q+t[:,None,None]*boost,V+boost,t,8,
                                        r_bins=8,origins=20,remove_com=True)
        for key in ('vacf','vccf','hydro_L','hydro_T','van_hove_distinct'):
            np.testing.assert_allclose(base[key],moved[key],atol=2e-12)
        # Ideal-gas independent uniform positions have distinct g(r)=1.
        self.assertLess(np.abs(base['rdf'][2:]-1).max(),.10)
        # Explicit self exclusion: two separated fixed particles have no r=0 mass.
        Q2=np.repeat(np.array([[[0.,0.,0.],[2.,0.,0.]]]),5,axis=0)
        V2=np.repeat(np.array([[[1.,0.,0.],[-1.,0.,0.]]]),5,axis=0)
        small=model.compute_observables(Q2,V2,np.arange(5),8,r_bins=8)
        self.assertEqual(small['rdf'][0],0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
