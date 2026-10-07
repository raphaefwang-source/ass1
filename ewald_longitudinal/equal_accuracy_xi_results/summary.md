# Equal-accuracy xi optimisation: summary

- tol 1e-05, system A: xi=0.5: 7.59 ms (p=4, eta=0.7, M=10), xi=0.7: 3.81 ms (p=6, eta=0.8, M=12), xi=1: 4.24 ms (p=5, eta=0.7, M=20), xi=1.4: 3.32 ms (p=6, eta=0.8, M=25), xi=2: 8.51 ms (p=6, eta=0.8, M=36)  ->  cheapest tested xi = 1.4
- tol 1e-05, system B: xi=0.5: 5.68 ms (p=6, eta=0.8, M=15), xi=0.7: 2.39 ms (p=6, eta=0.8, M=20), xi=1: 5.24 ms (p=6, eta=0.8, M=30), xi=1.4: 11.18 ms (p=7, eta=0.8, M=40), xi=2: 34.49 ms (p=6, eta=0.8, M=60)  ->  cheapest tested xi = 0.7
- tol 1e-05, system C: xi=0.5: 163.93 ms (p=5, eta=0.8, M=45), xi=0.7: 128.63 ms (p=5, eta=0.8, M=64), xi=1: 269.84 ms (p=6, eta=0.8, M=90), xi=1.4: 662.52 ms (p=7, eta=0.8, M=120), xi=2: 1999.17 ms (p=6, eta=0.8, M=180)  ->  cheapest tested xi = 0.7
- tol 1e-07, system A: xi=0.5: 8.52 ms (p=5, eta=0.5, M=18), xi=0.7: 4.66 ms (p=7, eta=0.7, M=16), xi=1: 5.83 ms (p=8, eta=0.8, M=20), xi=1.4: 9.61 ms (p=8, eta=0.8, M=30), xi=2: 20.47 ms (p=8, eta=0.7, M=48)  ->  cheapest tested xi = 0.7
- tol 1e-07, system B: xi=0.5: 7.02 ms (p=6, eta=0.7, M=20), xi=0.7: 8.36 ms (p=7, eta=0.7, M=27), xi=1: 9.57 ms (p=8, eta=0.8, M=36), xi=1.4: 23.27 ms (p=8, eta=0.8, M=50), xi=2: 119.07 ms (p=8, eta=0.7, M=80)  ->  cheapest tested xi = 0.5
- tol 1e-07, system C: xi=0.5: 332.03 ms (p=6, eta=0.7, M=60), xi=0.7: 311.12 ms (p=7, eta=0.7, M=80), xi=1: 532.30 ms (p=8, eta=0.8, M=100), xi=1.4: 1366.26 ms (p=8, eta=0.8, M=150), xi=2: 8145.20 ms (p=8, eta=0.7, M=240)  ->  cheapest tested xi = 0.7
- tol 1e-09, system A: xi=0.5: 10.19 ms (p=7, eta=0.5, M=18), xi=0.7: 11.68 ms (p=8, eta=0.6, M=24), xi=1: 11.75 ms (p=8, eta=0.5, M=36), xi=1.4: 25.44 ms (p=8, eta=0.5, M=50), xi=2: 82.59 ms (p=8, eta=0.5, M=75)  ->  cheapest tested xi = 0.5
- tol 1e-09, system B: xi=0.5: 9.58 ms (p=8, eta=0.7, M=24), xi=0.7: 14.13 ms (p=8, eta=0.6, M=36), xi=1: 35.01 ms (p=8, eta=0.5, M=60), xi=1.4: 165.85 ms (p=8, eta=0.5, M=90), xi=2: 533.86 ms (p=8, eta=0.5, M=125)  ->  cheapest tested xi = 0.5
- tol 1e-09, system C: xi=0.5: 591.99 ms (p=8, eta=0.7, M=64), xi=0.7: 738.95 ms (p=8, eta=0.6, M=108), xi=1: 2791.26 ms (p=8, eta=0.5, M=180), xi=1.4: 12704.24 ms (p=8, eta=0.5, M=250)  ->  cheapest tested xi = 0.5
- not feasible / not timed: tol 1e-09 xi 2 system C
- all accepted runs PSD observed: True; certified: 44/44
- dense evaluations: 2030; wall time 767s
