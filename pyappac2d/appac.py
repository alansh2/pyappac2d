import numpy as np

import PySide6
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore


def influence(co, af, part):
    # Convert control point to local panel coords
    xt = co[:,0].reshape((-1, 1)) - af['xo'].T
    yt = co[:,1].reshape((-1, 1)) - af['yo'].T

    # Precompute trig expressions as 1-by-m arrays
    costh = np.cos(af['theta'].T)
    sinth = np.sin(af['theta'].T)

    # Find control point coords in CS of all panels
    xp = xt*costh + yt*sinth
    yp = -xt*sinth + yt*costh
    x2 = af['dx'].T*costh + af['dy'].T*sinth

    # Find theta1, theta2, r1, r2
    theta1 = np.arctan2(yp,xp)
    theta2 = np.arctan2(yp,xp-x2)
    dtheta = theta2 - theta1
    dtheta[(np.abs(yp) < 1e-12) & (xp > 0.) & (xp < x2)] = part*np.pi

    ln = 0.5*np.log(((xp-x2)**2 + yp**2)/(xp**2 + yp**2))

    # Compute influence coefficients
    ap = yp*ln + xp*dtheta
    bp = xp*ln + x2 - yp*dtheta
    am = -yp*ln + (x2-xp)*dtheta
    bm = (x2-xp)*ln - x2 + yp*dtheta
    c = 1./(2.*np.pi*x2)

    # Velocity decomposition used in Katz and Plotkin Eq. 11.103
    ua = c*(am*costh - bm*sinth)
    ub = c*(ap*costh - bp*sinth)
    va = c*(am*sinth + bm*costh)
    vb = c*(ap*sinth + bp*costh)

    # Perform staggered sum using integer indexing arrays
    i = np.cumsum(af['m']+1, dtype=np.intp)
    j = np.arange(i[-1], dtype=np.intp)
    k = np.delete(j, i-1, None)
    u = np.zeros((co.shape[0], i[-1]))
    v = np.zeros((co.shape[0], i[-1]))
    u[:,k+1] = ub
    u[:,k] = u[:,k] + ua
    v[:,k+1] = vb
    v[:,k] = v[:,k] + va

    return u, v


def solveWake(foils, Ainv, RHS, CT):
    # Default solver options (TO DO: parse options argument)
    FunctionTolerance = 1e-6
    MaxIterations = 50
    RelaxationFactor =  0.5
    N = 101
    L = 9.

    # Initialize wake dict
    wakes = {'m': np.array([N, N]), 
             'xo': np.zeros((2*N, 1)), 
             'yo': np.zeros((2*N, 1)), 
             'dx': np.zeros((2*N, 1)), 
             'dy': np.zeros((2*N, 1)), 
             'theta': np.zeros((2*N, 1))}
    # Create initial wake shape
    for i in range(2):
        k = i*N + np.arange(N)
        wakes['xo'][k,0] = foils['xo'][i*foils['m'][0],0] + L*(1. - np.cos(np.linspace(0., 0.5*np.pi, N)))
        wakes['yo'][k,0] = foils['yo'][i*foils['m'][0],0] + np.zeros(N)
        wakes['dx'][k,0] = np.concatenate((np.diff(wakes['xo'][k,0]), [1e3]))
    wakes['co'] = np.hstack((wakes['xo']+0.5*wakes['dx'], wakes['yo']))

    # Assign initial wake circulation
    gammaInf = np.sqrt(2.*CT + 1.) - 1.
    wakes['gamma'] = np.repeat(np.array([gammaInf, -gammaInf]), N+1)
    gnew = wakes['gamma'].copy()

    # Make plotting window to monitor wake shape iteration
    app = pg.mkQApp()
    win = pg.GraphicsLayoutWidget(show=True, title="solveWake")
    p = win.addPlot()
    p.setAspectLocked(True)
    k = 0
    for i in range(len(foils['m'])):
        p.plot(foils['xo'][k+np.arange(foils['m'][i]),0], foils['yo'][k+np.arange(foils['m'][i]),0])
        k += foils['m'][i]
    h1 = p.plot(wakes['xo'][:N,0], wakes['yo'][:N,0])
    h2 = p.plot(wakes['xo'][N:,0], wakes['yo'][N:,0])

    # Solve for global circulation and wake shape solutions using Shollenberger's iterative method
    iter = 0
    E = 1.
    while E > FunctionTolerance and iter < MaxIterations:
        iter += 1

        # Update plot with current wake shape
        h1.setData(wakes['xo'][:N,0], wakes['yo'][:N,0])
        h2.setData(wakes['xo'][N:,0], wakes['yo'][N:,0])
        app.processEvents()

        # Solve for airfoil circulation
        U, V = influence(foils['co'], wakes, 1.)
        A = -U*np.sin(foils['theta']) + V*np.cos(foils['theta'])
        gamma = np.matmul(Ainv, RHS - np.concatenate((np.matmul(A, wakes['gamma']), 
                                                      [-wakes['gamma'][0], -wakes['gamma'][N+1]], 
                                                      np.zeros(len(foils['m'])-2))))
        
        # Calculate inner-outer-averaged velocities on the wake boundaries
        U, V = influence(wakes['co'], foils, 1.)
        u = np.matmul(U, gamma) + 1.
        v = np.matmul(V, gamma)
        U, V = influence(wakes['co'], wakes, 0.)
        u += np.matmul(U, wakes['gamma'])
        v += np.matmul(V, wakes['gamma'])

        # Update wake shape
        wakes['dy'][:,0] = v / u * wakes['dx'][:,0]
        wakes['dy'][N-1,0] = 0.
        wakes['dy'][-1,0] = 0.
        wakes['yo'][1:N,0] = wakes['yo'][0,0] + np.cumsum(wakes['dy'][:N-1,0])
        wakes['yo'][N+1:,0] = wakes['yo'][N,0] + np.cumsum(wakes['dy'][N:-1,0])
        wakes['theta'] = np.arctan2(wakes['dy'], wakes['dx'])
        wakes['co'][:,1] = wakes['yo'][:,0] + 0.5*wakes['dy'][:,0]

        # Update wake circulation
        g = CT / np.sqrt(u*u + v*v)
        g[N:] *= -1.
        gnew[1:N] = g[:N-1] + np.diff(g[:N]) * wakes['dx'][:N-1,0] / (wakes['dx'][:N-1,0] + wakes['dx'][1:N,0])
        gnew[N+2:-1] = g[N:-1] + np.diff(g[N:]) * wakes['dx'][N:-1,0] / (wakes['dx'][N:-1,0] + wakes['dx'][N+1:,0])
        gnew[0] = 2.*g[0] - gnew[1]
        gnew[N+1] = 2.*g[N] - gnew[N+2]

        # Calculate residual between current and previous iteration
        E = np.sum(np.abs(gnew - wakes['gamma'])) / (2.*N*gammaInf + 4e-16)

        # Adjust wake circulation by a relaxation factor
        wakes['gamma'] = RelaxationFactor*gnew + (1. - RelaxationFactor)*wakes['gamma']

    return wakes, gamma


def panel2d(surfaces, alphaDeg, CT=0.):
    nSurfs = len(surfaces)

    # Right CW rotation matrix
    alphaRad = alphaDeg * np.pi/180.
    R = np.array([[np.cos(alphaRad), -np.sin(alphaRad)], [np.sin(alphaRad), np.cos(alphaRad)]])

    # Determine the number of panels for each surface
    foils = {'m': np.zeros(nSurfs, dtype=np.intp)}
    for i in range(nSurfs):
        foils['m'][i] = surfaces[i].shape[0] - 1
    M = np.sum(foils['m']) # total number of panels

    # Build unified dict for all lifting surfaces
    foils.update({'xo': np.zeros((M,1)), 'yo': np.zeros((M,1)), 'dx': np.zeros((M,1)), 'dy': np.zeros((M,1))})
    k = 0
    for i in range(nSurfs):
        coords = np.matmul(surfaces[i], R) # rotate coordinates by AoA
        if np.linalg.det(surfaces[i][[1, -2],:] - surfaces[i][0]) > 0.:
            coords = np.flipud(coords) # method requires CW defined coordinates
        foils['xo'][k:foils['m'][i]+k,0] = coords[:-1,0]
        foils['yo'][k:foils['m'][i]+k,0] = coords[:-1,1]
        foils['dx'][k:foils['m'][i]+k,0] = coords[1:,0] - coords[:-1,0]
        foils['dy'][k:foils['m'][i]+k,0] = coords[1:,1] - coords[:-1,1]
        k += foils['m'][i]
    foils['theta'] = np.arctan2(foils['dy'], foils['dx'])
    foils['co'] = np.hstack((foils['xo']+0.5*foils['dx'], foils['yo']+0.5*foils['dy']))

    # Create aerodynamic influence coefficient matrix
    A = np.zeros((M+nSurfs, M+nSurfs))
    k = 0
    for i in range(nSurfs):
        A[M+i,k] = 1.
        A[M+i,k+foils['m'][i]] = 1.
        k += foils['m'][i] + 1
    U, V = influence(foils['co'], foils, 1.)
    A[:M,:] = -U*np.sin(foils['theta']) + V*np.cos(foils['theta'])
    B = U*np.cos(foils['theta']) + V*np.sin(foils['theta'])
    RHS = np.concatenate((np.sin(foils['theta'][:,0]), np.zeros(nSurfs)))
    
    # Run wake-coupled aerodynamic solver
    wakes, foils['gamma'] = solveWake(foils, np.linalg.inv(A), RHS, CT)
    U, V = influence(foils['co'], wakes, 1.)
    D = U*np.cos(foils['theta']) + V*np.sin(foils['theta'])

    Qtan = np.matmul(B, foils['gamma']) + np.matmul(D, wakes['gamma']) + np.cos(foils['theta'][:,0])
    Cp_arr = 1. - Qtan**2
    xc_arr = np.matmul(foils['co'], R[0,:])

    # Package outputs
    Cp = []
    xc = []
    k = 0
    for i in range(nSurfs):
        Cp.append(Cp_arr[k:foils['m'][i]+k])
        xc.append(xc_arr[k:foils['m'][i]+k])
        k += foils['m'][i]

    return Cp, xc, foils, wakes