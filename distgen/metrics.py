#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 21:18:56 2022

@author: colwyngulliford
"""

import numpy as np

from .tools import linspace
from .tools import trapz


#--------------------------------------------------------------
# Comparing distributions / shaping metrics
#--------------------------------------------------------------

def mean_and_sigma(x, rho):
    x0 = np.trapz(rho*x, x)
    x2 = np.trapz(rho*(x-x0)**2, x)
    
    return x0, np.sqrt(x2)


# Distribution comparison functions:

def resample_pq(xp, P, xq, Q):
    
    # Get the new grid:
    xmin = min([xp.min(), xq.min()])
    xmax = max([xp.max(), xq.max()])
    
    dxp, dxq = np.mean(np.diff(xp)), np.mean(np.diff(xq))
    dx =0.5*(dxp + dxq)
    
    x = linspace(xmin, xmax, int(np.floor( (xmax-xmin)/dx )))
    
    # Interpolate to grid
    Pi = np.interp(x, xp, P, left=0, right=0)
    Qi = np.interp(x, xq, Q, left=0, right=0)
    
    # Renormalize:
    Pi, Qi = Pi/trapz(Pi, x), Qi/trapz(Qi, x)
    
    return (x, Pi, Qi)

def kullback_liebler_div(xp, P, xq, Q, adjusted=False):
    
    # Check that input P, Q are PDFs (up to normalization):
    if(np.sum(P)==0.0): raise ValueError('PDF array P sums to zero!')
        
    if(np.sum(Q)==0.0): raise ValueError('PDF array Q sums to zero!')
        
    if(len(P[P<0])>0): raise ValueError('P array has negative values, and is not a true PDF.')
        
    if(len(Q[Q<0])>0): raise ValueError('Q array has negative values, and is not a true PDF.')
    
    xi, P, Q = resample_pq(xp, P, xq, Q)  # Interpolates to same grid, and renormalizes
         
    if(adjusted):
        
        q0 = (Q==0)
        P0 = P[q0]
        Q[q0] = P0*np.exp(-P0/P.max()**2)
        
    p_and_q_nonzero = (P>0) & (Q>0)
    
    P0 = P[p_and_q_nonzero]
    Q0 = Q[p_and_q_nonzero]
    x0 = xi[p_and_q_nonzero]
    
    return np.trapz(P0*( np.log(P0/Q0) ), x0 )
            

def res2(xp, P, xq, Q):
    xi, P, Q = resample_pq(xp, P, xq, Q)  # Interpolates to same grid, and renormalizes
    return np.trapz((P-Q)**2, xi)