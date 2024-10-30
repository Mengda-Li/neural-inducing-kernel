## IMPORT STATEMENTS ##
from IPython.display import display, Image
import math, time, os, sys, pickle, collections, warnings
warnings.filterwarnings(action='once')
import traceback
import datetime
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats as st
from scipy.signal import find_peaks, convolve
from scipy.io import loadmat
from scipy.ndimage.filters import generic_filter as gf
from scipy.linalg import circulant, dft
from scipy.interpolate import splrep, BSpline, interp1d
from scipy.integrate import dblquad
import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 300

## MODEL PARAMETERS ##
 # earth model parameters
R=6371
h=687

## IMPORT MIGUEL'S MODEL ##
mean_vec = np.array([256.03185595, 255.82869165, 255.28088473, 254.48303442, 253.39758773,
			252.01499891, 250.31602682, 248.2823423,  245.9041934,  243.18997538,
			240.14883441, 236.81780999])

# Extrapolate
mean_vec = [2*mean_vec[0] - mean_vec[1]] + list(mean_vec) + [2*mean_vec[-1]-mean_vec[-2], 3*mean_vec[-1]-2*mean_vec[-2]]

pcs_ = np.array([[-2.78790888e-01, -2.79373243e-01, -2.80891970e-01,
		-2.83020383e-01, -2.85707088e-01, -2.88780920e-01,
		-2.92018473e-01, -2.95059047e-01, -2.97278346e-01,
		-2.97903995e-01, -2.95627132e-01, -2.88706080e-01],
		[ 2.64545519e-01,  2.59725261e-01,  2.46366317e-01,
		 2.25855353e-01,  1.95634827e-01,  1.53341227e-01,
		 9.52053509e-02,  1.60146706e-02, -9.07495628e-02,
		-2.32540911e-01, -4.19942747e-01, -6.64139849e-01],
	   [-3.01146113e-01, -2.79033775e-01, -2.20939076e-01,
		-1.39604150e-01, -3.58748014e-02,  8.36126866e-02,
		 2.08817040e-01,  3.22207514e-01,  3.91354793e-01,
		 3.55477518e-01,  1.12909581e-01, -5.61405418e-01],
	   [-2.03675980e-01, -1.75944654e-01, -1.11205758e-01,
		-2.34240463e-02,  8.18707702e-02,  1.85985873e-01,
		 2.68366718e-01,  2.91310961e-01,  1.48385971e-01,
		-1.52184893e-01, -7.15647730e-01,  3.98920438e-01],
	   [-1.22657189e-01, -1.04579976e-01, -5.82434121e-02,
		 1.10886558e-03,  7.05231596e-02,  1.46314065e-01,
		 1.95533586e-01,  1.83558942e-01,  9.24349009e-02,
		-8.17750033e-01,  4.40703260e-01, -2.89385219e-02],
	   [-9.57170916e-02, -7.89987174e-02, -4.24938118e-02,
		 9.71812410e-04,  6.47905692e-02,  1.37973216e-01,
		 1.25747501e-01,  4.62310796e-01, -8.31171079e-01,
		 1.60135593e-01,  1.14672184e-01, -1.93432116e-02],
	   [-1.24526058e-01, -1.02125721e-01, -5.62698721e-02,
		-1.35932283e-04,  1.11385408e-01,  1.56224525e-01,
		 6.84640479e-01, -6.50622487e-01, -1.69576311e-01,
		 1.01386384e-01,  5.45148092e-02, -5.92238143e-03],
	   [-1.12049727e-01, -8.58791703e-02, -4.21184298e-02,
		 5.50053814e-04, -4.19594815e-02,  8.62864374e-01,
		-4.32466289e-01, -2.06492426e-01, -5.84102518e-03,
		 4.28086463e-02,  2.15505131e-02, -1.58146273e-03],
	   [-1.84825276e-01, -1.29984687e-01, -1.93369432e-02,
		-1.31265818e-01,  9.13266349e-01, -1.46105050e-01,
		-2.58980400e-01, -8.82520826e-02,  9.46661368e-04,
		 2.67765343e-02,  1.88908326e-02, -1.68421299e-03],
	   [-2.61692761e-01, -1.62160885e-01, -2.10294591e-01,
		 9.09866646e-01, -8.35011315e-03, -1.31274739e-01,
		-1.12041746e-01, -5.17335868e-02, -5.07684576e-03,
		 2.01598511e-02,  1.31860223e-02, -1.02642854e-03],
	   [ 4.49801918e-01,  2.18993608e-01, -8.54902863e-01,
		-1.56891857e-02,  1.19547511e-01,  5.95095510e-02,
		 2.36751444e-02,  1.20030184e-02, -1.51124398e-03,
		-7.65596403e-03, -4.04007740e-03,  4.24183396e-04],
	   [ 6.02399158e-01, -7.87111159e-01,  1.16737423e-01,
		 5.92802075e-02,  1.87950443e-02,  1.91037762e-03,
		-6.75975605e-03, -4.60044404e-03, -2.08822549e-03,
		 1.05258168e-03, -1.46035602e-04,  5.07419337e-04]])

#Extrapolate
pcs = [ [2*pc[0] - pc[1]] + list(pc) + [2*pc[-1]-pc[-2], 3*pc[-1]-2*pc[-2]] for pc in pcs_ ]

var = np.array([7.29414071e+03, 2.34715558e+02, 5.16586712e+00, 2.19147960e-01,
	   8.68484209e-02, 5.83654258e-02, 3.51077566e-02, 1.71269312e-02,
	   8.12774193e-03, 4.78727353e-03, 1.96296841e-03, 6.02597099e-04])

angles = np.array([-2.5, 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5])

## HELPER FUNCTIONS ##
def clip(x, lower=0, upper=1):
	x[x>upper] = upper
	x[x<lower] = lower
	return x

def is_visible(xi, ya, R=R, h=h):
	t = horizon_xi_val(ya, R=R, h=h)
	return np.abs(xi) <= t

def horizon_xi_val(ya, R=R, h=h):
	a = R/((R+h)*np.cos(ya))
	num = (R+h)**2*np.cos(ya)**2-R**2
	den = (R+h)**2 - R**2 # rho^2 at horizon
	b = np.sqrt(num/den)
	return a*b

def get_eta(xi, ya, R=R, h=h):
	discriminant = (R+h)**2*np.cos(ya)**2*xi**4 - (R**2 + (R+h)**2)*xi**2 + R**2
	rho = np.sqrt( R**2 + (R+h)**2 - 2*(R+h)**2*np.cos(ya)**2*xi**2-2*(R+h)*np.cos(ya)*np.sqrt(discriminant) )
	xa = np.arcsin(-rho*xi/R)
	eta = R*np.sin(ya)*np.cos(xa)/rho
	return eta


def dblquad_montecarlo_jacobian(u, v, xis_ctr, yas_ctr, dxi, dya, n=100, R=R, h=h):
	ngrid = len(xis_ctr)
	xis_rel = np.random.uniform(low=-1*np.ones((ngrid,n))*dxi/2.0, high=np.ones((ngrid,n))*dxi/2.0, size=(ngrid,n))
	yas_rel = np.random.uniform(low=-1*np.ones((ngrid,n))*dya/2.0, high=np.ones((ngrid,n))*dya/2.0, size=(ngrid,n))
	xis_sample = np.hstack([np.zeros((ngrid,1)), xis_rel]) + np.tile(xis_ctr.reshape((ngrid,1)), (1,n))
	yas_sample = np.hstack([np.zeros((ngrid,1)), yas_rel]) + np.tile(yas_ctr.reshape((ngrid,1)), (1,n))
	within_horizon = np.abs(xis_sample) <= R/((R+h)*np.cos(yas_sample))*np.sqrt(((R+h)**2*np.cos(yas_sample)**2-R**2)/((R+h)**2-R**2))
	xi = xis_sample[within_horizon]
	ya = yas_sample[within_horizon]
	rhosq = R**2 + (R+h)**2 - 2*(R+h)**2*np.cos(ya)**2*xi**2-2*(R+h)*np.cos(ya)*np.sqrt((R+h)**2*np.cos(ya)**2*xi**4-(R**2+(R+h)**2)*xi**2+R**2)
	rho = np.sqrt(rhosq)
	xa = np.arcsin(-rho*xi/R)
	eta = R*np.sin(ya)*np.cos(xa)/rho
	theta = np.pi-np.arcsin(np.sqrt(xi**2+eta**2))
	J = J_xi_ya(rho, eta, theta, xa, ya, R=R, h=h, ablation=False)
	res = np.zeros(yas_sample.shape, dtype=np.complex128)
	res[within_horizon] = np.exp(-2*np.pi*1j*(u*xi + v*eta))*J
	res[np.logical_not(within_horizon)] = np.nan
	return np.nanmean(res, axis=1)*dxi*dya


def dblquad_montecarlo(u, v, xis_ctr, yas_ctr, dxi, dya, n=100, R=R, h=h):
	ngrid = len(xis_ctr)
	xis_rel = np.random.uniform(low=-1*np.ones((ngrid,n-1))*dxi/2.0, high=np.ones((ngrid,n-1))*dxi/2.0, size=(ngrid,n-1))
	yas_rel = np.random.uniform(low=-1*np.ones((ngrid,n-1))*dya/2.0, high=np.ones((ngrid,n-1))*dya/2.0, size=(ngrid,n-1))
	xis_sample = np.hstack([np.zeros((ngrid,1)), xis_rel]) + np.tile(xis_ctr.reshape((ngrid,1)), (1,n))
	yas_sample = np.hstack([np.zeros((ngrid,1)), yas_rel]) + np.tile(yas_ctr.reshape((ngrid,1)), (1,n))
	within_horizon = np.abs(xis_sample) <= R/((R+h)*np.cos(yas_sample))*np.sqrt(((R+h)**2*np.cos(yas_sample)**2-R**2)/((R+h)**2-R**2))
	xi = xis_sample[within_horizon]
	ya = yas_sample[within_horizon]
	rhosq = R**2 + (R+h)**2 - 2*(R+h)**2*np.cos(ya)**2*xi**2-2*(R+h)*np.cos(ya)*np.sqrt((R+h)**2*np.cos(ya)**2*xi**4-(R**2+(R+h)**2)*xi**2+R**2)
	eta = np.sin(ya)*np.sqrt(R**2/rhosq - xi**2) ## TODO : CORRECT? I THINK IT IS sqrt(sin^2ya*(R^2/rho^2-xi^2) - xi^2)
	res = np.zeros(yas_sample.shape, dtype=np.complex128)
	res[within_horizon] = np.exp(-2*np.pi*1j*(u*xi + v*eta))
	res[np.logical_not(within_horizon)] = np.nan
	return np.nanmean(res, axis=1)


def integrate_box(xictr, yactr, u, v, dxi=0.001, dya=2*np.pi/(5400*5), jacobian=False, epsabs=1e-2, epsrel=1e-2):
	gfunc = lambda y: np.maximum(-1*horizon_xi_val(y), xictr-dxi)
	hfunc = lambda y: np.minimum(horizon_xi_val(y), xictr+dxi)

	if jacobian:
		pass
	else:
		freal = lambda xi, ya: np.cos(-2*np.pi*(u*xi+v*get_eta(xi,ya)))  #*np.exp(1j*omega*N/T)*np.exp(-2*np.pi*1j*Ti*omega/T)
		fcplx = lambda xi, ya: np.sin(-2*np.pi*(u*xi+v*get_eta(xi,ya)))
		res = dblquad(freal, yactr-dya, yactr+dya, gfunc, hfunc, epsabs=epsabs, epsrel=epsrel)[0] + 1j*dblquad(fcplx, yactr-dya, yactr+dya, gfunc, hfunc, epsabs=epsabs, epsrel=epsrel)[0]
		return res

def cexp_int_weights(xi_sample, ya_sample, u, v, dxi=0.001, dya=2*np.pi/(5400*5), epsabs=1e-2, epsrel=1e-2):
	int_fn = np.vectorize(lambda x, y: integrate_box(x,y,u,v,dxi,dya,jacobian=False,epsabs=epsabs,epsrel=epsrel))
	weights = int_fn(xi_sample, ya_sample)
	return weights

def get_cexp_term(xi_sample, ya_sample, baselines, dxi=0.001, dya=2*np.pi/(5400*5), epsabs=1e-2, epsrel=1e-2):
	f = np.vectorize(lambda u, v: cexp_int_weights(xi_sample, ya_sample, u, v, dxi, dya, epsabs, epsrel))
	return f(baselines[:,0], baselines[:,1])


def compute_t_theta(img,angle,mean_vec=mean_vec,var=var,pcs=pcs,nparam=3):
	temps = mean_vec[angle] + sum(img[:,:,i]*np.sqrt(var[i])*np.array(pcs[i][angle]) for i in range(nparam))
	return temps


def convert_from_m_n(m,n,M=None,N=None,R=R,h=h,ysat=R*np.pi):
	if M is None:
		M = m.shape[0]
	if N is None:
		N = n.shape[1]

	x = (np.pi*R/M)*((M-1)/2.0 - m) # signed geodesic distance from subsatellite point across-track
	y = (n/N)*2*np.pi*R*np.cos(x/R) - ysat*np.cos(x/R) # signed geodesic distance from subsatellite point along-track


	xang = x/R # signed angular geodesic distance from subsatellite point across-track (longitudinal distance from SSP)
	yang = y/(R*np.cos(xang)) # signed angular geodesic distance from subsatellite point (latitudinal distance from SSP)

	# Using the Law of Cosines, find the distance from the observed point at (fractional) pixel index (m,n) to the satellite
	# By Napier's Law, the cosine of the angle opposite rho is the product of cos(xang) and cos(yang)
	rho = np.sqrt( R**2 + (R+h)**2 -
				 2*R*(R+h)*np.multiply(np.cos(xang), np.cos(yang)) )
	visible_mask = (rho <= np.sqrt(2*R*h+h**2)) # within the horizon
	nvisible = np.sum(visible_mask)

	if ysat/R < np.arccos(R/(R+h)): # left wraparound
		res = img_to_array(yang, visible_mask, kind='C')
		yang[visible_mask] = np.where(res > 2*np.pi-np.arccos(R/(R+h)), res - 2*np.pi, res)
		res = img_to_array(y, visible_mask, kind='C')
		resx = img_to_array(x, visible_mask, kind='C')
		y[visible_mask] = np.where(res > 2*np.pi*R-R*np.cos(resx)*np.arccos(R/(R+h)),
				res - 2*np.pi*R*np.cos(resx), res)
	if ysat/R > 2*np.pi-np.arccos(R/(R+h)): # right wraparound
		res = img_to_array(yang, visible_mask, kind='C')
		yang[visible_mask] = np.where(res <= 0, res + 2*np.pi, res)
		res = img_to_array(y, visible_mask, kind='C')
		resx = img_to_array(x, visible_mask, kind='C')
		y[visible_mask] = np.where(res < 0,
				res + 2*np.pi*R*np.cos(resx), res)

	xang_visible = img_to_array(xang, visible_mask)
	yang_visible = img_to_array(yang, visible_mask)
	assert np.allclose(img_to_array(rho, visible_mask)**2,
						R**2 + (R+h)**2 - 2*R*(R+h)*np.cos(xang_visible)*np.cos(yang_visible))
	rho_visible = img_to_array(rho, visible_mask)
	
	assert np.allclose(yang_visible, yang[visible_mask])
	# TODO why does this line give warning when plotting antpattenr
	phi_prime = np.where( yang_visible != 0,
			np.arctan(np.abs(np.divide(np.sin(xang_visible), np.sin(yang_visible)*np.cos(xang_visible), where=yang_visible != 0))),
			np.pi/2)


	quad_1_mask = np.where(np.logical_and(xang_visible < 0, yang_visible >= 0))
	quad_2_mask = np.where(np.logical_and(xang_visible >= 0, yang_visible > 0))
	quad_3_mask = np.where(np.logical_and(xang_visible > 0, yang_visible <= 0))
	quad_4_mask = np.where(np.logical_and(xang_visible <= 0, yang_visible < 0))

	phi = np.zeros(phi_prime.shape)
	phi[quad_1_mask] = np.pi/2.0 - phi_prime[quad_1_mask]
	phi[quad_2_mask] = np.pi/2.0 + phi_prime[quad_2_mask]
	phi[quad_3_mask] = 3*np.pi/2.0 - phi_prime[quad_3_mask]
	phi[quad_4_mask] = 3*np.pi/2.0 + phi_prime[quad_4_mask]

	phi[phi >= 2*np.pi] = phi[phi >= 2*np.pi] - 2*np.pi

	theta1 = np.pi - np.arccos( ( R+h-R*np.cos(xang_visible)*np.cos(yang_visible) ) / rho_visible )
	theta = np.pi - np.arccos( (rho_visible**2 + 2*R*h + h**2)/ (2*(R+h)*rho_visible) )
	assert(np.allclose(theta, theta1, equal_nan=True))
	
	xi = np.sin(theta) * np.cos(phi)
	eta = np.sin(theta) * np.sin(phi)
	
	assert np.all(xi**2 + eta**2 <= 1+1e-14)
	
	return xi, eta, visible_mask, rho_visible, phi, theta, x, y, xang, yang


def convert_from_phi_theta(phi,theta,M,N,R=R,h=h,ysat=R*np.pi):
	visible_mask = theta >= np.pi - np.arcsin(R/(R+h))

	phis_visible = phi[visible_mask]
	thetas_visible = theta[visible_mask]

	# valid for all directions
	xi = np.multiply(np.sin(theta), np.cos(phi))
	eta = np.multiply(np.sin(theta), np.sin(phi))

	xis_visible = xi[visible_mask]
	etas_visible = eta[visible_mask]

	# valid for directions that intersect with Earth
	rhos_visible = -(R+h)*np.cos(thetas_visible) - np.sqrt( (R**2 - (R+h)**2 * np.sin(thetas_visible)**2) )

	xsatcartesians_visible = rhos_visible * xis_visible
	ysatcartesians_visible = rhos_visible * etas_visible

	xangs_visible = np.arcsin( -xsatcartesians_visible/R )
	yangs_visible = np.arcsin( np.divide(ysatcartesians_visible, R*np.cos(xangs_visible)) ) # valid for directions that intersect with Earth
	
	xs_visible = R*xangs_visible
	ys_visible = R*yangs_visible*np.cos(xangs_visible)

	ns_visible = (ys_visible+ysat*np.cos(xangs_visible))*( M/(np.pi*R*np.cos(xangs_visible)) )
	ms_visible = (M-1)/2.0 - (M/(np.pi*R))*xs_visible
	
	return ms_visible, ns_visible, visible_mask, xis_visible, etas_visible, xs_visible, ys_visible, xangs_visible, yangs_visible


def xi_eta_to_phi_theta(xis, etas):
	xis, etas = type_check(xis, etas)

	sintheta = np.sqrt(xis**2 + etas**2)
	thetas = np.pi - np.arcsin(sintheta)

	#phi_prime = np.arctan( np.divide(-xis, etas) )
	phi_prime = np.zeros(xis.shape)
	phi_prime[np.where(xis!=0)] = np.arctan( np.divide(etas, xis, where=(xis!=0))[xis!=0] )

	quad_12_mask = np.where(etas>=0)
	quad_34_mask = np.where(etas<0)

	phis = np.zeros(phi_prime.shape)
	ones = np.zeros(phi_prime.shape)
	phis = phi_prime
	phis[xis < 0] += np.pi
	phis[ np.logical_and(etas < 0, xis >= 0) ] += 2*np.pi

	phis[phis >= 2*np.pi] = phis[phis >= 2*np.pi] - 2*np.pi

	return phis, thetas


def remove_dups(points_spatial):
	'''
	Remove duplicates
	'''
	return list(set(points_spatial))

def sort_arr(b):
	tuples = [tuple(t) for t in b]
	tuples = sorted(tuples)
	return np.array(tuples)


def make_rectangle(Nx, Ny, du=0.5, dv=0.5, s=1, t=1):
	xcoords = np.linspace(-(Nx-1)/2, (Nx-1)/2, Nx)*du
	ycoords = np.linspace(-(Ny-1)/2, (Ny-1)/2, Ny)*dv

	xcoords = xcoords[::s]
	ycoords = ycoords[::t]

	top = xcoords[-1]
	bottom = xcoords[0]
	left = ycoords[-1]
	right = ycoords[0]

	top_bar = [(top, y) for y in ycoords]
	bottom_bar = [(bottom, y) for y in ycoords]
	left_bar = [(x, left) for x in xcoords]
	right_bar = [(x, right) for x in xcoords]

	bars = top_bar + bottom_bar + left_bar + right_bar

	return np.array(bars)

def find_minimum_baseline_difference(b):
	b_prime = b.copy()
	b_prime.sort(axis=0) # sort each column independently
	diffs = np.diff(b_prime, axis=0)
	return np.max(diffs, axis=0)


def convert_from_xi_ya(xis, yas, N, ysat=np.pi*R, R=R, h=h, plot=True, fnames=[]):
	discriminant = (R+h)**2*np.cos(yas)**2*xis**4 - (R**2 + (R+h)**2)*xis**2 + R**2
	too_much = (R+h)**2 # too big to be a distance to a visible point
	rhos = np.where(discriminant >=0, ( R**2 + (R+h)**2 - 2*(R+h)**2*np.cos(yas)**2*xis**2 - 2*(R+h)*np.cos(yas)*np.sqrt(discriminant) )**0.5, too_much)

	mask_rhos = rhos <= np.sqrt(2*R*h+h**2)
	mask_xis = np.logical_and(xis<=R/(R+h),xis >= -R/(R+h))
	mask_yas = np.logical_and(yas<=np.arccos(R/(R+h)), yas >=-np.arccos(R/(R+h)))
	mask = np.logical_and(np.logical_and(mask_rhos, mask_xis), mask_yas)
	rhos = rhos.copy()[mask]
	xis = xis.copy()[mask]
	yas = yas.copy()[mask]
	if plot:
		t = np.linspace(-np.arccos(R/(R+h))+0.0000001, np.arccos(R/(R+h))-0.0000001, 500)
		y1 = R/((R+h)*np.cos(t)) * np.sqrt(( (R+h)**2*np.cos(t)**2 - R**2 )/( (R+h)**2 - R**2 ))
		y2 = -y1
		plt.figure(figsize=(16,14))
		plt.scatter(yas, xis, c=rhos, marker='.', alpha=0.5, linewidth=0.5)
		plt.title(r"Observed pixels, uniformly sampled in $\xi$-$y_a$ coords, plotted in $\xi$-$y_a$")
		cbar = plt.colorbar()
		cbar.set_label(r"$\rho$ (meters)")
		plt.xlabel(r"$y_a$")
		plt.ylabel(r"$\xi = \sin(\theta)\cos(\phi)$")
		plt.plot(t, y1, 'r--', label=r"horizon: $\xi = \pm \frac{R}{(R+h)\cos(y_a)}\sqrt{\frac{(R+h)^2\cos^2(y_a)-R^2}{(R+h)^2-R^2}}$")
		plt.plot(t, y2, 'r--')
		plt.legend(fontsize=7, loc=1)
		plt.savefig(fnames[0])
		plt.close()

	xas = np.arcsin(-rhos*xis/R)
	if plot:
		plt.figure(figsize=(16,14))
		plt.scatter(yas, xas, c=rhos, marker='.', alpha=0.5, linewidth=0.5)
		plt.xlabel(r"$y_a$")
		plt.ylabel(r"$x_a$")
		plt.title(r"Observed pixels uniform in $\xi$-$y_a$ coords, plotted in $x_a$-$y_a$")
		t = np.linspace(-np.arccos(R/(R+h))+0.00001, np.arccos(R/(R+h))-0.00001, 500)
		y1 = np.arccos(R/((R+h)*np.cos(t)))
		y2 = -y1
		plt.plot(t, y1, 'r--', label=r"horizon: $\cos(x_a)\cos(y_a) = \frac{R}{R+h}$")
		plt.plot(t, y2, 'r--')
		cbar = plt.colorbar()
		cbar.set_label(r"Distance from satellite $\rho$ (meters)")
		plt.legend(fontsize=8, loc=1)
		plt.savefig(fnames[1])
		plt.close()
	ps, ts, rs = xang_yang_to_phi_theta(xas, yas, R=R, h=h, return_rho=True, testing=False)
	m_vis, n_vis, mask, xi_vis, eta_vis, x_vis, y_vis, xang_vis, yang_vis = convert_from_phi_theta(ps, ts, M=N//2, N=N, R=R, h=h, ysat=ysat)
	if plot:
		plt.figure(figsize=(16,14))
		plt.scatter(eta_vis, xi_vis, c=rhos, marker='.', alpha=0.5, linewidth=0.5)
		plt.xlabel(r"$\eta=\sin(\theta)\sin(\phi)$")
		plt.ylabel(r"$\xi=\sin(\theta)\cos(\phi)$")
		plt.title(r"Observed pixels uniform in $\xi$-$y_a$ coords, plotted in $\xi$-$\eta$")
		t = np.linspace(-R/(R+h), R/(R+h), 500)
		y1 = np.sqrt(R**2/((R+h)**2)-t**2)
		y2 = -y1
		plt.plot(t, y1, 'r--', label=r"horizon: $\xi^2+\eta^2 = \frac{R^2}{(R+h)^2}$")
		plt.plot(t, y2, 'r--')
		cbar = plt.colorbar()
		cbar.set_label(r"$\rho$ (meters)")
		plt.legend(fontsize=8, loc=1)
		plt.savefig(fnames[2])
		plt.close()
	return m_vis, n_vis, mask, xi_vis, eta_vis, x_vis, y_vis, xang_vis, yang_vis, rhos, ps, ts


def xang_yang_to_phi_theta(xang, yang, R=R, h=h, return_rho=False, testing=True):
	cosgamma = np.multiply(np.cos(xang), np.cos(yang))
	rho = np.sqrt( R**2 + (R+h)**2 - 2*R*(R+h)*cosgamma )
	visible_mask = ( rho <= np.sqrt(2*R*h + h**2) )

	# R^2 = rho^2 + (R+h)^2 -2rho(R+h)cosu
	# cosu = (rho^2 + 2Rh + h^2)/(2*rho*(R+h))
	cosu = (rho**2 + 2*R*h + h**2)/(2*(R+h)*rho)
	theta = np.pi - np.arccos(cosu)

	sintheta = np.sin(theta)
	if testing:
		np.testing.assert_allclose(np.multiply(rho, sintheta), R*np.sqrt(np.sin(xang)**2 + np.multiply(np.sin(yang), np.cos(xang))**2), 1e-4, 1e-4)
	print("img to array", rho.shape, visible_mask.shape, np.sum(visible_mask))
	xang_visible = img_to_array(xang, visible_mask)
	yang_visible = img_to_array(yang, visible_mask)
	rho_visible = img_to_array(rho, visible_mask)
	print("now", xang_visible.shape, yang_visible.shape, rho_visible.shape)

	xi = (-R*np.sin(xang_visible)) / rho_visible
	eta = (R*np.sin(yang_visible)*np.cos(xang_visible)) / rho_visible

	if testing:
		np.testing.assert_allclose(sintheta, np.sqrt(xi**2 + eta**2), 1e-4, 1e-4)

	phi_prime = np.where(eta != 0, np.arctan(np.abs(np.divide(xi,eta,where=eta!=0))), np.pi/2)
	
	quad_1_mask = np.where(np.logical_and(xang_visible<0, yang_visible>=0))
	quad_2_mask = np.where(np.logical_and(xang_visible>=0, yang_visible>0))
	quad_3_mask = np.where(np.logical_and(xang_visible>0, yang_visible<=0))
	quad_4_mask = np.where(np.logical_and(xang_visible<=0, yang_visible<0))

	phi = np.zeros(phi_prime.shape)
	phi[quad_1_mask] = np.pi/2.0 - phi_prime[quad_1_mask]
	phi[quad_2_mask] = np.pi/2.0 + phi_prime[quad_2_mask]
	phi[quad_3_mask] = 3*np.pi/2.0 - phi_prime[quad_3_mask]
	phi[quad_4_mask] = 3*np.pi/2.0 + phi_prime[quad_4_mask]

	phi[phi >= 2*np.pi] = phi[phi >= 2*np.pi] - 2*np.pi
	if return_rho:
		return phi, theta, rho
	return phi, theta
	
def img_to_array(img, visible_mask, kind='F'):
	# By convention, we use Fortran ordering when writing the visible portion of an image of earth into a 1D array
	if len(img.shape) == 1:
		print('call on dim 1')
		return img[visible_mask]
	n = np.sum(visible_mask)
	if len(img.shape) == 3:
		assert img[visible_mask].shape[0] == n and img[visible_mask].shape[1] == img.shape[2]
		return img[visible_mask].copy().reshape((n,img.shape[2]), order=kind)
	assert len(img.shape) == 2

	return img[visible_mask].copy().reshape((n,), order=kind)
	
def J_xi_ya(rho, eta, theta, xa, ya, R=R, h=h, ablation=False):
	if ablation:
		return np.ones(rhos.shape)
	fac = R*np.cos(xa)
	num = np.abs(fac*(rho*np.cos(ya)-eta*(R+h)*np.sin(ya)))
	den = rho**2
	detadya = num/den
	return detadya/(-1*np.cos(theta))

def get_antenna_dict(points_spatial, antenna_patterns, antenna_types=['Cx', 'Cy'], ablation=False):
	if ablation:
		return {tuple(p):np.ones(antenna_patterns[(0, 'Cx')].shape) for p in points_spatial}
	return {tuple(p):antenna_patterns[(np.random.randint(0, 69), np.random.choice(antenna_types))] for i,p in enumerate(points_spatial)}


def sind(t,N):
	if np.issubdtype(type(t), np.number):
		t = np.array([t])
	if N % 2 == 0:
		return np.where(t==0, 1, np.divide( np.sin(np.pi*t), N*np.tan(np.pi/N * t), where=t!=0 ))
	else:
		return np.where(t==0, 1, np.divide( np.sin(np.pi*t), N*np.sin(np.pi/N * t), where=t!=0 ))

def interpolate_in_phi_theta_grid(antpattern, phis, incidences, pmin=0, tmin=0, dp=2.5*np.pi/180.0, dt=1.0*np.pi/180):
	pidx = (phis - pmin)/dp # radians divided by radians per pixel
	tidx = (incidences - tmin)/dt # radians divided by radians per pixel
	pts = np.zeros((len(phis),2))
	pts[:,0] = tidx
	pts[:,1] = pidx
	interpolated_values = periodic_interpolation(antpattern, pts)
	return interpolated_values

def periodic_interpolation(img, pts):
	M,N = img.shape
	I,c = pts.shape
	assert c == 2
	a = pts[:,0]
	b = pts[:,1]
	a_rep = np.repeat(a,M).reshape((M*I,1))
	b_rep = np.repeat(b,M).reshape((M*I,1))
	a_rep_tile = np.tile(a_rep, (1,N)) # a stack of I different MxN matrices, where the ith matrix in the stack is the constant pts[i,0]
	b_rep_tile = np.tile(b_rep, (1,N)) # a stack of I different MxN matrices, where the ith matrix in the stack is the constant pts[i,1]
	A = a_rep_tile - np.tile(np.tile(np.arange(M).reshape((M,1)), (1,N)), (I,1))
	B = b_rep_tile - np.tile(np.tile(np.arange(N).reshape((1,N)), (M,1)), (I,1))
	S = np.multiply(sind(A,M), sind(B,N))
	imgtile = np.tile(img, (I,1))
	prod = np.multiply(imgtile,S)
	list_of_arrays = np.split(prod, I, axis=0)
	return np.array([np.sum(x) for x in list_of_arrays])

def get_orthogonal_model_and_mean(incidence_degrees, angles=angles, pcs=pcs, mean_vec=mean_vec, n_components_simulation=12, kind='cubic', ablation=False):

	# interpolate mean vector over the grid
	Umean = interp1d(angles, mean_vec, kind=kind)(incidence_degrees)

	# interpolate twelve p.c.s over the grid
	pcs_snap = np.zeros((len(incidence_degrees), n_components_simulation))
	for i in range(n_components_simulation):
		pci = interp1d(angles, pcs[i], kind=kind)(incidence_degrees)
		pcs_snap[:,i] = pci

	if ablation:
		return Umean, pcs_snap, None, None

	# compute the *orthogonal* basis
	U, s, Vt = np.linalg.svd(pcs_snap, full_matrices=False)

	return Umean, U, s, Vt


def compute_temp(img, mask, Umean, U, number_components=12, var=var):
	# img : an MxNxC image
	# mask : an MxN boolean array True at locations visible False otherwise
	# For the remainder, call n the sum of mask
	# Umean : a length-n array
	# U : an nxC array whose cth column is the cth (normalized) principal component sampled over the n visible points in an array
	# number_components : the number of components in the model used to generate the simulated temperatures
	# var : the weights applied to the principal components

	temp_mean = Umean.copy() # mean temperature

	weighted_pcs = U[:,:number_components] @ np.diag(np.sqrt(var[:number_components]))

	n = np.sum(mask)

	img_vals = img[mask][:,:number_components] # n x number_components array

	simulated_temp = temp_mean + np.sum(img_vals * weighted_pcs, axis=1)
	return simulated_temp

def multiply_channels_indep(A,img):
	# A, a 2D KxM array
	# img, a 3D MxNxC array
	# returns KxNxC array by multiplying A by each channel of img, then writing this product back into the channels
	return (A@img.transpose((2,0,1))).transpose((1,2,0))

def regularize(x, d=np.arccos(R/(R+h))):
	return np.where(x<=d, x, x-2*np.pi)

def find_angle_in_list(ang, l, thresh=7):
	d1 = ang-l
	return np.abs(put_in_minus_pi_to_pi(d1)) < 10**(-1*thresh)
	#print("angle",  np.arctan(np.cos(d1), np.sin(d1)))
	#return np.abs( np.arctan(np.cos(d1), np.sin(d1)) ) < 10**(-1*thresh) 

def put_in_minus_pi_to_pi(a):
	b = a % (2*np.pi)
	if isinstance(b, np.floating):
		return b-2*np.pi if b>np.pi else b
	b[b>np.pi] = b[b>np.pi]-2*np.pi
	return b

def not_ascending(a):
	return np.any(np.diff(a) < 0)

def ascending(a):
	return np.all(np.diff(a) >= 0)

def B_k_mu(x,deg,mu,knots):
	res = np.zeros((deg,deg+1))
	for i in range(deg):
		a, b = knots[mu+i+1], knots[mu+i+1-deg]
		den = a-b
		res[i,i:i+2] = ((a-x)/den, (x-b)/den)
	return res

def R_mu(x,deg,mu,knots):
	res = np.array([1])
	for i in range(1,deg+1):
		res = res @ B_k_mu(x,i,mu,knots)
	return res

def b_spline_matrix(xeval_locs, deg, knots):
	assert ascending(knots)
	assert ascending(xeval_locs)
	assert np.max(xeval_locs) <= knots[-1]
	neval = len(xeval_locs)
	mus = np.array(list(map(lambda x: np.nonzero(knots>x)[0][0] - 1, xeval_locs)))
	res = np.zeros((neval,len(knots)))
	for i, x, mu in zip(range(neval),xeval_locs, mus):
		bs = R_mu(x,deg,mu,knots)
		if mu-deg >= 0:
			res[i,mu-deg:mu+1] = bs
	return res

def get_img_of_params_bspline(img,xang_vis,number_knots,deg,d=np.arccos(R/(R+h)), margin=0.02, regularization=False, l=0):
	try:
		M, N, C = img.shape
	except:
		print("one-channel image")
		M, N = img.shape
		C = 1
		img = img.reshape((img.shape[0], img.shape[1], 1))
		
	ms, ns = np.meshgrid(np.arange(M), np.arange(N), indexing='ij')
	xs_all = (ms - (M-1)/2)* np.pi/M
	ys_all = ns*2*np.pi/N
	horizon_mask = np.logical_and(xs_all >= -d - margin, xs_all <= d + margin)
	nxs = len(np.unique(np.around(xs_all[horizon_mask], thresh)))
	min_m = int(np.min(xs_all[horizon_mask]*M/np.pi+(M-1)/2))
	img_horizon = img[min_m:min_m+nxs,:]

	# get interior knots
	dquantile = 100/(number_knots+1)
	knots = np.zeros((number_knots,))
	for i in range(1,number_knots+1):
		knots[i-1] = np.percentile(xang_vis,i*dquantile)    
	
	xs_sorted = np.sort(np.unique(np.around(xs_all[horizon_mask], thresh)))
	knot_prime = np.concatenate( ([xs_sorted[0]-eps]*(deg+1), knots, [xs_sorted[-1]+eps]*(deg)) )

	B = b_spline_matrix(xs_sorted, deg, knot_prime)
	if regularization:
		param_img = multiply_channels_indep(np.linalg.pinv(B+l*np.eye(B.shape[0],M=B.shape[1])), img_horizon)
	else:
		param_img = multiply_channels_indep(np.linalg.pinv(B),img_horizon)

	return param_img, B, knot_prime, xs_all, img_horizon

def assign_and_resample_antenna_pattern(ps_vis, incidences, points_spatial, delta_xi, delta_y, seed=10,thresh=10, ablation_antenna=False):
	SM_mat = loadmat('LICEFavpA.mat')
	fname = f'antpattern_dict_for_xi_{delta_xi}_ya_{delta_y}.pkl'
	if os.path.isfile(fname):
		with open(fname, 'rb') as inf:
			antenna_patterns = pickle.load(inf)
	else:
		antenna_patterns = {}
		for ant_no in range(69):
			print("ant_no", ant_no)
			for type_ in ['Cx', 'Cy']:
				antpattern = SM_mat[type_][0:91,:,ant_no]
				real_interpolated_values = interpolate_in_phi_theta_grid(np.real(antpattern), ps_vis, incidences)
				imag_interpolated_values = interpolate_in_phi_theta_grid(np.imag(antpattern), ps_vis, incidences)
				interpolated_values = real_interpolated_values + 1j*imag_interpolated_values
				antenna_patterns[(ant_no,type_)] = interpolated_values
		with open(fname, 'wb') as outf:
			pickle.dump(antenna_patterns, outf)
	np.random.seed(seed)
	A = get_antenna_dict(points_spatial, antenna_patterns, antenna_types=['Cx', 'Cy'], ablation=ablation_antenna)
	return A

def assign_and_resample_antenna_pattern_matrix(ps_vis, incidences, points_spatial, delta_xi, delta_y, seed=10,thresh=10, ablation_antenna=False, scenario_name=''):
	SM_mat = loadmat('LICEFavpA.mat')
	fname = f'antpattern_matrix_for_xi_{delta_xi}_ya_{delta_y}_{scenario_name}.pkl'
	np.random.seed(seed)
	if os.path.isfile(fname):
		with open(fname, 'rb') as inf:
			antenna_patterns = pickle.load(inf)
	else:
		antenna_patterns = np.zeros((len(points_spatial), len(ps_vis)), dtype=np.complex128) 
		solved = {}
		for ant_i in range(len(points_spatial)):
			print("ant_no", ant_i)
			ant_no = np.random.choice(range(69))
			type_ = np.random.choice(['Cx', 'Cy'])
			if (ant_no, type_) in solved:
				antenna_patterns[ant_i,:] = solved[(ant_no, type_)]
				continue
			antpattern = SM_mat[type_][0:91,:,ant_no]
			real_interpolated_values = interpolate_in_phi_theta_grid(np.real(antpattern), ps_vis, incidences)
			imag_interpolated_values = interpolate_in_phi_theta_grid(np.imag(antpattern), ps_vis, incidences)
			interpolated_values = real_interpolated_values + 1j*imag_interpolated_values
			antenna_patterns[ant_i,:] = interpolated_values
			solved[(ant_no, type_)] = interpolated_values
		with open(fname, 'wb') as outf:
			pickle.dump(antenna_patterns, outf)
	return antenna_patterns


def get_vis_one_pixel_stride_vectorized(param_img,deg,knots,Ti,Tf,
							 points_spatial, xi_vis, eta_vis, xang_vis, yang_vis,
							 hkl, Umean, U, var=var,
							 n_components_simulation=1, fname=None):
	try:
		Mh, N, C = param_img.shape
	except:
		print("given one-channel image")
		Mh, N = param_img.shape
		C=1
	assert n_components_simulation <= C

	if os.path.isfile(fname):
		with open(fname, 'rb') as inf:
			vis, vtilde, vis_with_mean = pickle.load(inf)
			return vis, vtilde, vis_with_mean

	Nant = len(points_spatial)
	Nvis = Nant**2
	print("Nvis", Nvis, Tf, Ti, "why not integer???")
	vis = np.zeros((int(Nvis),int(Tf-Ti)+1), dtype=np.complex128)
	vis_with_mean = np.zeros((int(Nvis), int(Tf-Ti)+1), dtype=np.complex128)
	
	distinct_yas = np.sort(np.unique(np.around(yang_vis, thresh)))
	dy = np.diff(distinct_yas)[0]
	Nys = len(distinct_yas)
	assert Nys % 2 == 1
	
	resampled_imgs = {}
	
	# precompute images of resampled params
	for i, ya_col in enumerate(distinct_yas):
		mask_ya = np.abs(np.around(yang_vis,thresh) - ya_col) < 10**(-1*thresh)
		xa_vals_sorted = np.sort(xang_vis[mask_ya])
		B = b_spline_matrix(xa_vals_sorted, deg, knots)
		resampled_imgs[i] = multiply_channels_indep(B, param_img)

	for ssp_i in range(Ti, Tf+1):
		temp_map = np.zeros(yang_vis.shape)
		for idx_in_resampled, col_offset in enumerate(np.arange(-(Nys//2), Nys//2+1)):
			col = (ssp_i + col_offset) % N
			for c_ in range(n_components_simulation):
				col_bt_params = resampled_imgs[idx_in_resampled][:,col,c_]
				radians_of_col = distinct_yas[idx_in_resampled]
				idx_match = np.nonzero(find_angle_in_list(radians_of_col, yang_vis, thresh))[0][::-1]
				temp_map[idx_match] += col_bt_params*U[idx_match,c_]*np.sqrt(var[c_])
				 
		vis[:, ssp_i-Ti] = hkl @ temp_map
		vis_with_mean[:,ssp_i-Ti] = hkl @ (temp_map + Umean)

	vtilde = np.fft.fft(vis, axis=1)
	with open(fname, 'wb') as f:
		pickle.dump([vis, vtilde, vis_with_mean], f)

	return vis, vtilde, vis_with_mean

def invert_vis_one_pixel_stride_vectorized(vtilde,deg,knots,Ti,Tf,Mh,No,C,xa_rs,ya_rs,xi_rs,eta_rs,hkl,U,points_spatial,offset_xi=0,orbit_fraction=0.5,n_components_inversion=1,thresh=10,ablation_cexp=False,fname=None,var=var):
	if os.path.isfile(fname):
		with open(fname, 'rb') as inf:
			param_est, I, Gs, param_est_full = pickle.load(inf)
			return param_est, I, Gs, param_est_full
	
	T = Tf-Ti+1

	Nant = len(points_spatial)
	Nvis = Nant**2
	Ngrid = len(ya_rs)

	# Get snapshot grid -- identical for each snapshot besides mask
	all_ys = np.sort(np.unique(np.around(ya_rs,thresh)))
	assert np.all(np.abs(all_ys) < np.arccos(R/(R+h)))
	Nys = len(all_ys) 
	all_xs = np.unique(np.around(xa_rs,thresh))
	number_xs = len(all_xs) 
	
	yoffset = np.arange(-(Nys//2), Nys//2+1)*2*np.pi/No
	print("all_ys", all_ys, "yoffset", yoffset)
	def xs(y):
		mask = np.abs(ya_rs-y) < 10**(-1*thresh)
		return np.sort(xa_rs[mask])
	Ms = [b_spline_matrix(xs(y), deg, knots) for y in yoffset]
	Gs = []
	Ipartial = np.zeros((Mh*n_components_inversion, T), dtype=np.complex128)
	for omega in range(T):
		print(r"Inverting snapshot frequency $\omega$={}".format(omega))
		G = np.zeros((Nvis,Mh*n_components_inversion), dtype=np.complex128)
		xy = hkl*np.exp(-2*np.pi*1j*Ti*omega/T)*np.exp(1j*omega*N*ya_rs/T)

		for ii,ya in enumerate(all_ys):
			mask_idxs = np.abs(np.around(ya_rs,thresh) - ya) < 10**(-1*thresh)
			mask_idx_numbers = np.flatnonzero(mask_idxs)
			mask_idxs = mask_idx_numbers[xa_rs[mask_idxs].argsort()]
			xys = xy[:,mask_idxs]
			for c in range(n_components_inversion):
				std = np.sqrt(var[c])
				Tc = std*U[:,c]
				Tcs = np.tile(Tc[mask_idxs].reshape((1,len(mask_idxs))), (Nvis,1))
				QTM = (xys*Tcs)@Ms[ii]
				G[:,Mh*c:Mh*(c+1)] += QTM

		Ginv = np.linalg.pinv(G)
		Gs.append((G,Ginv))
		res = Ginv @ vtilde[:,omega]
		Ipartial[:,omega] = res
	I = np.real(np.fft.ifft(Ipartial, axis=1))
	param_est = np.zeros((Mh, T, n_components_inversion))

	for iter_c in range(n_components_inversion):
		for iter_t in range(T):
			param_est[:,iter_t,iter_c] = I[(Mh*iter_c):(Mh*(iter_c+1)),iter_t]

	param_est_full = np.zeros((Mh, No, n_components_inversion))
	param_est_full[:, Ti:Ti+T, :] = param_est
	
	with open(fname, 'wb') as f:
		pickle.dump([param_est, I, Gs, param_est_full], f)

	return param_est, I, Gs, param_est_full

## SIMULATION EXECUTION ##
for Nx in [65]:
	for Ny in [65]:
		'''
		points_spatial = make_rectangle(Nx, Ny, 0.5, 0.5)
		points_spatial = sort_arr(np.array(remove_dups([tuple(t) for t in points_spatial])))

		nant = points_spatial.shape[0]
		nvis = nant**2
		
		first_antennas = np.repeat(points_spatial, nant, axis=0)
		second_antennas = np.tile(points_spatial, (nant,1))
		baselines_ = (second_antennas - first_antennas)
		baselines = sort_arr(baselines_)

		dc_baseline_idxs = np.where(np.sum(baselines == 0.0, 1) == 2)[0]

		n_unique_baselines = len(set(tuple(b) for b in baselines))
		idxs = np.zeros((n_unique_baselines,), dtype=np.dtype(int))
		ss = set()
		cnt = 0
		for i,b in enumerate(baselines):
			if tuple(b) not in ss:
				idxs[cnt] = i
				ss.add(tuple(b))
				cnt += 1

		unique_baselines = baselines[idxs,:]
		'''

		baselines, points_spatial = np.array([[0, 19.25*np.sqrt(3)]]), np.array([[0,0], [0, 19.25*np.sqrt(3)]])
		nvis = len(baselines)
		nant = len(points_spatial)

		# minor params
		margin = 0.02
		d = np.arccos(R/(R+h))
		eps = 0.00001
		thresh=10
		n = 500
		
		# major params
		for decimation_factor_x, decimation_factor_y in zip([4],[1]): # can be fractional due to upsampling zoom
			N = 21600/decimation_factor_y
			try:
				print("Got here!", decimation_factor_x, decimation_factor_y)
				for number_components in [3]:
					cexp_weights = {}
					for zoom in [1]:
						for dxi, dya in zip([0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0.00005,0.00002,0.00001], [2*np.pi/(N*zoom)]*11):
							cnt = 0
							offset_xi = 0
							orbit_fraction = 0.5 # gives offset_ya
							# get Riemann sum grid
							N_xi = 2//dxi + 1
							xis = np.array([offset_xi + dxi*i for i in np.arange(-N_xi//2-1, N_xi//2+1) if np.abs(offset_xi+dxi*i)<1])
							ns = np.arange(N*zoom)
							ysat = 2*np.pi*orbit_fraction
							yangs = ns*(2*np.pi)/(N*zoom) - ysat # signed angular geodesic distance from subsatellite point (latitudinal distance from SSP)
							xis = np.array([offset_xi + dxi*i for i in np.arange(-N_xi//2-1, N_xi//2+1) if np.abs(offset_xi+dxi*i)<1])
							xis_grid, yangs_grid = np.meshgrid(xis, yangs, indexing='ij')
							xis_all, yangs_all = xis_grid.flatten(), yangs_grid.flatten()

							m_vis, n_vis, mask, xi_vis, eta_vis, x_vis, y_vis, xang_vis, yang_vis, rhos, ps, ts = convert_from_xi_ya(xis_all, yangs_all, N, plot=False)
							for b in baselines:
								cexp_weights[(dxi, dya, b[0], b[1])] = dblquad_montecarlo_jacobian(b[0], b[1], xi_vis, yang_vis, dxi, dya, n=n, R=R, h=h)
								#cexp = cexp_int_weights(xi_vis, yang_vis, b[0], b[1], dxi=dxi, dya=dya)
								#cexp_weights[(dxi, dya, b[0], b[1])] = cexp
								cnt += 1
								if cnt % 100 == 1:
									print(f"{cnt/len(baselines):.4f} fraction done")
									res = cexp_weights[(dxi,dya,b[0],b[1])]
							with open(f"weights_1925_{dxi:.6f}_{dya:.6f}_{decimation_factor_x}_{decimation_factor_y}_n_{n}.npy", "wb") as f:
								pickle.dump(cexp_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
						

			except Exception as e:
				print("Error:", e)
				print(traceback.print_exc())
				print(traceback.format_exc())
				print("tb", sys.exc_info()[2])
				continue
