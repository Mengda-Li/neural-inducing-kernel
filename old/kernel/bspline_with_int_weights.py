## IMPORT STATEMENTS ##
from IPython.display import display, Image
import math, time, os, sys, pickle, collections, warnings
warnings.filterwarnings(action='once')
import traceback
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats as st
from scipy.signal import find_peaks, convolve
from scipy.io import loadmat
from scipy.ndimage.filters import generic_filter as gf
from scipy.linalg import circulant, dft
from scipy.interpolate import splrep, BSpline, interp1d, RectBivariateSpline
from scipy.integrate import dblquad
import pickle 
import datetime

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

def eta(xi, ya, R=R, h=h):
    discriminant = (R+h)**2*np.cos(ya)**2*xi**4 - (R**2 + (R+h)**2)*xi**2 + R**2
    rho = np.sqrt( R**2 + (R+h)**2 - 2*(R+h)**2*np.cos(ya)**2*xi**2-2*(R+h)*np.cos(ya)*np.sqrt(discriminant) )
    xa = np.arcsin(-rho*xi/R)
    eta = R*np.sin(ya)*np.cos(xa)/rho
    return eta

def integrate_box(xictr, yactr, u, v, dxi=0.001, dya=2*np.pi/(5400*5), jacobian=False, epsabs=1e-4, epsrel=1e-4):
    gfunc = lambda y: np.minimum(-1*horizon_xi_val(y), xictr-dxi)
    hfunc = lambda y: np.minimum(horizon_xi_val(y), xictr+dxi)

    if jacobian:
        pass
    else:
        freal = lambda xi, ya: np.cos(-2*np.pi*1j*(u*xi+v*eta(xi,ya)))  #*np.exp(1j*omega*N/T)*np.exp(-2*np.pi*1j*Ti*omega/T)
        fcplx = lambda xi, ya: np.sin(-2*np.pi*1j*(u*xi+v*eta(xi,ya)))
        return dblquad(freal, yactr-dya, yactr+dya, gfunc, hfunc, epsabs=epsabs, epsrel=epsrel) + 1j*dblquad(fcplx, yactr-dya, yactr+dya, gfunc, hfunc, epsabs=epsabs, epsrel=epsrel)


def cexp_int_weights(xi_sample, ya_sample, u, v, dxi=0.001, dya=2*np.pi/(5400*5) ,epsabs=1e-4, epsrel=1e-4):
    int_fn = np.vectorize(lambda x, y: integrate_box(x,y,u,v,dxi,dya,epsabs,epsrel))
    weights = int_fn(xi_sample, ya_sample)
    return weights

def get_cexp_term(xi_sample, ya_sample, baselines, dxi=0.001, dya=2*np.pi/(5400*5), epsabs=1e-4, epsrel=1e-4):
    f = np.vectorize(lambda u, v: cexp_int_weights(xi_sample, ya_sample, u, v, dxi, dya, epsabs, epsrel))
    return f(baselines[:,0], baselines[:,1])

def load_weights(xi_sample, ya_sample, baselines, dxi, dya, dfx, dfy, n):
	fname = f"weights_{dxi:.6f}_{dya:.6f}_{dfx}_{dfy}_n_{n}.npy"
	with open(fname, 'rb') as f:
		d = pickle.load(f)
	res = np.zeros((len(baselines), len(xi_sample)), dtype=np.complex128)
	print("loading weights, expected shape", res.shape, "actual shape", d[(dxi, dya, baselines[0,0], baselines[0,1])].shape)
	print("d", d.keys())
	for i,b in enumerate([tuple(ab) for ab in baselines]):
		res[i,:] = d[(dxi, dya, b[0], b[1])]
	return res

def cubic_hermite(img, dy, fracs):
    # fracs = pixel fractions at which to sample, e.g., [f*dy for f in fracs]
    h00 = lambda t: (1+2*t)*(1-t)**2
    h10 = lambda t: t*(1-t)**2
    h01 = lambda t: t**2*(3-2*t)
    h11 = lambda t: t**2*(t-1)

    M,N = img.shape
    dimg = np.diff(img, axis=1, prepend=img[:,-1].reshape((M,1)), append=img[:,[0,1]])/dy
    m0 = dimg[:,:-2] # left finite difference from left endpoint
    m1 = dimg[:,2:] # right finite difference from right endpoint
    p0 = img # left endpoint
    p1 = np.hstack((img[:,1:], img[:,0].reshape((M,1)))) # right endpoint
    ms, ns = np.meshgrid(np.arange(M), np.arange(N), indexing='ij')
    out = np.zeros((M,N*(len(fracs)+1)))
    out[:,::(len(fracs)+1)] = img
    for i,t in enumerate(fracs):
        interpimg = h00(t)*p0 + h10(t)*m0*dy + h01(t)*p1 + h11(t)*m1*dy
        out[:, (1+i)::(len(fracs)+1)] = interpimg
    return out


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

def three_part_plot(pi, pe, title, xlabel, ylabel, fname, cbarticks=np.arange(-8,5), cbarlabels=[r'$10^{-8}$', r'$10^{-7}$',r'$10^{-6}$',r'$10^{-5}$',r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$']):
	print("PI", pi.shape, "PE", pe.shape)
	temp2p5_sim = compute_t_theta(pi,0,mean_vec=mean_vec,var=var,pcs=pcs,nparam=number_components)
	temp2p5_rec = compute_t_theta(pe,0,mean_vec=mean_vec,var=var,pcs=pcs,nparam=number_components)
	temp22p5_sim = compute_t_theta(pi,4,mean_vec=mean_vec,var=var,pcs=pcs,nparam=number_components)
	temp22p5_rec = compute_t_theta(pe,4,mean_vec=mean_vec,var=var,pcs=pcs,nparam=number_components)
	temp57p5_sim = compute_t_theta(pi,11,mean_vec=mean_vec,var=var,pcs=pcs,nparam=number_components)
	temp57p5_rec = compute_t_theta(pe,11,mean_vec=mean_vec,var=var,pcs=pcs,nparam=number_components)
	mm, nn = pe.shape[:2]
	synth = np.zeros((mm, nn))
	synth[:,:nn//3] = np.abs(temp2p5_sim-temp2p5_rec)[:,:nn//3]
	synth[:,nn//3:2*nn//3] = np.abs(temp22p5_sim-temp22p5_rec)[:,nn//3:2*nn//3]
	synth[:,2*nn//3:] = np.abs(temp57p5_sim-temp57p5_rec)[:,2*nn//3:]
	fig, ax = plt.subplots(1)
	cax = ax.imshow(np.log10(synth))
	plt.ylabel(ylabel)
	plt.title(title)
	plt.xlabel(xlabel)
	ax.xaxis.set_ticks([])
	ax.yaxis.set_ticks([])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	cbar = fig.colorbar(cax, fraction=0.0195, pad=0.04, ticks=cbarticks)
	cbar.ax.set_yticklabels(cbarlabels)
	plt.savefig(fname)
	plt.close()
	return temp2p5_sim, temp2p5_rec, temp22p5_sim, temp22p5_rec, temp57p5_sim, temp57p5_rec


def plot_with_cbar_sidebar_knots(img, main_title, fname, rows=None, show_sidebar=True, show_knots=False, knot_idxs=[],
								main_x=None, main_y=None, ctitle=None, padleft=1, cbarsize='5%', rightsize='10%', padright=0.2, 
								figsize=(20,8), clim=None, dk=0.01, cmap='coolwarm', main_fontsize=20):
	fig = plt.figure(figsize=figsize)
	ax0 = plt.axes()
	ax0.set_title(main_title, fontsize=main_fontsize)
	ax0.tick_params(direction='in', top=True, right=True)
	im = ax0.imshow(img, cmap=cmap)
	ax0.set_axis_off()
	if main_x is not None:
		ax0.set_xlabel(main_x)
	if main_y is not None:
		ax0.set_ylabel(main_y)

	divider = make_axes_locatable(ax0)
	cax = divider.append_axes('left', size=cbarsize, pad=padleft)
	cbar = fig.colorbar(im, cax=cax)
	if clim is not None:
		im.set_clim(clim[0], clim[1])
	cbar.ax.tick_params(axis='y', color='white', left=True, right=True, length=5, width=1.5)
	if ctitle is not None:
		cbar.ax.set_title(ctitle, loc='left')

	if show_sidebar:
		ax2 = divider.append_axes('right', size=rightsize, pad=padright)
		ax2.tick_params(direction='out', labelleft=False)
		ax2.barh(np.arange(img.shape[0]), np.mean(img, axis=1), xerr=0.1*np.std(img, axis=1))
		ax2.invert_yaxis()
		if show_knots:
			ts = np.linspace(np.min(rows)-np.abs(np.min(rows))*dk, np.max(rows)+np.abs(np.min(rows))*dk, 100)
			for k in knot_idxs:
				plt.plot(ts, k*np.ones(100,), 'g--')

	plt.savefig(fname)
	plt.close()


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

	ns_visible = (ys_visible+ysat*np.cos(xangs_visible))*N/(2*np.pi*R*np.cos(xangs_visible))
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
	if Ny == 1:
		ycoords = np.array([0])
	else:
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
	xang_visible = img_to_array(xang, visible_mask)
	yang_visible = img_to_array(yang, visible_mask)
	rho_visible = img_to_array(rho, visible_mask)

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

def rect_bivariate_spline(img, pts, **args):
    assert len(img.shape) == 2
    M,N = img.shape
    I,c = pts.shape
    assert c == 2
    interpolator = RectBivariateSpline(np.arange(M), np.arange(N), img, **args)
    interpolated = interpolator(pts[:,0], pts[:,1], grid=False)
    return interpolated

def interpolate_in_phi_theta_grid(antpattern, phis, incidences, interp_func=rect_bivariate_spline, pmin=0, tmin=0, dp=2.5*np.pi/180.0, dt=1.0*np.pi/180):
    pidx = (phis - pmin)/dp # radians divided by radians per pixel
    tidx = (incidences - tmin)/dt # radians divided by radians per pixel
    pts = np.zeros((len(phis),2))
    pts[:,0] = tidx
    pts[:,1] = pidx
    interpolated_values = interp_func(antpattern, pts)
    return interpolated_values

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


def get_vis(param_img, deg, knots, Ti, Tf, s,
			points_spatial, xang_vis, 
			yang_vis, hkl, Umean, U, var=var,
			n_components_simulation=1, fname=None, thresh=10):
	try:
		L, N, C = param_img.shape
	except:
		print("given one-channel image")
		L, N = param_img.shape
		C=1
	assert n_components_simulation <= C

	if os.path.isfile(fname):
		with open(fname, 'rb') as inf:
			vis, vtilde, vis_with_mean = pickle.load(inf)
			return vis, vtilde, vis_with_mean

	Nant = len(points_spatial)
	Nvis = Nant**2
	print("computing visibilities,...Nvis", Nvis, "Tf", Tf, "Ti", Ti)
	print("U", U.shape, "Umean", Umean.shape, "hkl", hkl.shape)
	vis = np.zeros((int(Nvis), int(Tf-Ti)+1), dtype=np.complex128)
	vis_with_mean = np.zeros((int(Nvis), int(Tf-Ti)+1), dtype=np.complex128)
	
	distinct_yas = np.sort(np.unique(np.around(yang_vis, thresh)))
	dy = np.diff(distinct_yas)[0]
	Nys = len(distinct_yas)
	assert Nys % 2 == 1
	
	resampled_imgs = {}
	
	# precompute images of resampled params
	for i, ya_col in enumerate(distinct_yas):
		mask_ya = find_angle_in_list(ya_col, yang_vis, thresh)
		xa_vals_sorted = np.sort(xang_vis[mask_ya])
		B = b_spline_matrix(xa_vals_sorted, deg, knots)
		resampled_imgs[i] = multiply_channels_indep(B, param_img)
	for ssp_i in range(Ti, Tf+1, s):
		print("ssp_i", ssp_i)
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


def Q_c(ya_col, yas, hkl_tiled, Tc_tiled, thresh=10):
	idx_match = np.nonzero(find_angle_in_list(ya_col, yas, thresh))[0][::-1]
	return hkl_tiled[:,idx_match]*Tc_tiled[:,idx_match]


def xs(y, xas, yas, thresh=10):
	#for y__ in np.unique(np.around(yas,10)):
	#	print(y, y__) 
	mask = find_angle_in_list(y, yas, thresh=6)
	return np.sort(xas[np.nonzero(mask)[0][::-1]])

def ufunc(j, c, omega, s, T, z, ya_col, xas, yas, hkl_tiled, Tc_tiled, dya, thresh=10):

	p0 = lambda t: (1+2*t)*(1-t)**2
	p1 = lambda t: t*(1-t)**2
	p2 = lambda t: t**2*(3-2*t)
	p3 = lambda t: t**2*(t-1)

	Nvis = Tc_tiled.shape[0]

	all_ys = np.sort(np.unique(np.around(yas, thresh)))	
	all_xs = np.sort(np.unique(np.around(xas,thresh)))
	number_xs = len(all_xs)
	yoffset = np.array([j+q/(z+1) for q in range(z+1)])*dya
	Bs = [b_spline_matrix(xs(y, xas, yas, thresh), deg, knots) for y in yoffset if np.abs(y) <= np.arccos(R/(R+h))]
	
	L = Bs[0].shape[1]

	r0 = np.zeros((Nvis,L), dtype=np.complex128)
	Nx = Bs[0].shape[0]
	r1 = Q_c(j*dya, yas, hkl_tiled, Tc_tiled, thresh).reshape((Nvis,Nx))@Bs[0]
	r2 = np.zeros((Nvis,L), dtype=np.complex128)
	r3 = np.zeros((Nvis,L), dtype=np.complex128)
	
	Nz = len(Bs)

	for q in range(1,Nz):
		ya_col = (j+q/(z+1))*dya
		B = Bs[q]
		Nx = Bs[q].shape[0]
		Q_c_yacol = Q_c(ya_col, yas, hkl_tiled, Tc_tiled, thresh).reshape((Nvis,Nx))
		r0 -= p1(q/(z+1))*Q_c_yacol@B
		r1 += (p0(q/(z+1))+p1(q/(z+1)))*Q_c_yacol@B
		r2 += (p2(q/(z+1))-p3(q/(z+1)))*Q_c_yacol@B
		r3 += p3(q/(z+1))*Q_c_yacol@B

	return (1/s)*(np.exp(2*np.pi*1j*(j-1)*omega/(s*T))*r0 + np.exp(2*np.pi*1j*j*omega/(s*T))*r1 + \
			 np.exp(2*np.pi*1j*(j+1)*omega/(s*T))*r2 + np.exp(2*np.pi*1j*(j+2)*omega/(s*T))*r3)

def invert_vis(vtilde,deg,knots,Ti,Tf,s,z,Nys,L,N,C,xa_rs,ya_rs,xi_rs,eta_rs,dya,hkl_tiled,
			Tcs,n_components_inversion=1,thresh=10,fname=None,save_g=False):

	if os.path.isfile(fname):
		with open(fname, 'rb') as inf:
			if save_g:
				param_est, I, Gs, param_est_full = pickle.load(inf)
				return param_est, I, Gs, param_est_full
			else:
				param_est, I, param_est_full = pickle.load(inf)
				return param_est, I, param_est_full
	T = (Tf-Ti)+1
	Nvis = hkl_tiled.shape[0]
	js = np.arange(-(Nys//2), Nys//2+1)
	if save_g:
		Gs = []
	Ipartial = np.zeros((L*n_components_inversion, s*T), dtype=np.complex128)
	for omega in range((Tf-Ti)//s+1):
		print(r"Computing G for snapshot frequency $\omega$={}".format(omega*s), datetime.datetime.now())
		G = np.zeros((Nvis,s*L*n_components_inversion), dtype=np.complex128)
###xy = hkl*np.exp(-2*np.pi*1j*Ti*omega/T)*np.exp(1j*omega*N*ya_rs/T)

		for ii,j in enumerate(js):
			for c in range(n_components_inversion):
				ya_col = j*dya
				ujw = ufunc(j, c, omega, s, T, z, ya_col, xa_rs, ya_rs, hkl_tiled, Tcs[c], dya, thresh=thresh)
				G[:,L*c*s:L*(c+1)*s] += np.tile(ujw, (1,s))

		print(r"Inverting G for snapshot frequency $\omega=${}".format(omega*s), f"{Nvis}x{s*L*n_components_inversion}", datetime.datetime.now())
		Ginv = np.linalg.pinv(G)
		print(r"Inverted!", datetime.datetime.now())
		if save_g:
			Gs.append((G,Ginv))
		res = Ginv @ vtilde[:,omega]
		cols = [(omega+k*T) % N for k in range(s)]
		for i_, c_ in enumerate(cols):
			for c in range(C):
				Ipartial[(L*c):(L*(c+1)),c_] = res[L*s*c+i_*L:L*s*c+(i_+1)*L]
	I = np.real(np.fft.ifft(Ipartial, axis=1))
	param_est = np.zeros((L, T, n_components_inversion))

	for iter_c in range(n_components_inversion):
		for iter_t in range(T):
			param_est[:,iter_t,iter_c] = I[(L*iter_c):(L*(iter_c+1)),iter_t]

	param_est_full = np.zeros((L, N, n_components_inversion))
	param_est_full[:, Ti:Ti+T, :] = param_est
	
	with open(fname, 'wb') as f:
		if save_g:
			pickle.dump([param_est, I, Gs, param_est_full], f)
			return param_est, I, Gs, param_est_full
		else:
			pickle.dump([param_est, I, param_est_full], f)
			return param_est, I, param_est_full

## SIMULATION EXECUTION ##
for Nx in [65]:
	for Ny in [65]:
		du = 64/(Nx-1)*0.5
		dv = 64/(Ny-1)*0.5 if Ny > 1 else 0
		points_spatial = make_rectangle(Nx, Ny, du, dv)
		points_spatial = sort_arr(np.array(remove_dups([tuple(t) for t in points_spatial])))

		nant = points_spatial.shape[0]
		nvis = nant**2
		
		print("original len ps", len(points_spatial))

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
		print("baselines", baselines)

		max_u, max_v = np.max(np.abs(unique_baselines), axis=0)
		nyquist_xi, nyquist_eta = 1/(2*max_u), 1/(2*max_v)
		print("nyquist xi, nyquist eta", nyquist_xi, nyquist_eta)

		# minor params
		margin = 0.02
		d = np.arccos(R/(R+h))
		eps = 0.00001
		thresh=8
		save_g = False	
		# major params
		for decimation_factor_x, decimation_factor_y in zip([1],[2]):
			try:
				print("Got here!", decimation_factor_x, decimation_factor_y)
				for number_components in [3]:
					# import image 
					img = mpimg.imread('bluemarble1.jpg')/255.0
					img2 = img.copy()[::decimation_factor_x, ::decimation_factor_y, :number_components]
					M,N = img2.shape[:2]
					C = 1 if len(img2.shape) == 2 else img2.shape[2]

					# Plot import image
					if len(img2.shape) == 2:
						plt.figure()
						plt.imshow(img2)
						plt.axis('off')
						plt.title("Image of BT params")
						plt.colorbar()
						plt.savefig(f"original_image_dfx_{decimation_factor_x}_dfy_{decimation_factor_y}_component_1_out_of_1.png")
						plt.close()
					elif len(img2.shape) ==3:
						for c in range(img2.shape[2]):
							plt.figure()
							plt.imshow(img2[:,:,c])
							plt.axis('off')
							plt.title(f"Image of BT params, param no {c+1}")
							plt.colorbar()
							plt.savefig(f"original_image_dfx_{decimation_factor_x}_dfy_{decimation_factor_y}_component_{c}_out_of_{img2.shape[2]}.png")
							plt.close()
                    
					for dxi, dya in zip([0.04], [2*np.pi/N]):
						for long_upsample_bcoeffs, z in zip([True, True, True], [1]):
							offset_xi = 0
							orbit_fraction = 0.5 # implicitly gives offset_ya
							# get Riemann sum grid
							N_xi = 2//dxi + 1
							xis = np.array([offset_xi + dxi*i for i in np.arange(-N_xi//2-1, N_xi//2+1) if np.abs(offset_xi+dxi*i)<1])
							ns = np.arange(N)
							ysat = 2*np.pi*orbit_fraction
							yangs = ns*(2*np.pi)/N - ysat # signed angular geodesic distance from subsatellite point (latitudinal distance from SSP)
							xis_grid, yangs_grid = np.meshgrid(xis, yangs, indexing='ij')
							xis_all, yangs_all = xis_grid.flatten(), yangs_grid.flatten()

							m_vis, n_vis, mask, xi_vis, eta_vis, x_vis, y_vis, xang_vis, yang_vis, rhos, ps, ts = convert_from_xi_ya(xis_all, yangs_all, N, plot=False)
							m_vis_, n_vis_, mask_, xi_vis_, eta_vis_, x_vis_, y_vis_, xang_vis_, yang_vis_, rhos_, ps_, ts_ = m_vis.copy(), n_vis.copy(), mask.copy(), xi_vis.copy(), eta_vis.copy(), x_vis.copy(), y_vis.copy(), xang_vis.copy(), yang_vis.copy(), rhos.copy(), ps.copy(), ts.copy()
							
							for number_knots, deg in zip([165], [3]):
								for reg_ in [0.01]:
									regularization=False
									l_ = 0
									if number_knots > 20:
										regularization=True
										l_ = reg_
									param_img, B, knots, xs_all, img_horizon = get_img_of_params_bspline(img2,xang_vis,number_knots,deg,regularization=regularization,l=l_)
									print("KNOT SPACING:", np.min(np.diff(knots)), np.max(np.diff(knots)), np.mean(np.diff(knots)))
									L = param_img.shape[0]
									C = 1 if len(param_img.shape) == 2 else param_img.shape[2]
		
									if long_upsample_bcoeffs:
										ns_z = np.arange((z+1)*N)
										yangs = ns_z*(2*np.pi)/(N*(z+1)) - np.pi # signed angular geodesic distance from subsatellite point (latitudinal distance from SSP)
								
										xis_grid, yangs_grid = np.meshgrid(xis, yangs, indexing='ij')
										xis_all, yangs_all = xis_grid.flatten(), yangs_grid.flatten()
										fracs = [k/(z+1) for k in range(1,z+1)]
										upsample = np.zeros((L,N*(z+1),C))
										for c in range(C):
											upsample[:,:,c] = cubic_hermite(param_img[:,:,c], dya/(z+1), fracs)

										m_vis, n_vis, mask, xi_vis, eta_vis, x_vis, y_vis, xang_vis, yang_vis, rhos, ps, ts = convert_from_xi_ya(xis_all, yangs_all, N, plot=False)

									print("before upsample", "m", m_vis_.shape, "n", n_vis_.shape, "param img", param_img.shape)
									print("after upsample", "m", m_vis.shape, "n", n_vis.shape, "upsample img", upsample.shape)
									
									print("Y VALUES RELATIVE TO SSP", np.unique(np.around(yang_vis, thresh)))
									Nys = len(np.unique(np.around(yang_vis_, thresh)))

									# vectorize cexp formation
									ngrid = len(xi_vis)
									us, vs = baselines[:,0], baselines[:,1]
									us_tiled, vs_tiled = np.tile(us.reshape((nvis,1)), (1,ngrid)), np.tile(vs.reshape((nvis,1)), (1,ngrid))
									print("ngrid", ngrid, "nvis", nvis, "xi_vis", xi_vis.shape, "baselines", baselines.shape)
									xis_tiled, etas_tiled = np.tile(xi_vis.reshape((1,ngrid)), (nvis,1)), np.tile(eta_vis.reshape((1,ngrid)), (nvis,1))
									print("load weights", xi_vis.shape, yang_vis.shape, baselines.shape)
									print("Nys", Nys, "dya", dya, "dya/(z+1)", dya/(z+1))
									cexp = load_weights(xi_vis, yang_vis, baselines, dxi, dya/(z+1), dfx=decimation_factor_x, dfy=decimation_factor_y, n=1)

									# antenna product
									antpattern_matrix = assign_and_resample_antenna_pattern_matrix(ps, np.pi-ts, points_spatial, delta_xi=dxi, delta_y=dya/(z+1), 
																									seed=10, thresh=10, ablation_antenna=False, scenario_name=f'{Nx}x{Ny}rectangularframe')
									Amat = np.tile(antpattern_matrix, (nant,1)) * np.conj(np.repeat(antpattern_matrix, nant, axis=0)) # row k*nant+l has product of antenna k with conjugate of antenna l        
									# jacobian
									J = J_xi_ya(rhos, eta_vis, ts, xang_vis, yang_vis, R=R, h=h, ablation=False)
									J = np.tile(J.reshape((1,ngrid)), (nvis,1))

									# hkl
									hkl = cexp*Amat*J
									print("NAN in Amat", np.sum(np.isnan(Amat)))
									print("NAN in cexp", np.sum(np.isnan(cexp)))
									print("NAN in J", np.sum(np.isnan(J)))
									del J
									del Amat
									del cexp

									# basis functions
									Umean, U, sval, Vt = get_orthogonal_model_and_mean((np.pi-ts)*180/np.pi, angles=angles, pcs=pcs, mean_vec=mean_vec, n_components_simulation=number_components, kind='cubic', ablation=False)
									Tcs = [np.tile((U[:,c]*sval[c]).reshape((1,ngrid)), (nvis, 1)) for c in range(C)]


									print("mask.shape", mask.shape, np.sum(mask), "mask_", mask_.shape, np.sum(mask_))
									print("B", B.shape)
									print("M", M, "N", N, M*N) 
									nparam = param_img.shape[0]
									plot_ms_up, plot_ns_up = np.meshgrid(np.arange(nparam), yangs, indexing='ij')
									plot_ms, plot_ns = np.meshgrid(np.arange(nparam), np.arange(N)*2*np.pi/N-ysat, indexing='ij')
									plt.figure(figsize=(20,20))
									plt.scatter(plot_ns, plot_ms, c=param_img[:,:,0].flatten(), alpha=0.5, marker='.', linewidth=0.5)
									plt.scatter(plot_ns_up, plot_ms_up, c=upsample[:,:,0].flatten(), alpha=0.5, marker='.', linewidth=0.5)
									plt.ylabel("parameter number")
									plt.xlabel("longitude")
									plt.colorbar()
									plt.savefig("upsamplecomparison.png")
									plt.close()


									nparam = B.shape[0]
									plot_ms_up, plot_ns_up = np.meshgrid(np.arange(nparam), yangs, indexing='ij')
									plot_ms, plot_ns = np.meshgrid(np.arange(nparam), np.arange(N)*2*np.pi/N-ysat, indexing='ij')
									plt.figure(figsize=(20,20))
									plt.scatter(plot_ns, plot_ms, c=clip(B@param_img[:,:,0]).flatten(), alpha=0.5, marker='.', linewidth=0.5)
									plt.ylabel("bt parameter sample")
									plt.xlabel("longitude")
									plt.colorbar()
									plt.savefig("originalfit-reconstruction.png")
									plt.close()


									plt.figure(figsize=(20,20))
									plt.scatter(plot_ns, plot_ms, c=clip(B@param_img[:,:,0]).flatten(), alpha=0.5, marker='.', linewidth=0.5)
									plt.scatter(plot_ns_up, plot_ms_up, c=clip(B@upsample[:,:,0]).flatten(), alpha=0.5, marker='.', linewidth=0.5)
									plt.ylabel("bt parameter sample")
									plt.xlabel("longitude")
									plt.colorbar()
									plt.savefig("interp-upsamplecomparison.png")
									plt.close()


									u,sval,vh = np.linalg.svd(B @ np.linalg.pinv(B))
									plt.figure()
									plt.plot(1+np.arange(len(sval)), sval)
									plt.xlabel("Singular value number")
									plt.ylabel(r"singular value")
									plt.savefig(f"reconstruction_svs_dfx_{decimation_factor_x}_dfy_{decimation_factor_y}_l_{l_}_reg_{regularization}_deg_{deg}_nknots_{number_knots}.png")
									plt.close()


									u,sval,vh = np.linalg.svd(B)
									plt.figure()
									plt.plot(1+np.arange(len(sval)), np.log10(sval))
									plt.xlabel("Singular value number")
									plt.ylabel(r"$\log$ singular value")
									plt.savefig(f"B_svs_dfx_{decimation_factor_x}_dfy_{decimation_factor_y}_l_{l_}_reg_{regularization}_deg_{deg}_nknots_{number_knots}.png")
									plt.close()


									plt.figure()
									plt.axis('off')
									if len(img_horizon.shape) == 2:
										plt.imshow(img_horizon)
										plt.title("Image of BT params (visible strip)")
										fname_BT_params = f"visible_strip_dfx_{decimation_factor_x}_dfy_{decimation_factor_y}_component_1_out_of_1.png"
									elif len(img2.shape) == 3:
										for c in range(img_horizon.shape[2]):
											plt.imshow(img_horizon[:,:,c])
											plt.title(f"Image of BT params (visible strip), param no {c+1}")
											fname_BT_params = f"visible_strip_dfx_{decimation_factor_x}_dfy_{decimation_factor_y}_component_{c}_out_of_{img2.shape[2]}.png"
									plt.colorbar()	
									plt.savefig(fname_BT_params)
									plt.close()

									if len(param_img.shape) == 2:
										plt.figure()
										plt.imshow(param_img)
										plt.axis('off')
										plt.title("Image of spline weights on BT params")
										plt.colorbar()
										plt.savefig(f"bspline_weights_dfx_{decimation_factor_x}_dfy_{decimation_factor_y}_reg_{regularization}_lambda_{l_}_nknots_{number_knots}_deg_{deg}_component_1_out_of_1.png")
										plt.close()
								
										plt.figure()
										plt.imshow(B@param_img)
										plt.colorbar()
										plt.title("Recovered image of BT params")
										plt.savefig(f"bspline_recovered_img_dfx_{decimation_factor_x}_dfy_{decimation_factor_y}_reg_{regularization}_lambda_{l_}_nknots_{number_knots}_deg_{deg}_component_1_out_of_1.png")
										plt.close()

										plt.figure()
										plt.imshow(clip(np.abs(B@param_img-img_horizon)))
										plt.colorbar()
										plt.title("Absolute spline interpolation error (clipped at 1)")
										plt.savefig(f"bspline_err_img_dfx_{decimation_factor_x}_dfy_{decimation_factor_y}_reg_{regularization}_lambda_{l_}_nknots_{number_knots}_deg_{deg}_component_1_out_of_1.png")
										plt.close()

										plt.figure()
										plt.imshow(np.log10(abs(B@param_img-img_horizon)))
										plt.colorbar()
										plt.title("Log absolute spline interpolation error")
										plt.savefig(f"log_bspline_err_img_dfx_{decimation_factor_x}_dfy_{decimation_factor_y}_reg_{regularization}_lambda_{l_}_nknots_{number_knots}_deg_{deg}_component_1_out_of_1.png")
										plt.close()

									elif len(param_img.shape) ==3:
										for c in range(param_img.shape[2]):
											plt.figure()
											plt.imshow(param_img[:,:,c])
											plt.axis('off')
											plt.title(f"Image of col spline weights, param no {c+1}")
											plt.colorbar()
											plt.savefig(f"bspline_weights_dfx_{decimation_factor_x}_dfy_{decimation_factor_y}_reg_{regularization}_lambda_{l_}_nknots_{number_knots}_deg_{deg}_component_{c}_out_of_{param_img.shape[2]}.png")
											plt.close()

											plt.figure()
											plt.imshow(clip(B@param_img[:,:,c]))
											plt.title(f"Recovered image of BT params, param no {c+1}")
											plt.colorbar()
											plt.savefig(f"bspline_recovered_img_dfx_{decimation_factor_x}_dfy_{decimation_factor_y}_reg_{regularization}_lambda_{l_}_nknots_{number_knots}_deg_{deg}_component_{c}_out_of_{param_img.shape[2]}.png")
											plt.close()

											plt.figure()
											print("param_img", param_img.shape, "B", B.shape, "img_horizon", img_horizon.shape)
											plt.imshow(clip(np.abs(multiply_channels_indep(B,param_img)-img_horizon)[:,:,0]))
											plt.colorbar()
											plt.title("Absolute spline interpolation error")
											plt.savefig(f"bspline_err_img_dfx_{decimation_factor_x}_dfy_{decimation_factor_y}_reg_{regularization}_lambda_{l_}_nknots_{number_knots}_deg_{deg}_component_{c}_out_of_{param_img.shape[2]}.png")
											plt.close()

											plt.figure()
											print("param_img", param_img.shape, "B", B.shape, "img_horizon", img_horizon.shape)
											plt.imshow(np.log10(np.abs(multiply_channels_indep(B,param_img)-img_horizon)[:,:,0]))
											plt.colorbar()
											plt.title("Log absolute spline interpolation error")
											plt.savefig(f"log_bspline_err_img_dfx_{decimation_factor_x}_dfy_{decimation_factor_y}_reg_{regularization}_lambda_{l_}_nknots_{number_knots}_deg_{deg}_component_{c}_out_of_{param_img.shape[2]}.png")
											plt.close()

									for Ti, Tf, s in zip([0], [param_img.shape[1]//10-1], [1]):
										add_noise = False
										fname = f"full_noise_{add_noise}_rect_Ti_{Ti}_Tf_{Tf}_stride_{s}_Ny_{Ny}_Nx_{Nx}_du_{du}_dv_{dv}_dxi_{dxi}_dya_{dya:.4f}_dfx_{decimation_factor_x}_dfy_{decimation_factor_y}_reg_{regularization}_lambda_{l_}_nknots_{number_knots}_deg_{deg}_numbcomp_{number_components}.pkl"
										vis, vtilde, vis_with_mean = get_vis(param_img, deg, knots, Ti, Tf, s,
																			 points_spatial, xang_vis, yang_vis, hkl, Umean,
																			 U, var=var, n_components_simulation=number_components, fname="vec_vis_"+fname)
										Bwidth = 20e6
										tau = 1
										if add_noise:
											print("vis with mean", vis_with_mean.shape)
											print("vis with mean 0", np.sum(np.isnan(vis_with_mean[:,0])))
											print("Mean cplx", np.mean(np.imag(vis_with_mean[dc_baseline_idxs])))
											print("Mean DC power:", np.mean(np.real(vis_with_mean[dc_baseline_idxs])))
											print("DC power standard error across baseliness:", np.std(np.real(vis[dc_baseline_idxs]))/len(dc_baseline_idxs))
											sigma = np.mean(dxi*dya*np.abs(vis_with_mean[dc_baseline_idxs]))/np.sqrt(2*Bwidth*tau)
											print("Noise variance (added independently of multiplicity of baseline):", sigma**2)
											noise = np.random.normal(0, sigma/np.sqrt(2), vis.shape) + 1j*np.random.normal(0, sigma/np.sqrt(2), vis.shape)
											vtilde += np.fft.fft(noise,axis=1)
                                        
										if save_g:
											param_est, I, Gs, param_est_full = invert_vis(vtilde,deg,knots,Ti,Tf,s,z,Nys,L,N,C,xang_vis,yang_vis,xi_vis,
																					eta_vis,dya,hkl,Tcs,n_components_inversion=number_components,
                																	thresh=10,fname="vec_inv_"+fname, save_g=save_g)
										else:
											param_est, I, param_est_full = invert_vis(vtilde,deg,knots,Ti,Tf,s,z,Nys,L,N,C,xang_vis,yang_vis,xi_vis,
											eta_vis,dya,hkl,Tcs,n_components_inversion=number_components,
											thresh=10,fname="vec_inv_"+fname, save_g=save_g)

										if save_g:
											plt.figure()
											errrs = np.array([np.linalg.norm(Gpinv @ G - np.eye(Gpinv.shape[0])) for G, Gpinv in Gs])
											plt.plot(range(len(errrs)), errrs)
											plt.title(r"$||G^{\dagger}G-I||_F^2||$ vs $\omega$")
											plt.xlabel(r"Orbital frequency $\omega$")
											plt.ylabel(r"$||G(\omega)^\dagger G(\omega) - I||_F^2$")
											plt.savefig(f"GpinvG_m_I_"+fname[:-4]+".png")
											plt.close()


										if len(param_est.shape) == 2:
											plot_with_cbar_sidebar_knots(param_est, "Image of spline weights on BT params",  f"recovered_param_img_component_1_oo_1_"+fname[:-4]+".png", show_sidebar=False)
											plot_with_cbar_sidebar_knots(clip(np.abs(param_est-param_img[:,Ti:Tf+1])), "Error image of spline weights on BT params", f"error_in_param_img_component_1_oo_1"+fname[:-4]+".png")

											plot_with_cbar_sidebar_knots(np.log10(abs(param_est-param_img[:,Ti:Tf+1])),"Log error image of spline weights on BT params", f"log_error_in_param_img_component_1_oo_1"+fname[:-4]+".png")


										elif len(param_img.shape) ==3:
											_, _, _, _, _, _ = three_part_plot(img_horizon.copy()[:,Ti:Tf+1], multiply_channels_indep(B, param_img[:,Ti:Tf+1]), r"Log projn & Reconstn BT error ($C=$" + f"{number_components}, " + r"$B\cdot\tau=$" + f"{Bwidth*tau:.1e})", r"Ground Truth & B-Spline Approximation BT curves sampled at [2.5$^\circ$ | 22.5$^\circ$ | 57.5$^\circ$]", "Error in BT (K)", "reconstruct_temp_err_"+fname[:-4]+".png")

											temp2p5_sim, temp2p5_rec, temp22p5_sim, temp22p5_rec, temp57p5_sim, temp57p5_rec = three_part_plot(multiply_channels_indep(B, param_img[:,Ti:Tf+1]), multiply_channels_indep(B, param_est), r"Log projn & Reconstn BT error ($C=$" + f"{number_components}, " + r"$B\cdot\tau=$" + f"{Bwidth*tau:.1e})", r"Ground Truth & B-Spline Approximation BT curves sampled at [2.5$^\circ$ | 22.5$^\circ$ | 57.5$^\circ$]", "Error in BT (K)", "reconstruct_temp_err_"+fname[:-4]+".png")

											min_57, max_57 = np.around(np.min(temp57p5_sim))-1, np.around(np.max(temp57p5_sim))+1

											plot_with_cbar_sidebar_knots(temp57p5_sim, r"Simulated BT (K), observed at $57.5^\circ$", "simulated_temp_57p5"+fname[:-4]+".png",
																			ctitle="BT (K)", figsize=(20,8), show_sidebar=False)

											plot_with_cbar_sidebar_knots(temp57p5_rec, r"Recovered BT (K), observed at $57.5^\circ$", "recovered_temp_57p5"+fname[:-4]+".png",
                                                                            ctitle="BT (K)", figsize=(20,8), clim=[min_57, max_57], show_sidebar=False)

											plot_with_cbar_sidebar_knots(clip(np.abs(temp57p5_rec-temp57p5_sim),0,10), r"BT error (K), observed at $57.5^\circ$", "error_temp_57p5"+fname[:-4]+".png",
                                                                            ctitle="BT (K)", figsize=(20,8))

											min_22, max_22 = np.around(np.min(temp22p5_sim))-1, np.around(np.max(temp22p5_sim))+1

											plot_with_cbar_sidebar_knots(temp22p5_sim, r"Simulated BT (K), observed at $22.5^\circ$", "simulated_temp_22p5"+fname[:-4]+".png",
                                                                            ctitle="BT (K)", figsize=(20,8), show_sidebar=False)

											plot_with_cbar_sidebar_knots(temp22p5_rec, r"Recovered BT (K), observed at $22.5^\circ$", "recovered_temp_22p5"+fname[:-4]+".png",
                                                                            ctitle="BT (K)", figsize=(20,8), clim=[min_22, max_22], show_sidebar=False)

											plot_with_cbar_sidebar_knots(clip(np.abs(temp22p5_rec-temp22p5_sim),0,10), r"BT error (K), observed at $22.5^\circ$", "error_temp_22p5"+fname[:-4]+".png",
                                                                            ctitle="BT (K)", figsize=(20,8))

											min_2, max_2 = np.around(np.min(temp2p5_sim))-1, np.around(np.max(temp2p5_sim))+1


											plot_with_cbar_sidebar_knots(temp2p5_sim, r"Simulated BT (K), observed at $2.5^\circ$", "simulated_temp_2p5"+fname[:-4]+".png",
																			ctitle="BT (K)", figsize=(20,8), show_sidebar=False)

											plot_with_cbar_sidebar_knots(temp2p5_rec, r"Recovered BT (K), observed at $2.5^\circ$", "recovered_temp_2p5"+fname[:-4]+".png",
																			ctitle="BT (K)", figsize=(20,8), clim=[min_2, max_2], show_sidebar=False)

											plot_with_cbar_sidebar_knots(clip(np.abs(temp2p5_rec-temp2p5_sim),0,10), r"BT error (K), observed at $2.5^\circ$", "error_temp_2p5"+fname[:-4]+".png",
                                                                            ctitle="BT (K)", figsize=(20,8))


											for c in range(param_img.shape[2]):
												
												plot_with_cbar_sidebar_knots(clip(param_est[:,:,c],-50,50), r"Image of B-spline weights, param no. "+f"{c+1}", 
												"bspline_param_img_"+f"component_{c+1}_out_of_{param_img.shape[2]}.png", 
												ctitle="(clipped)", figsize=(20,8), show_sidebar=False)

												plot_with_cbar_sidebar_knots(clip(np.abs(param_est[:,:,c]-param_img[:,Ti:Tf+1,c]),0,10), f"Absolute error of B-spline weights, param no. {c+1}",
												"error_in_bspline_param_img_"+f"component_{c+1}_out_of_{param_img.shape[2]}.png",
												ctitle="(clipped)", figsize=(20,8))

												plot_with_cbar_sidebar_knots(np.log10(np.abs(param_est[:,:,c]-param_img[:,Ti:Tf+1,c])), f"Log B-spline weight error, param no. {c+1}",
                                                "log_error_in_bspline_param_img_"+f"component_{c+1}_out_of_{param_img.shape[2]}.png",
                                                ctitle="(clipped)", figsize=(20,8))

												plot_with_cbar_sidebar_knots(clip(B@param_est[:,:,c]), r"Image of BT param weights, param no. "+f"{c+1}",
                                                "resampled_bt_param_img_"+f"component_{c+1}_out_of_{param_img.shape[2]}.png",
                                                ctitle="(clipped)", figsize=(20,8), show_sidebar=False)

												plot_with_cbar_sidebar_knots(clip(np.abs(B @ param_est[:,:,c])-img_horizon[:,Ti:Tf+1,c]), f"Error in interpolated BT params, param no {c+1}", f"error_in_interpolated_img_"+fname[:-4]+f"_component_{c+1}_out_of_{param_img.shape[2]}.png", ctitle="(clipped)", figsize=(20,8), show_sidebar=False)
			except Exception as e:
				print("Error:", e)
				print(traceback.print_exc())
				print(traceback.format_exc())
				print("tb", sys.exc_info()[2])
				continue
