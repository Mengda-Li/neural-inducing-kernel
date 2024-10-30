import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.image as mpimg
plt.rc('axes', axisbelow=True)
from scipy.interpolate import splrep, BSpline, interp1d, RectBivariateSpline
from scipy.io import loadmat

# earth model parameters
R=6371
h=687

# hexagon sampling

def get_hexagon(Nt, dx):
    X = np.zeros((Nt, Nt))
    Y = np.zeros((Nt, Nt))
    I = np.zeros((Nt, Nt))
    for i in range(Nt):
        for j in range(Nt):
            if i+j > Nt:
                ii = i-Nt
                jj = j-Nt
                if (i-2*j) > 0:
                    ii = i-Nt
                    jj = j
                if (2*i-j) <= 0:
                    ii = i
                    jj = j-Nt
            else:
                ii = i
                jj = j
                if (2*i-j-Nt) > 0:
                    ii = i-Nt
                    jj = j
                if (i-2*j+Nt) <= 0:
                    ii = i
                    jj = j-Nt
            X[i,j] = (0.5*np.sqrt(3)*jj)*dx
            Y[i,j] = (ii - 0.5*jj)*dx
            I[i,j] = i*Nt+j
    return X, Y, I


# interpolation helpers

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

def assign_and_resample_antenna_pattern_matrix(ps_vis, incidences, points_spatial, seed=10,thresh=10, ablation_antenna=False, scenario_name=''):
    SM_mat = loadmat('LICEFavpA.mat')
    fname = f'antpattern_matrix_for_{scenario_name}.pkl'
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

# angle matching helpers
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


# bspline helpers

def multiply_channels_indep(A,img):
    # A, a 2D KxM array
    # img, a 3D MxNxC array
    # returns KxNxC array by multiplying A by each channel of img, then writing this product back into the channels
    return (A@img.transpose((2,0,1))).transpose((1,2,0))

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

# Jacobian

def J_xi_ya(rho, eta, theta, xa, ya, R=R, h=h, ablation=False):
    if ablation:
        return np.ones(rhos.shape)
    fac = R*np.cos(xa)
    num = np.abs(fac*(rho*np.cos(ya)-eta*(R+h)*np.sin(ya)))
    den = rho**2
    detadya = num/den
    return detadya/(-1*np.cos(theta))

def J_xi_eta(theta):
    return -1*np.cos(theta)**(-1)

# coordinate helpers

def xi_eta_to_phi_theta(xis, etas):
    
    sintheta = np.sqrt(xis**2 + etas**2)
    thetas = np.pi - np.arcsin(sintheta)
    
    print("SIN THETA", np.min(sintheta), np.max(sintheta))
    print("THETAS", np.min(thetas), np.max(thetas))

    phis = np.ones(xis.shape)*np.where(etas>=0, np.pi/2, 3*np.pi/2)

    q1 = np.logical_and(etas > 0, xis < 0)
    q2 = np.logical_and(etas > 0, xis > 0)
    q3 = np.logical_and(etas < 0, xis > 0)
    q4 = np.logical_and(etas < 0, xis < 0)

    phis[np.where(q1)] = np.arctan(np.divide(etas, -xis, where=q1))[q1]
    phis[np.where(q2)] = np.pi/2+np.arctan(np.divide(xis, etas, where=q2))[q2]
    phis[np.where(q3)] = np.arctan(np.divide(etas, -xis, where=q3))[q3] + np.pi 
    phis[np.where(q4)] = 3*np.pi/2 + np.arctan(np.divide(xis, etas, where=q4))[q4]
    
    yplus = np.logical_and(xis>0, etas==0)
    yminus = np.logical_and(xis<0, etas==0)
    phis[yplus] = np.pi
    phis[yminus] = 0
    
    return phis, thetas

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

def convert_from_phi_theta(phi,theta,M,N,R=R,h=h,ysat=R*np.pi):
    print("PHI", phi.shape, "THETA", theta.shape)
    visible_mask = theta >= np.pi - np.arcsin(R/(R+h))

    phis_visible = phi[visible_mask]
    thetas_visible = theta[visible_mask]

    print("PHI VIS", phis_visible.shape, thetas_visible.shape)

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

# visibilities 

def get_vis(param_img, deg, knots, Ti, Tf, s,
            points_spatial, xang_vis,
            yang_vis, hkl, Umean, U, var,
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


# smos helpers

def smos_layout_in_wavelengths(L, d=0.875):
    arm1 = [(i*d*np.cos(np.pi/3),d*i*np.sin(np.pi/3)) for i in range(-3,L) if i != 0 and i!=-2 ]
    arm2 = [(i*d*np.cos(np.pi), i*d*np.sin(np.pi)) for i in range(-3,L) if i !=0 and i!=-2]
    arm3 = [(i*d*np.cos(-np.pi/3), i*d*np.sin(-np.pi/3)) for i in range(-3,L) if i!=0 and i!=-2]
    return np.array(arm1+arm2+arm3)

def smos_baselines_in_wavelengths(L,d=0.875):
    ants = smos_layout_in_wavelengths(L,d)
    return np.vstack(ants[:,None]-ants), ants


# load earth img
decimation_factor_x, decimation_factor_y, number_components = 4, 2, 1
img = mpimg.imread('ultrabluemarble.png')
img2 = img.copy()[::decimation_factor_x, ::decimation_factor_y, :number_components]*255
M,N = img2.shape[:2]
C = 1 if len(img2.shape) == 2 else img2.shape[2]


# get smos instrument baselines and antenna positions

baselines, points_spatial = smos_baselines_in_wavelengths(23)
nvis = len(baselines)
nant = len(points_spatial)

# get xi-eta sampling grid

visibilities = {}

N1 = 1024
N2 = 1400

for Nt in [N1, N2]:
    coords = np.zeros((Nt**2, 2))
    dx = 2.0/Nt
    X, Y, I = get_hexagon(Nt, dx)
    xis_all, etas_all = X.flatten(), Y.flatten()
    keep = xis_all**2 + etas_all**2 <= 1
    xis_all, etas_all = xis_all[keep], etas_all[keep]
    vis_mask = xis_all**2 + etas_all**2 <= (R/(R+h))**2
    xi_vis = xis_all[vis_mask]
    eta_vis = etas_all[vis_mask]
    xis_sky = xis_all[~vis_mask]
    etas_sky = etas_all[~vis_mask]
    nsky = np.sum(~vis_mask)
    ngrid = len(xis_all)
    nvisible = len(xi_vis)

    ps, ts = xi_eta_to_phi_theta(xis_all, etas_all)
    ps_vis, ts_vis = xi_eta_to_phi_theta(xi_vis, eta_vis)
    print(np.min(ts_vis), np.max(ts_vis), np.pi-np.arcsin(R/(R+h)), np.pi)

    #rhos = -1*(R+h)*np.cos(ts) - np.sqrt((R+h)**2*np.cos(ts)**2 - (R+h)**2 + R**2)
    m_vis, n_vis, _, _, _, _, _, _, _ = convert_from_phi_theta(ps, ts, M=M, N=N, R=R, h=h, ysat=0.5*2*np.pi*R)
    
    # vectorize cexp formation
    us, vs = baselines[:,0], baselines[:,1]
    us_tiled, vs_tiled = np.tile(us.reshape((nvis,1)), (1,nvisible)), np.tile(vs.reshape((nvis,1)), (1,nvisible))
    xis_tiled, etas_tiled = np.tile(xi_vis.reshape((1,nvisible)), (nvis,1)), np.tile(eta_vis.reshape((1,nvisible)), (nvis,1))
    cexp = np.exp(-2*np.pi*1j*(us_tiled*xis_tiled + vs_tiled*etas_tiled))


    antpattern_matrix = assign_and_resample_antenna_pattern_matrix(ps[vis_mask], np.pi-ts[vis_mask], points_spatial,
																	seed=10, thresh=10, ablation_antenna=False, scenario_name=f'SMOS_hexagon_{Nt}')
    Amat = np.tile(antpattern_matrix, (nant,1)) * np.conj(np.repeat(antpattern_matrix, nant, axis=0)) # row k*nant+l has product of antenna k with conjugate of antenna l        
	
    # jacobian
    J = J_xi_eta(ts[vis_mask])
    J = np.tile(J.reshape((1,nvisible)), (nvis,1))

	# hkl
    hkl = cexp*Amat*J

    Nvis = len(baselines)
    img_vals = interpolate_in_phi_theta_grid(img2[:,:,0], n_vis, m_vis, dt=np.pi/M, dp=2*np.pi/N)
    
    #J_sky = J_xi_eta(ts[~vis_mask])
    #J_sky = np.tile(J_sky.reshape((1,nsky)), (nvis,1))
    
    #antpattern_matrix_sky = assign_and_resample_antenna_pattern_matrix(ps[~vis_mask], np.pi-ts[~vis_mask], points_spatial,
	#																seed=10, thresh=10, ablation_antenna=False, scenario_name=f'SMOS_hexagon_sky_{Nt}')
    #Amat_sky = np.tile(antpattern_matrix_sky, (nant,1)) * np.conj(np.repeat(antpattern_matrix_sky, nant, axis=0)) # row k*nant+l has product of antenna k with conjugate of antenna l        

    #xis_tiled_sky, etas_tiled_sky = np.tile(xis_sky.reshape((1,nsky)), (nvis,1)), np.tile(etas_sky.reshape((1,nsky)), (nvis,1))
    #us_tiled_sky, vs_tiled_sky = np.tile(us.reshape((nvis,1)), (1,nsky)), np.tile(vs.reshape((nvis,1)), (1,nsky))
    #cexp_sky = np.exp(-2*np.pi*1j*(us_tiled_sky*xis_tiled_sky + vs_tiled_sky*etas_tiled_sky))
    #hkl_sky = J_sky * Amat_sky * cexp_sky
    #img_sky = 3*np.ones((nsky,))

    #print("HKL", hkl.shape, "IMG", img_vals.shape, "HKL_SKY", hkl_sky.shape, "IMG_SKY", img_sky.shape)

    visibilities[Nt] = (hkl @ img_vals)*2*np.sqrt(3)*(1/Nt)**2

print(visibilities[N1].shape, np.sum(np.isnan(visibilities[N1])))
min_ = np.min([np.min(np.real(visibilities[N1])), np.min(np.imag(visibilities[N1])), np.min(np.real(visibilities[N2])), np.min(np.imag(visibilities[N2]))])
max_ = np.max([np.max(np.real(visibilities[N1])), np.max(np.imag(visibilities[N1])), np.max(np.real(visibilities[N2])), np.max(np.imag(visibilities[N2]))])    
xs_ = np.linspace(min_, max_, 1000)
print("Baselines", baselines.shape, nvis, np.linalg.norm(baselines,axis=1).shape)
print("Visibilities", visibilities[N1].shape, visibilities[N2].shape)
plt.figure()
plt.scatter(np.real(visibilities[N1]), np.real(visibilities[N2]), label='real part', marker='.', s=2, alpha=0.5)
plt.scatter(np.imag(visibilities[N1]), np.imag(visibilities[N2]), label='imag part', marker='.', s=2, alpha=0.5)
plt.plot(xs_, xs_, '--', alpha=0.5, label='no error')
plt.xlabel(f"Visibilities, {N1}x{N1} Hexagonal Grid")
plt.ylabel(f"Visibilities, {N2}x{N2} Hexagonal Grid")
plt.legend()
plt.title(f"Visibility Errors, {N1}x{N1} and {N2}x{N2} Hexagonal Grids")
plt.savefig(f"visibilities_comp_{N1}x{N2}.png")
plt.close()

plt.figure()
plt.scatter(np.abs(baselines), np.abs(visibilities[N1]-visibilities[N2]), s=2, alpha=0.5)
plt.xlabel(f"baseline magnitude")
plt.ylabel(f"absolute error in visibility")
plt.title(f"Visibility Errors, {N1}x{N1} vs. {N2}x{N2} Hexagonal Sampling")
plt.savefig(f"visibilities_comp2_bl_{N1}x{N2}.png")
plt.close()

plt.figure()
plt.scatter(baselines[:,0], baselines[:,1], c=np.abs(visibilities[N2] - visibilities[N1]), label='absolute error in visibility', marker='.', s=2, alpha=0.5)
plt.xlabel(r"baseline, $u$ coordinate")
plt.ylabel(r"baseline, $v$ coordinate")
plt.title(f"Error in Visibilities, {N1}x{N1} Hexagonal Grid")
plt.legend()
plt.colorbar()
plt.savefig(f"visibilities_err_plt_{N1}x{N2}.png")
plt.close()


plt.figure()
plt.scatter(baselines[:,0], baselines[:,1], c=np.abs(visibilities[N2] - visibilities[N1])/np.abs(visibilities[N1]), label='absolute error in visibility (%)', marker='.', s=2, alpha=0.5)
plt.xlabel(r"baseline, $u$ coordinate")
plt.ylabel(r"baseline, $v$ coordinate")
plt.title(f"Error in Visibilities, {N1}x{N1} Hexagonal Grid")
plt.legend()
plt.colorbar()
plt.savefig(f"visibilities_err_plt_prct_{N1}x{N2}.png")
plt.close()
