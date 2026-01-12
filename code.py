import numpy as np
import matplotlib.pyplot as plt
import pypulseq as pp
from numpy.polynomial.legendre import leggauss
from numba import njit, prange

#%% Functions

def excitation (system, type='sinc',fa_ex=90,slice_thickness=30e-3, rf_dur=3e-3, tbp=3, apodization=0.5, n_bands=3, band_sep=20):
    """
    Pypulseq event for excitation, taken from 
    https://github.com/pulseq/MR-Physics-with-Pulseq/blob/main/tutorials/02_rf_pulses/notebooks/se2d_sliceprofile_solution.ipynb 
    by Prof. Dr. Tony Stöcker
    """
    # sinc pulse
    if type == 'sinc':
            rf1,gz1,gzr1 = pp.make_sinc_pulse(
                flip_angle=fa_ex * np.pi / 180, phase_offset=90 * np.pi / 180, duration=rf_dur,
                slice_thickness=slice_thickness, apodization=apodization, time_bw_product=tbp,
                system=system, return_gz=True)
    # sinc pulse
    elif type == 'gauss':
            rf1,gz1,gzr1 = pp.make_gauss_pulse(
            flip_angle=fa_ex * np.pi / 180, phase_offset=90 * np.pi / 180, duration=rf_dur,
            slice_thickness=slice_thickness, apodization=apodization, time_bw_product=tbp,
            system=system, return_gz=True)
    # SLR pulse using the sigpy.rf interface in pypulseq v1.4.0
    elif type == 'slr':
            sigpy_cfg = pp.SigpyPulseOpts(pulse_type='slr', ptype='ex')
            rf1, gz1, gzr1, _ = pp.sigpy_n_seq(
            flip_angle=fa_ex * np.pi / 180, phase_offset=90 * np.pi / 180, duration=rf_dur,
            slice_thickness=slice_thickness, time_bw_product=tbp,
            system=system, return_gz=True, pulse_cfg=sigpy_cfg, plot=False )
    # SMS pulse using the sigpy.rf interface in pypulseq v1.4.0
    elif type == 'sms':
            sigpy_cfg = pp.SigpyPulseOpts(pulse_type='sms', ptype='ex', n_bands=n_bands, band_sep=band_sep)
            rf1, gz1, gzr1, _ = pp.sigpy_n_seq(
                flip_angle=fa_ex * np.pi / 180, phase_offset=90 * np.pi / 180, duration=rf_dur,
                slice_thickness=slice_thickness, time_bw_product=tbp,
                system=system, return_gz=True, pulse_cfg=sigpy_cfg, plot=False)
            rf1.signal = 9.5e-3*rf1.signal # ??? the scaling of the SMS pulse was far too high ??? reduced by trial-and-error 
    else:
            raise Exception('error in excitation() - unknown pulse type: '+type)
        
    return rf1,gz1,gzr1

def gradient_waveform_from_event(gz, grad_raster, rf_time):
    """
    Build the physical trapezoid gradient waveform from a pypulseq gz-event
    and resample it onto the RF time grid rf_time.
    Output is in Hz/m.
    """

    #  trapezoid on native gradient raster 
    nrise = max(1, int(round(gz.rise_time / grad_raster)))
    nflat = max(1, int(round(gz.flat_time / grad_raster)))
    nfall = max(1, int(round(gz.fall_time / grad_raster)))

    ramp_up   = np.linspace(0, gz.amplitude, nrise, endpoint=False)
    flat      = np.full(nflat, gz.amplitude)
    ramp_down = np.linspace(gz.amplitude, 0, nfall, endpoint=True)

    grad_native = np.concatenate([ramp_up, flat, ramp_down])

    # delay
    if gz.delay > 0:
        ndelay = int(round(gz.delay / grad_raster))
        grad_native = np.concatenate([np.zeros(ndelay), grad_native])

    # time grid of gradient
    gtime = np.arange(len(grad_native)) * grad_raster

    # resample to RF time grid 
    grad_rf = np.interp(
        rf_time, 
        gtime, 
        grad_native,
        left=0.0,
        right=0.0
    )

    return grad_rf   

def cauchy_quadrature_nodes_weights(N):
    """
    Helper function: precomputes nodes for Gauss-Legendre quadrature
    """
    
    x, w = leggauss(N)       # nots / weights on [-1,1]
    theta = 0.5 * np.pi * x  # [-π/2, π/2]
    dw_unit = np.tan(theta)  # standard cauchy knots

    # weight for ∫ f(δ) p(δ) dδ with δ = tan θ
    # gives w_cauchy = w / 2
    w_cauchy = 0.5 * w

    return dw_unit.astype(np.float64), w_cauchy.astype(np.float64)

@njit
def simulation_isochromat(time     # in s
                          , w1     # in rad/s
                          , dw_t   # in rad/s
                          , R1     # in 1/s
                          , R2     # in 1/s
                          , M0):
    """
    Bloch simulation for one isochromat, extension of the Cauchy-Klein simulation from  
    # https://github.com/pulseq/MR-Physics-with-Pulseq/blob/main/tutorials/02_rf_pulses/notebooks/se2d_sliceprofile_solution.ipynb 
    # by Prof. Dr. Tony Stöcker
    """
    dt = time[1] - time[0]
    E1 = np.exp(-dt * R1)
    E2 = np.exp(-dt * R2)

    Mx, My, Mz = M0[0], M0[1], M0[2]
    outx = np.zeros(len(time))
    outy = np.zeros(len(time))
    outz = np.zeros(len(time))

    # store initial condition explicitly
    outx[0] = Mx
    outy[0] = My
    outz[0] = Mz
    
    for j in range(1, len(time)):
        w = dw_t[j]
        s = w1[j]
        Bx, By, Bz = np.real(s), np.imag(s), w
    
        omega = np.sqrt(Bx*Bx + By*By + Bz*Bz)
        
        if omega > 0:
            nx, ny, nz = Bx/omega, By/omega, Bz/omega
            phi = -omega * dt
    
            c  = np.cos(phi/2)
            s2 = np.sin(phi/2)
            ar = c
            ai = -nz * s2
            br = ny * s2
            bi = -nx * s2
    
            Mx_rot = ((ar*ar - ai*ai + bi*bi - br*br)*Mx +
                      (2*ai*ar - 2*bi*br)*My +
                      (2*ai*bi + 2*ar*br)*Mz)
    
            My_rot = ((-2*ai*ar - 2*bi*br)*Mx +
                      (ar*ar - ai*ai - bi*bi + br*br)*My +
                      (-2*ai*br + 2*ar*bi)*Mz)
    
            Mz_rot = ((2*ai*bi - 2*ar*br)*Mx +
                      (-2*ai*br - 2*ar*bi)*My +
                      (ai*ai + ar*ar - bi*bi - br*br)*Mz)
    
            Mx, My, Mz = Mx_rot, My_rot, Mz_rot
    
        # relaxation
        Mx *= E2
        My *= E2
        Mz = 1.0 - (1.0 - Mz)*E1
    
        outx[j] = Mx
        outy[j] = My
        outz[j] = Mz

    return outx, outy, outz

@njit
def simulation_t2dash(time, w1, grad, z, dw_unit, w_cauchy, R1=1, R2=3, R2dash=2, dw0=0.0, M0=(1,0,0)):
    """
    Ensemble-average over Cauchy-distributed off-resonances to simulate static dephasing due to field inhomogeneities.
    """

    N  = dw_unit.shape[0]
    nt = time.shape[0]

    MT_mean = np.zeros(nt, dtype=np.complex128)
    MZ_mean = np.zeros(nt)

    for k in range(N):
        # static off-resonance für this quadrature point
        dw_static = R2dash * dw_unit[k] + dw0
        
        # full time dependent detuning
        dw_t = grad * z * 2*np.pi + dw_static

        # cayley rotation
        Mx, My, Mz = simulation_isochromat(time, w1, dw_t, R1,R2, M0)

        w = w_cauchy[k]

        for j in range(nt):
            MT_mean[j] += w * (Mx[j] + 1j*My[j])
            MZ_mean[j] += w * Mz[j]

    return MT_mean, MZ_mean

@njit(parallel=True)
def simulation_slice_selection(z_positions, time, w1, grad, dw_unit, w_cauchy, R1, R2, R2dash, M0=(0,0,1)):
    """
    Application to slice selection; 
    adaptable to other scenarios with external off-resonance the Cauchy-distribution is centered around, such as CEST-sweeps.
    """
    
    nf = z_positions.shape[0]
    nt = time.shape[0]

    Mxy_t = np.zeros((nf, nt), dtype=np.complex128)
    Mz_t  = np.zeros((nf, nt))

    for i in prange(nf):
        z = z_positions[i]

        MT_mean, MZ_mean = simulation_t2dash(
            time, w1, grad, z,
            dw_unit, w_cauchy,
            R1=R1,R2=R2,R2dash=R2dash,
            dw0=0.0, M0=M0
        )

        for j in range(nt):
            Mxy_t[i, j] = MT_mean[j]
            Mz_t[i, j]  = MZ_mean[j]

    return Mxy_t, Mz_t

#%% Usage example slice selection: comparison of full simualtion to hard pulse approximation (=hard-pulse + free precession/decay after peak)

# --- system
sys = pp.Opts(
    max_grad=28, grad_unit="mT/m",
    max_slew=150, slew_unit="T/m/s",
    rf_ringdown_time=20e-6, rf_dead_time=100e-6,
    adc_dead_time=20e-6,
    grad_raster_time=20e-6,
    rf_raster_time=1e-5
)

# --- pulse / slice
fa_ex = 90
excitation_type = "gauss"
slice_thickness = 5e-3
tbp = 4
apodization = 0.5
rf_dur = 8e-3                       

z_positions = np.linspace(-5e-3, 5e-3, 101)

# --- relaxation / inhomogeneity
R1 = 0.0
R2 = 0.0
R2dash = 1/0.1588

# --- Cauchy quadrature for full T2'
N_gl = 200
dw_unit_gl, w_cauchy = cauchy_quadrature_nodes_weights(N_gl)
dw_unit_ref = np.zeros(1)
w_cauchy_ref = np.ones(1)

# --- build RF + slice gradient
rf1, gz1, gzr1 = excitation(
    sys, type=excitation_type, rf_dur=rf_dur, fa_ex=fa_ex,
    slice_thickness=slice_thickness, tbp=tbp, apodization=apodization
)

time = rf1.t
grad = gradient_waveform_from_event(gz1, sys.grad_raster_time, time)
w1 = rf1.signal * 2*np.pi

# --- (A) full T2' during RF (ensemble)
Mxy_full, Mz_full = simulation_slice_selection(
    z_positions, time, w1, grad,
    dw_unit_gl, w_cauchy,
    R1, R2, R2dash,
    M0=(0,0,1)
)

# --- (B) T2* during RF (single isochromat with R2+R2')
Mxy_t2s, Mz_t2s = simulation_slice_selection(
    z_positions, time, w1, grad,
    dw_unit_ref, w_cauchy_ref,
    R1, R2+R2dash, 0.0,
    M0=(0,0,1)
)

# --- (C) no-rel during RF (for hard-pulse start value)
Mxy_nr, Mz_nr = simulation_slice_selection(
    z_positions, time, w1, grad,
    dw_unit_ref, w_cauchy_ref,
    0.0, 0.0, 0.0,
    M0=(0,0,1)
)

# --- hard pulse: take complex state at pulse end, back-rotate phase to j0, then relax/precess from j0

# indices
j0 = int(np.argmax(np.abs(w1)))     # instantaneous pulse time
j_end = -1                          # pulse end

t0 = time[j0]
t_end = time[j_end]
dt_back = t_end - t0

t_free = time - t0
t_free[t_free < 0] = 0.0

# post-pulse off-resonance model for hard pulse
Delta_free = 2*np.pi * grad[j0] * z_positions
Mxy_end = Mxy_nr[:, j_end]          # complex
Mz_end  = Mz_nr[:, j_end]

Mxy0 = Mxy_end * np.exp(+1j * Delta_free * dt_back)
Mz0  = Mz_end

Mxy_hp = np.zeros((len(z_positions), len(time)), dtype=complex)
Mz_hp  = np.ones((len(z_positions), len(time)), dtype=float)

Mxy_hp[:, j0:] = (
    Mxy0[:, None]
    * np.exp(-(R2 + R2dash) * t_free[None, j0:])
    * np.exp(-1j * Delta_free[:, None] * t_free[None, j0:])
)
Mz_hp[:, j0:] = (Mz0[:, None] - 1.0) * np.exp(-R1 * t_free[None, j0:]) + 1.0

# --- slice plots at pulse end
z_mm = z_positions * 1e3

plt.figure(figsize=(7,5))
plt.plot(z_mm, np.abs(Mxy_full[:, j_end]), lw=2, label="full (T2′ ensemble)")
plt.plot(z_mm, np.abs(Mxy_t2s[:,  j_end]), lw=2, ls="--", label="T2* during RF")
plt.plot(z_mm, np.abs(Mxy_hp[:,   j_end]), lw=2, ls="-", label="hard pulse")
plt.xlabel("z (mm)")
plt.ylabel(r"$|M_{xy}|$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
plt.plot(z_mm, Mz_full[:, j_end], lw=2, label="full (T2′ ensemble)")
plt.plot(z_mm, Mz_t2s[:,  j_end], lw=2, ls="--", label="T2* during RF")
plt.plot(z_mm, Mz_hp[:,   j_end], lw=2, ls="-", label="hard pulse")
plt.xlabel("z (mm)")
plt.ylabel(r"$M_z$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
