"""
================================================================================
LABORATORIO PYTHON 13: DOS MANERAS DE OBSERVAR EL EFECTO TÚNEL (V3 CORREGIDO)
================================================================================
Materia: Mecánica Cuántica
Tema: Efecto Túnel en Barrera Rectangular 1D
Modos: Modo 1 (Estado Estacionario) y Modo 2 (Paquete de Ondas Dinámico)
Paleta: Viridis Completo (Estética Unificada Teal/Verde/Amarillo/Morado)
Correcciones:
 1. Dominio ampliado Lx = 64.0 (Nx = 2048) para evitar wrapping periódico de FFT.
 2. Transmisión espectral T_k continua en E_pos = V0 mediante np.isclose y valor exacto del umbral.
 3. Botones Etapa 1 a 4 calculan y acumulan la curva continua en el historial.
 4. Control de Tiempo activo con Slider 'Tiempo t' y evolución fluida.
================================================================================
"""

import sys
import os
import numpy as np
import sympy as sp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.fft import fft, ifft, fftfreq

# Forzado de Backend interactivo para ejecuciones en terminal / VS Code
if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] != '--test'):
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass


def run_sympy_analysis():
    """
    Análisis simbólico con SymPy del efecto túnel en una barrera rectangular 1D:
    Coeficientes T(E) sub/superumbral, punto crítico T(V0), WKB opaco y resonancias.
    """
    print("=" * 80)
    print(" 1. ANÁLISIS SIMBÓLICO CON SYMPY - LABORATORIO 13: EFECTO TÚNEL")
    print("=" * 80)

    E, V0, a, hbar, m = sp.symbols('E V0 a hbar m', real=True, positive=True)

    # 1. Parámetros k, kappa y q
    k = sp.sqrt(2 * m * E) / hbar
    kappa = sp.sqrt(2 * m * (V0 - E)) / hbar
    q = sp.sqrt(2 * m * (E - V0)) / hbar

    # 2. Transmisión subumbral (E < V0)
    beta2_sub = (V0**2) / (4 * E * (V0 - E))
    T_sub = 1 / (1 + beta2_sub * sp.sinh(kappa * a)**2)

    print("\n• Coeficiente de Transmisión Subumbral T(E < V0):")
    sp.pprint(sp.Eq(sp.Symbol('T_sub'), sp.simplify(T_sub)))

    # 3. Transmisión superumbral (E > V0)
    beta2_sup = (V0**2) / (4 * E * (E - V0))
    T_sup = 1 / (1 + beta2_sup * sp.sin(q * a)**2)

    print("\n• Coeficiente de Transmisión Superumbral T(E > V0):")
    sp.pprint(sp.Eq(sp.Symbol('T_sup'), sp.simplify(T_sup)))

    # 4. Límite exacto en el umbral (E -> V0)
    T_umbral = 1 / (1 + (m * V0 * a**2) / (2 * hbar**2))
    print("\n• Coeficiente de Transmisión Exacto en el Umbral T(E = V0):")
    sp.pprint(sp.Eq(sp.Symbol('T(E=V0)'), T_umbral))

    # 5. Límite de Barrera Opaca (kappa * a >> 1)
    T_opaca = (16 * E * (V0 - E) / (V0**2)) * sp.exp(-2 * kappa * a)
    print("\n• Aproximación de Barrera Opaca T_opaca (kappa*a >> 1):")
    sp.pprint(sp.Eq(sp.Symbol('T_opaca'), T_opaca))

    # 6. Energías de Resonancia de Transmisión Perfecta (qa = n * pi)
    n_res = sp.Symbol('n', integer=True, positive=True)
    E_res = V0 + (n_res**2 * sp.pi**2 * hbar**2) / (2 * m * a**2)
    print("\n• Energías de Resonancia (T = 1 para E > V0):")
    sp.pprint(sp.Eq(sp.Symbol('E_res'), E_res))
    print("=" * 80 + "\n")


def compute_stationary_state(E_val, V0_val, a_val, m_val=1.0, hbar_val=1.0, x_grid=None):
    """
    Parte A y B: Calcula la solución estacionaria exacta por tramos ψ(x) para onda incidente A=1.
    Región I (x < 0), Región II (0 <= x <= a), Región III (x > a).
    """
    if x_grid is None:
        x_grid = np.linspace(-12.0, 12.0 + a_val, 1500)

    A = 1.0
    dx = x_grid[1] - x_grid[0]

    if E_val <= 1e-6:
        E_val = 1e-6

    k = np.sqrt(2.0 * m_val * E_val) / hbar_val

    if E_val < V0_val:
        # Régimen Subumbral (Túnel)
        kappa = np.sqrt(2.0 * m_val * (V0_val - E_val)) / hbar_val
        beta = (k**2 - kappa**2) / (2.0 * k * kappa)
        D_den = np.cosh(kappa * a_val) - 1j * beta * np.sinh(kappa * a_val)

        r = -1j * ((k**2 + kappa**2) / (2.0 * k * kappa)) * np.sinh(kappa * a_val) / D_den
        t_amp = np.exp(-1j * k * a_val) / D_den

        psi = np.zeros_like(x_grid, dtype=complex)

        # Región I: x < 0
        mask1 = x_grid < 0.0
        psi[mask1] = A * np.exp(1j * k * x_grid[mask1]) + A * r * np.exp(-1j * k * x_grid[mask1])

        # Región II: 0 <= x <= a
        mask2 = (x_grid >= 0.0) & (x_grid <= a_val)
        x2 = x_grid[mask2]
        psi[mask2] = (A / D_den) * (np.cosh(kappa * (a_val - x2)) - 1j * (k / kappa) * np.sinh(kappa * (a_val - x2)))

        # Región III: x > a
        mask3 = x_grid > a_val
        psi[mask3] = A * t_amp * np.exp(1j * k * (x_grid[mask3] - a_val))

        T_exact = 1.0 / (np.cosh(kappa * a_val)**2 + ((V0_val**2) / (4.0 * E_val * (V0_val - E_val))) * np.sinh(kappa * a_val)**2)
        R_exact = 1.0 - T_exact

    elif np.isclose(E_val, V0_val, atol=1e-5):
        # Punto exacto del Umbral E = V0
        D_umbral = 1.0 - 0.5 * 1j * k * a_val
        r = (-0.5 * 1j * k * a_val) / D_umbral
        t_amp = np.exp(-1j * k * a_val) / D_umbral

        psi = np.zeros_like(x_grid, dtype=complex)
        mask1 = x_grid < 0.0
        psi[mask1] = A * np.exp(1j * k * x_grid[mask1]) + A * r * np.exp(-1j * k * x_grid[mask1])

        mask2 = (x_grid >= 0.0) & (x_grid <= a_val)
        x2 = x_grid[mask2]
        psi[mask2] = (A * t_amp * np.exp(1j * k * a_val)) * (1.0 + 1j * k * (x2 - a_val))

        mask3 = x_grid > a_val
        psi[mask3] = A * t_amp * np.exp(1j * k * (x_grid[mask3] - a_val))

        T_exact = 1.0 / (1.0 + (m_val * V0_val * a_val**2) / (2.0 * hbar_val**2))
        R_exact = 1.0 - T_exact

    else:
        # Régimen Superumbral (E > V0)
        q = np.sqrt(2.0 * m_val * (E_val - V0_val)) / hbar_val
        D_sup = np.cos(q * a_val) - 0.5 * 1j * ((k**2 + q**2) / (k * q)) * np.sin(q * a_val)

        r = 0.5 * 1j * ((q**2 - k**2) / (k * q)) * np.sin(q * a_val) / D_sup
        t_amp = np.exp(-1j * k * a_val) / D_sup

        psi = np.zeros_like(x_grid, dtype=complex)
        mask1 = x_grid < 0.0
        psi[mask1] = A * np.exp(1j * k * x_grid[mask1]) + A * r * np.exp(-1j * k * x_grid[mask1])

        mask2 = (x_grid >= 0.0) & (x_grid <= a_val)
        x2 = x_grid[mask2]
        psi[mask2] = (A / D_sup) * (np.cos(q * (a_val - x2)) + 1j * (k / q) * np.sin(q * (a_val - x2)))

        mask3 = x_grid > a_val
        psi[mask3] = A * t_amp * np.exp(1j * k * (x_grid[mask3] - a_val))

        T_exact = 1.0 / (1.0 + ((V0_val**2) / (4.0 * E_val * (E_val - V0_val))) * np.sin(q * a_val)**2)
        R_exact = 1.0 - T_exact

    # Corriente de probabilidad j(x) = (hbar/m) * Im(psi* dpsi/dx)
    dpsi = np.gradient(psi, dx)
    j_net = (hbar_val / m_val) * np.imag(np.conj(psi) * dpsi)

    j_inc = (hbar_val * k / m_val) * np.abs(A)**2
    j_ref = -(hbar_val * k / m_val) * np.abs(r * A)**2
    j_trans = (hbar_val * k / m_val) * np.abs(t_amp * A)**2

    # Aproximación WKB opaca (kappa * a >> 1)
    if E_val < V0_val:
        kappa_wkb = np.sqrt(2.0 * m_val * (V0_val - E_val)) / hbar_val
        T_wkb = (16.0 * E_val * (V0_val - E_val) / (V0_val**2)) * np.exp(-2.0 * kappa_wkb * a_val)
    else:
        T_wkb = 1.0

    return {
        'x_grid': x_grid,
        'psi': psi,
        'prob_density': np.abs(psi)**2,
        'j_net': j_net,
        'j_inc': j_inc,
        'j_ref': j_ref,
        'j_trans': j_trans,
        'T_exact': T_exact,
        'R_exact': R_exact,
        'T_wkb': T_wkb,
        'r_amp': r,
        't_amp': t_amp,
        'k': k
    }


class WavePacketTunnelingSimulator:
    """
    Parte C y D: Simulador dinámico por Split-Step Fourier Method para evolucionar
    un paquete gaussiano contra la barrera de potencial rectangular.
    Dominio ampliado Lx = 64.0 (Nx = 2048) para evitar wrapping periódico de FFT.
    """
    def __init__(self, E0=0.8, V0=1.0, a=1.5, sigma=1.2, x0=-10.0, m=1.0, hbar=1.0,
                 N_x=2048, L_x=64.0, dt=0.015):
        self.E0 = E0
        self.V0 = V0
        self.a = a
        self.sigma = sigma
        self.x0 = x0
        self.m = m
        self.hbar = hbar
        self.N_x = N_x
        self.L_x = L_x
        self.dt = dt

        self.reset()

    def reset(self):
        self.t = 0.0
        self.x_grid = np.linspace(-self.L_x/2.0, self.L_x/2.0, self.N_x, endpoint=False)
        self.dx = self.x_grid[1] - self.x_grid[0]

        # Potencial V(x)
        self.V_grid = np.where((self.x_grid >= 0.0) & (self.x_grid <= self.a), self.V0, 0.0)

        # Momento central k0
        self.k0 = np.sqrt(2.0 * self.m * self.E0) / self.hbar

        # Paquete Gaussiano Normalizado Psi(x,0)
        norm_factor = (2.0 * np.pi * self.sigma**2)**(-0.25)
        self.psi = norm_factor * np.exp(-((self.x_grid - self.x0)**2) / (4.0 * self.sigma**2)) * np.exp(1j * self.k0 * self.x_grid)

        # Grilla de momentos en Fourier (k_space)
        self.k_grid = 2.0 * np.pi * fftfreq(self.N_x, self.dx)

        # Operadores del Split-Step Fourier Method: exp(-i V dt / 2) y exp(-i T dt)
        self.U_V = np.exp(-1j * (self.V_grid / self.hbar) * (self.dt / 2.0))
        self.U_T = np.exp(-1j * (self.hbar * self.k_grid**2 / (2.0 * self.m)) * self.dt)

        # Amplitud espectral inicial a_in(k) por FFT
        self.a_in_k = fft(self.psi) * self.dx / np.sqrt(2.0 * np.pi)

    def step(self):
        """Avanza un paso temporal dt mediante Split-Step Fourier (Unitario)."""
        self.psi = self.U_V * self.psi
        psi_k = fft(self.psi)
        psi_k = self.U_T * psi_k
        self.psi = ifft(psi_k)
        self.psi = self.U_V * self.psi
        self.t += self.dt

    def compute_metrics(self):
        """Calcula probabilidades por región (P_L, P_B, P_R), corriente j(x,t) y análisis espectral sin cota artificial."""
        prob_density = np.abs(self.psi)**2
        dx = self.dx

        # Integración de probabilidad por regiones
        mask_L = self.x_grid < 0.0
        mask_B = (self.x_grid >= 0.0) & (self.x_grid <= self.a)
        mask_R = self.x_grid > self.a

        P_L = np.sum(prob_density[mask_L]) * dx
        P_B = np.sum(prob_density[mask_B]) * dx
        P_R = np.sum(prob_density[mask_R]) * dx
        norm_total = np.sum(prob_density) * dx

        # Corriente de probabilidad j(x,t)
        dpsi = np.gradient(self.psi, dx)
        j_grid = (self.hbar / self.m) * np.imag(np.conj(self.psi) * dpsi)

        # Energía Media <H>
        T_psi_k = (self.hbar**2 * self.k_grid**2 / (2.0 * self.m)) * fft(self.psi)
        T_psi = ifft(T_psi_k)
        E_mean = np.real(np.sum(np.conj(self.psi) * (T_psi + self.V_grid * self.psi)) * dx)

        # Filtrado espectral Parte D: P(<) subumbral vs P(>) superumbral
        k_positive_mask = self.k_grid >= 0.0
        k_pos = self.k_grid[k_positive_mask]
        a_pos = self.a_in_k[k_positive_mask]

        sort_idx = np.argsort(k_pos)
        k_pos = k_pos[sort_idx]
        a_pos = a_pos[sort_idx]

        E_pos = (self.hbar**2 * k_pos**2) / (2.0 * self.m)
        kappa_pos = np.where(E_pos < self.V0, np.sqrt(2.0 * self.m * np.maximum(self.V0 - E_pos, 0.0)) / self.hbar, 1e-12)
        q_pos = np.where(E_pos > self.V0, np.sqrt(2.0 * self.m * np.maximum(E_pos - self.V0, 0.0)) / self.hbar, 1e-12)

        # CORRECCIÓN ERROR 1: Asignación limpia de T_k en E_pos = V0 evitando cota artificial 1e-10
        T_k = np.zeros_like(E_pos)
        mask_sub_k = E_pos < self.V0
        mask_sup_k = E_pos > self.V0
        mask_umb_k = np.isclose(E_pos, self.V0, atol=1e-5)

        denom_sub = 4.0 * np.maximum(E_pos[mask_sub_k] * (self.V0 - E_pos[mask_sub_k]), 1e-12)
        T_k[mask_sub_k] = 1.0 / (1.0 + ((self.V0**2) / denom_sub) * np.sinh(kappa_pos[mask_sub_k] * self.a)**2)

        denom_sup = 4.0 * np.maximum(E_pos[mask_sup_k] * (E_pos[mask_sup_k] - self.V0), 1e-12)
        T_k[mask_sup_k] = 1.0 / (1.0 + ((self.V0**2) / denom_sup) * np.sin(q_pos[mask_sup_k] * self.a)**2)

        T_umbral_val = 1.0 / (1.0 + (self.m * self.V0 * self.a**2) / (2.0 * self.hbar**2))
        T_k[mask_umb_k] = T_umbral_val

        dk = k_pos[1] - k_pos[0] if len(k_pos) > 1 else 1.0
        prob_k = np.abs(a_pos)**2
        P_trans_teor = np.sum(prob_k * T_k) * dk

        k_c = np.sqrt(2.0 * self.m * self.V0) / self.hbar
        mask_sub = k_pos < k_c
        mask_sup = k_pos >= k_c

        P_sub = np.sum(prob_k[mask_sub] * T_k[mask_sub]) * dk
        P_sup = np.sum(prob_k[mask_sup] * T_k[mask_sup]) * dk

        return {
            'x_grid': self.x_grid,
            'V_grid': self.V_grid,
            'psi': self.psi,
            'prob_density': prob_density,
            'j_grid': j_grid,
            'P_L': P_L,
            'P_B': P_B,
            'P_R': P_R,
            'norm_total': norm_total,
            'E_mean': E_mean,
            'P_trans_teor': P_trans_teor,
            'P_sub': P_sub,
            'P_sup': P_sup,
            'k_grid': k_pos,
            'prob_k': prob_k,
            'T_k': T_k,
            'k_c': k_c
        }


class TunnelingLabApp:
    """
    Aplicación Interactiva Unificada en Paleta VIRIDIS para el Laboratorio 13.
    Integra Modo 1 (Estado Estacionario: Partes A y B) y Modo 2 (Paquete de Ondas: Partes C, D y E).
    """
    def __init__(self):
        run_sympy_analysis()

        # Parámetros iniciales
        self.E_init = 0.8
        self.V0_init = 1.0
        self.a_init = 1.5
        self.sigma_init = 1.2
        self.m_init = 1.0

        self.mode = ['Paquete']  # 'Estacionario' o 'Paquete'
        self.is_animating = False

        self.sim_packet = WavePacketTunnelingSimulator(
            E0=self.E_init, V0=self.V0_init, a=self.a_init, sigma=self.sigma_init, m=self.m_init
        )

        plt.style.use('dark_background')

        # Paleta VIRIDIS COMPLETO (Estética general unificada)
        self.bg_fig = '#07151a'       # Fondo negro-turquesa profundo viridis
        self.bg_card = '#0e242b'      # Violeta-teal oscuro viridis
        self.bg_control = '#14353f'   # Verde-azul viridis medio

        self.fig = plt.figure(figsize=(16, 9.8), facecolor=self.bg_fig)

        # Contenedores de Subplots (2x2)
        self.ax_1 = self.fig.add_subplot(2, 2, 1, facecolor=self.bg_card)
        self.ax_2 = self.fig.add_subplot(2, 2, 2, facecolor=self.bg_card)
        self.ax_3 = self.fig.add_subplot(2, 2, 3, facecolor=self.bg_card)
        self.ax_4 = self.fig.add_subplot(2, 2, 4, facecolor=self.bg_card)

        plt.subplots_adjust(left=0.07, right=0.95, top=0.92, bottom=0.28, wspace=0.25, hspace=0.34)

        # Sliders y Controles
        ax_E     = self.fig.add_axes([0.12, 0.18, 0.32, 0.022], facecolor=self.bg_control)
        ax_V0    = self.fig.add_axes([0.12, 0.13, 0.32, 0.022], facecolor=self.bg_control)
        ax_a     = self.fig.add_axes([0.12, 0.08, 0.32, 0.022], facecolor=self.bg_control)

        ax_sigma = self.fig.add_axes([0.58, 0.18, 0.32, 0.022], facecolor=self.bg_control)
        ax_m     = self.fig.add_axes([0.58, 0.13, 0.32, 0.022], facecolor=self.bg_control)
        ax_t     = self.fig.add_axes([0.58, 0.08, 0.32, 0.022], facecolor=self.bg_control)

        ax_radio     = self.fig.add_axes([0.52, 0.025, 0.18, 0.05], facecolor=self.bg_card)
        ax_btn_play  = self.fig.add_axes([0.72, 0.025, 0.10, 0.045], facecolor='#21918c')
        ax_btn_reset = self.fig.add_axes([0.83, 0.025, 0.10, 0.045], facecolor='#3b528b')

        # Botones de Etapas de Colisión (Parte E)
        ax_btn_e1 = self.fig.add_axes([0.12, 0.025, 0.08, 0.035], facecolor='#35b779')
        ax_btn_e2 = self.fig.add_axes([0.21, 0.025, 0.08, 0.035], facecolor='#21918c')
        ax_btn_e3 = self.fig.add_axes([0.30, 0.025, 0.08, 0.035], facecolor='#3b528b')
        ax_btn_e4 = self.fig.add_axes([0.39, 0.025, 0.08, 0.035], facecolor='#fde725')

        # Sliders en paleta Viridis (#21918c, #35b779, #fde725, #40e0d0, #3b528b, #f8765c)
        self.s_E     = Slider(ax_E,     'Energía E', 0.1, 2.5, valinit=self.E_init, color='#21918c')
        self.s_V0    = Slider(ax_V0,    'Barrera V₀', 0.2, 2.5, valinit=self.V0_init, color='#35b779')
        self.s_a     = Slider(ax_a,     'Ancho a', 0.2, 3.0, valinit=self.a_init, color='#fde725')

        self.s_sigma = Slider(ax_sigma, 'Ancho σ', 0.5, 3.0, valinit=self.sigma_init, color='#40e0d0')
        self.s_m     = Slider(ax_m,     'Masa m', 0.5, 3.0, valinit=self.m_init, color='#3b528b')
        self.s_t     = Slider(ax_t,     'Tiempo t', 0.0, 15.0, valinit=0.0, color='#f8765c')

        self.radio_mode = RadioButtons(ax_radio, ('Modo: Paquete', 'Modo: Estacionario'), active=0, activecolor='#21918c')
        for txt in self.radio_mode.labels:
            txt.set_color('white')
            txt.set_fontsize(8.5)

        self.btn_play  = Button(ax_btn_play,  '▶ Evolucionar', color='#21918c', hovercolor='#35b779')
        self.btn_reset = Button(ax_btn_reset, '↺ Reiniciar', color='#3b528b', hovercolor='#21918c')

        self.btn_e1 = Button(ax_btn_e1, 'Etapa 1', color='#35b779', hovercolor='#21918c')
        self.btn_e2 = Button(ax_btn_e2, 'Etapa 2', color='#21918c', hovercolor='#35b779')
        self.btn_e3 = Button(ax_btn_e3, 'Etapa 3', color='#3b528b', hovercolor='#21918c')
        self.btn_e4 = Button(ax_btn_e4, 'Etapa 4', color='#fde725', hovercolor='#35b779')

        self.btn_play.on_clicked(self.toggle_anim)
        self.btn_reset.on_clicked(self.reset_params)
        self.radio_mode.on_clicked(self.on_mode_changed)

        self.btn_e1.on_clicked(lambda e: self.jump_stage(0.0))
        self.btn_e2.on_clicked(lambda e: self.jump_stage(2.5))
        self.btn_e3.on_clicked(lambda e: self.jump_stage(5.0))
        self.btn_e4.on_clicked(lambda e: self.jump_stage(8.5))

        self.s_E.on_changed(self.on_slider_changed)
        self.s_V0.on_changed(self.on_slider_changed)
        self.s_a.on_changed(self.on_slider_changed)
        self.s_sigma.on_changed(self.on_slider_changed)
        self.s_m.on_changed(self.on_slider_changed)
        self.s_t.on_changed(self.on_time_slider_changed)

        # Historial temporal de probabilidades
        self.history_t = []
        self.history_PL = []
        self.history_PB = []
        self.history_PR = []

        self.anim = None
        self.reset_simulation_state()
        self.update_plots()

    def reset_simulation_state(self):
        self.is_animating = False
        self.btn_play.label.set_text('▶ Evolucionar')
        self.btn_play.ax.set_facecolor('#21918c')

        self.sim_packet.E0 = self.s_E.val
        self.sim_packet.V0 = self.s_V0.val
        self.sim_packet.a = self.s_a.val
        self.sim_packet.sigma = self.s_sigma.val
        self.sim_packet.m = self.s_m.val
        self.sim_packet.reset()

        self.history_t.clear()
        self.history_PL.clear()
        self.history_PB.clear()
        self.history_PR.clear()

        # Guardar estado inicial t=0 en historial
        m0 = self.sim_packet.compute_metrics()
        self.history_t.append(self.sim_packet.t)
        self.history_PL.append(m0['P_L'])
        self.history_PB.append(m0['P_B'])
        self.history_PR.append(m0['P_R'])

        self.s_t.eventson = False
        self.s_t.set_val(0.0)
        self.s_t.eventson = True

    def toggle_anim(self, event=None):
        if self.mode[0] == 'Estacionario':
            return
        self.is_animating = not self.is_animating
        self.btn_play.label.set_text('❚❚ Pausa' if self.is_animating else '▶ Evolucionar')
        self.btn_play.ax.set_facecolor('#fde725' if self.is_animating else '#21918c')
        if self.is_animating and self.anim is None:
            self.run_animation()

    def run_animation(self):
        def animate(frame):
            if self.is_animating and self.mode[0] == 'Paquete':
                self.sim_packet.step()
                m_packet = self.sim_packet.compute_metrics()
                self.history_t.append(self.sim_packet.t)
                self.history_PL.append(m_packet['P_L'])
                self.history_PB.append(m_packet['P_B'])
                self.history_PR.append(m_packet['P_R'])

                self.s_t.eventson = False
                self.s_t.set_val(min(self.sim_packet.t, 15.0))
                self.s_t.eventson = True

                self.update_plots()

        self.anim = FuncAnimation(self.fig, animate, interval=30, cache_frame_data=False)

    def jump_stage(self, t_target):
        """CORRECCIÓN ERROR 2: Calcula y acumula progresivamente la curva temporal hasta t_target."""
        if self.mode[0] == 'Estacionario':
            return
        self.reset_simulation_state()
        steps = int(t_target / self.sim_packet.dt)

        for i in range(steps):
            self.sim_packet.step()
            if i % 2 == 0 or i == steps - 1:
                m_packet = self.sim_packet.compute_metrics()
                self.history_t.append(self.sim_packet.t)
                self.history_PL.append(m_packet['P_L'])
                self.history_PB.append(m_packet['P_B'])
                self.history_PR.append(m_packet['P_R'])

        self.s_t.eventson = False
        self.s_t.set_val(min(self.sim_packet.t, 15.0))
        self.s_t.eventson = True

        self.update_plots()

    def on_time_slider_changed(self, val):
        if self.mode[0] == 'Estacionario':
            return
        t_target = self.s_t.val
        if t_target < self.sim_packet.t:
            self.reset_simulation_state()

        steps = int((t_target - self.sim_packet.t) / self.sim_packet.dt)
        if steps > 0:
            for i in range(steps):
                self.sim_packet.step()
                if i % 2 == 0 or i == steps - 1:
                    m_packet = self.sim_packet.compute_metrics()
                    self.history_t.append(self.sim_packet.t)
                    self.history_PL.append(m_packet['P_L'])
                    self.history_PB.append(m_packet['P_B'])
                    self.history_PR.append(m_packet['P_R'])
        self.update_plots()

    def reset_params(self, event=None):
        self.s_E.set_val(self.E_init)
        self.s_V0.set_val(self.V0_init)
        self.s_a.set_val(self.a_init)
        self.s_sigma.set_val(self.sigma_init)
        self.s_m.set_val(self.m_init)
        self.reset_simulation_state()
        self.update_plots()

    def on_mode_changed(self, label):
        if 'Estacionario' in label:
            self.mode[0] = 'Estacionario'
        else:
            self.mode[0] = 'Paquete'
        self.reset_simulation_state()
        self.update_plots()

    def on_slider_changed(self, val=None):
        if not self.is_animating:
            if self.mode[0] == 'Paquete':
                self.reset_simulation_state()
            self.update_plots()

    def update_plots(self):
        E_val = self.s_E.val
        V0_val = self.s_V0.val
        a_val = self.s_a.val
        m_val = self.s_m.val

        if self.mode[0] == 'Estacionario':
            # ================= MODO 1: ESTADO ESTACIONARIO (PARTES A Y B) =================
            x_grid = np.linspace(-12.0, 12.0 + a_val, 1500)
            res_stat = compute_stationary_state(E_val, V0_val, a_val, m_val=m_val, x_grid=x_grid)

            psi = res_stat['psi']
            prob = res_stat['prob_density']

            # Panel 1: Parte A - Función de Onda ψ(x) y Densidad |ψ|^2
            self.ax_1.clear()
            self.ax_1.set_facecolor(self.bg_card)
            self.ax_1.plot(x_grid, prob, color='#fde725', linewidth=2.5, label=r'Densidad $|\psi(x)|^2$')
            self.ax_1.fill_between(x_grid, 0, prob, color='#fde725', alpha=0.25)
            self.ax_1.plot(x_grid, np.real(psi), color='#40e0d0', linewidth=1.5, label=r'$\text{Re}[\psi(x)]$')
            self.ax_1.plot(x_grid, np.imag(psi), color='#35b779', linewidth=1.5, linestyle=':', label=r'$\text{Im}[\psi(x)]$')

            # Potencial V(x) y Nivel de Energía E en tonos Viridis
            V_draw = np.where((x_grid >= 0.0) & (x_grid <= a_val), V0_val, 0.0)
            self.ax_1.plot(x_grid, V_draw, color='#b5367a', linewidth=2.2, label=f'Barrera V₀ = {V0_val:.2f}')
            self.ax_1.axhline(E_val, color='#21918c', linestyle='--', linewidth=1.5, label=f'Energía E = {E_val:.2f}')

            self.ax_1.set_xlim([x_grid[0], x_grid[-1]])
            self.ax_1.set_ylim([-1.2, max(np.max(prob)*1.2, V0_val*1.2, 2.0)])
            self.ax_1.set_xlabel('Posición x (u.a.)', color='white', fontsize=8)
            self.ax_1.set_ylabel('Amplitud / Potencial', color='white', fontsize=8)
            self.ax_1.set_title(r'PARTE A: Estado Estacionario $\psi(x)$', color='#35b779', fontsize=10, fontweight='bold')
            self.ax_1.grid(True, color='#1f2937', linestyle='--', alpha=0.5)
            self.ax_1.legend(loc='upper right', facecolor=self.bg_card, edgecolor='#21918c', fontsize=7)

            # Panel 2: Parte B1 - Transmisión T(E) vs Umbral y Resonancias
            self.ax_2.clear()
            self.ax_2.set_facecolor(self.bg_card)
            E_grid = np.linspace(0.01, 2.5 * V0_val, 600)
            T_grid = [compute_stationary_state(e_g, V0_val, a_val, m_val=m_val)['T_exact'] for e_g in E_grid]

            self.ax_2.plot(E_grid, T_grid, color='#40e0d0', linewidth=2.2, label=r'Transmisión Exacta $T(E)$')
            self.ax_2.axvline(V0_val, color='#b5367a', linestyle=':', linewidth=1.5, label=f'Umbral $V_0 = {V0_val:.2f}$')
            self.ax_2.plot(E_val, res_stat['T_exact'], 'o', color='#fde725', markersize=7, label=f'Punto Actual $T={res_stat["T_exact"]:.4f}$')

            self.ax_2.set_xlim([0, E_grid[-1]])
            self.ax_2.set_ylim([-0.05, 1.1])
            self.ax_2.set_xlabel('Energía Incidente E', color='white', fontsize=8)
            self.ax_2.set_ylabel('Transmisión T', color='white', fontsize=8)
            self.ax_2.set_title(r'PARTE B1: Transmisión $T(E)$ y Resonancias', color='#40e0d0', fontsize=10, fontweight='bold')
            self.ax_2.grid(True, color='#1f2937', linestyle='--', alpha=0.5)
            self.ax_2.legend(loc='lower right', facecolor=self.bg_card, edgecolor='#40e0d0', fontsize=7)

            # Panel 3: Parte B2 - Transmisión Opaca ln T(a) vs Ancho a
            self.ax_3.clear()
            self.ax_3.set_facecolor(self.bg_card)
            a_grid = np.linspace(0.1, 4.0, 300)
            lnT_exact = [np.log(np.maximum(compute_stationary_state(E_val, V0_val, a_g, m_val=m_val)['T_exact'], 1e-15)) for a_g in a_grid]
            lnT_wkb = [np.log(np.maximum(compute_stationary_state(E_val, V0_val, a_g, m_val=m_val)['T_wkb'], 1e-15)) for a_g in a_grid]

            self.ax_3.plot(a_grid, lnT_exact, color='#35b779', linewidth=2.0, label=r'$\ln T_{\text{exacta}}(a)$')
            self.ax_3.plot(a_grid, lnT_wkb, color='#fde725', linestyle='--', linewidth=1.5, label=r'$\ln T_{\text{WKB}}(a)$')
            self.ax_3.axvline(a_val, color='#21918c', linestyle=':', label=f'Ancho Actual $a = {a_val:.2f}$')

            self.ax_3.set_xlabel('Ancho de Barrera a', color='white', fontsize=8)
            self.ax_3.set_ylabel(r'$\ln T$', color='white', fontsize=8)
            self.ax_3.set_title(r'PARTE B2: Régimen Opaco $\ln T(a)$', color='#fde725', fontsize=10, fontweight='bold')
            self.ax_3.grid(True, color='#1f2937', linestyle='--', alpha=0.5)
            self.ax_3.legend(loc='upper right', facecolor=self.bg_card, edgecolor='#35b779', fontsize=7)

            # Panel 4: Panel de Información y Métricas Estacionarias
            self.ax_4.clear()
            self.ax_4.axis('off')

            info_text = (
                "=== MODO ESTADO ESTACIONARIO (PARTE A & B) ===\n\n"
                f" • Energía Incidente (E) : {E_val:.4f}\n"
                f" • Altura Barrera (V₀)  : {V0_val:.4f}\n"
                f" • Ancho Barrera (a)    : {a_val:.4f}\n"
                f" • Masa Partícula (m)   : {m_val:.4f}\n"
                f" • Régimen Físico       : {'Efecto Túnel (Subumbral)' if E_val < V0_val else 'Superumbral / Resonancias'}\n\n"
                "--- Coeficientes de Flujo ---\n"
                f" • Transmisión Exacta T : {res_stat['T_exact']:.5f} ({res_stat['T_exact']*100:.2f}%)\n"
                f" • Reflexión Exacta R   : {res_stat['R_exact']:.5f} ({res_stat['R_exact']*100:.2f}%)\n"
                f" • Aproximación WKB T   : {res_stat['T_wkb']:.5f}\n"
                f" • Balance R + T       : {res_stat['R_exact'] + res_stat['T_exact']:.5f} (Conservación 1.00)\n\n"
                "--- Corrientes de Probabilidad ---\n"
                f" • Corriente Incidente  : {res_stat['j_inc']:.4f}\n"
                f" • Corriente Transmitida: {res_stat['j_trans']:.4f}\n"
                f" • Corriente Neta j(x)  : {np.mean(res_stat['j_net']):.4f} (Constante en todo x)\n"
            )

            self.ax_4.text(0.05, 0.95, info_text, color='#e2e8f0', fontsize=8.5, fontfamily='monospace',
                           verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.8', facecolor=self.bg_control, edgecolor='#35b779', alpha=0.95))

        else:
            # ================= MODO 2: PAQUETE DE ONDAS DINÁMICO (PARTES C, D Y E) =================
            m_packet = self.sim_packet.compute_metrics()

            x_grid = m_packet['x_grid']
            psi = m_packet['psi']
            prob = m_packet['prob_density']

            # Panel 1: Parte C - Paquete de Ondas Psi(x,t)
            self.ax_1.clear()
            self.ax_1.set_facecolor(self.bg_card)
            self.ax_1.plot(x_grid, prob, color='#fde725', linewidth=2.5, label=r'Densidad $|\Psi(x,t)|^2$')
            self.ax_1.fill_between(x_grid, 0, prob, color='#fde725', alpha=0.25)
            self.ax_1.plot(x_grid, np.real(psi), color='#40e0d0', linewidth=1.2, alpha=0.7, label=r'$\text{Re}[\Psi]$')

            self.ax_1.plot(x_grid, m_packet['V_grid'], color='#b5367a', linewidth=2.2, label=f'Barrera V₀={V0_val:.2f}')
            self.ax_1.axhline(E_val, color='#21918c', linestyle='--', linewidth=1.2, label=f'E₀ = {E_val:.2f}')

            # CORRECCIÓN ERROR 3: Visualización amplia del dominio Lx=64 sin alias de frontera
            self.ax_1.set_xlim([-24.0, 24.0])
            self.ax_1.set_ylim([-0.8, max(np.max(prob)*1.2, V0_val*1.2, 1.5)])
            self.ax_1.set_xlabel('Posición x (u.a.)', color='white', fontsize=8)
            self.ax_1.set_ylabel('Amplitud / Potencial', color='white', fontsize=8)
            self.ax_1.set_title(f'PARTE C: Paquete de Ondas (t = {self.sim_packet.t:.3f} s)', color='#35b779', fontsize=10, fontweight='bold')
            self.ax_1.grid(True, color='#1f2937', linestyle='--', alpha=0.5)
            self.ax_1.legend(loc='upper right', facecolor=self.bg_card, edgecolor='#21918c', fontsize=7)

            # Panel 2: Parte D1 - Espectro |a(k)|^2 y Transmisión Espectral T(k)
            self.ax_2.clear()
            self.ax_2.set_facecolor(self.bg_card)
            k_grid = m_packet['k_grid']
            prob_k = m_packet['prob_k']

            self.ax_2.plot(k_grid, prob_k, color='#35b779', linewidth=2.2, label=r'Espectro Incidente $|a(k)|^2$')
            self.ax_2.fill_between(k_grid, 0, prob_k, color='#35b779', alpha=0.2)
            self.ax_2.axvline(m_packet['k_c'], color='#b5367a', linestyle='--', linewidth=1.5, label=f'k_critico (E=V₀) = {m_packet["k_c"]:.2f}')

            self.ax_2.set_xlim([0, max(k_grid)])
            self.ax_2.set_xlabel('Número de Onda k', color='white', fontsize=8)
            self.ax_2.set_ylabel('Densidad Espectral', color='white', fontsize=8)
            self.ax_2.set_title(r'PARTE D1: Filtrado Espectral de Momentos $|a(k)|^2$', color='#21918c', fontsize=10, fontweight='bold')
            self.ax_2.grid(True, color='#1f2937', linestyle='--', alpha=0.5)
            self.ax_2.legend(loc='upper right', facecolor=self.bg_card, edgecolor='#21918c', fontsize=7)

            # Panel 3: Parte D2 - Evolución Temporal de Probabilidades por Región
            self.ax_3.clear()
            self.ax_3.set_facecolor(self.bg_card)
            self.ax_3.plot(self.history_t, self.history_PL, color='#21918c', linewidth=2.0, label=r'$P_L(t)$ (Región Izquierda)')
            self.ax_3.plot(self.history_t, self.history_PB, color='#35b779', linewidth=2.0, label=r'$P_B(t)$ (En Barrera)')
            self.ax_3.plot(self.history_t, self.history_PR, color='#fde725', linewidth=2.0, label=r'$P_R(t)$ (Transmitido)')

            self.ax_3.set_xlim([0, max(max(self.history_t) if len(self.history_t) > 0 else 1.0, 1.0)])
            self.ax_3.set_ylim([-0.05, 1.15])
            self.ax_3.set_xlabel('Tiempo t (u.a.)', color='white', fontsize=8)
            self.ax_3.set_ylabel('Probabilidad por Región', color='white', fontsize=8)
            self.ax_3.set_title(r'PARTE D2: Probabilidades por Región vs Tiempo', color='#fde725', fontsize=10, fontweight='bold')
            self.ax_3.grid(True, color='#1f2937', linestyle='--', alpha=0.5)
            self.ax_3.legend(loc='upper right', facecolor=self.bg_card, edgecolor='#35b779', fontsize=7)

            # Panel 4: Parte E - Panel de Información y Métricas de Colisión
            self.ax_4.clear()
            self.ax_4.axis('off')

            info_text = (
                "=== MODO PAQUETE DE ONDAS (PARTE C, D & E) ===\n\n"
                f" • Energía Central E₀    : {E_val:.4f}\n"
                f" • Energía Media <H>     : {m_packet['E_mean']:.4f}\n"
                f" • Norma Total <Ψ|Ψ>    : {m_packet['norm_total']:.5f} (Conservada)\n\n"
                "--- Probabilidades en Tiempo Actual ---\n"
                f" • P_L(t) [Izquierda]   : {m_packet['P_L']:.4f} ({m_packet['P_L']*100:.1f}%)\n"
                f" • P_B(t) [En Barrera]  : {m_packet['P_B']:.4f} ({m_packet['P_B']*100:.1f}%)\n"
                f" • P_R(t) [Transmitido] : {m_packet['P_R']:.4f} ({m_packet['P_R']*100:.1f}%)\n\n"
                "--- Filtrado Espectral de Transmisión ---\n"
                f" • P_trans Teórico Total: {m_packet['P_trans_teor']:.5f}\n"
                f" • P_sub (E < V₀ Túnel) : {m_packet['P_sub']:.5f} ({m_packet['P_sub']/(m_packet['P_trans_teor']+1e-12)*100:.1f}% del total)\n"
                f" • P_sup (E > V₀ Por arriba): {m_packet['P_sup']:.5f}\n"
            )

            self.ax_4.text(0.05, 0.95, info_text, color='#e2e8f0', fontsize=8.5, fontfamily='monospace',
                           verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.8', facecolor=self.bg_control, edgecolor='#35b779', alpha=0.95))

        self.fig.suptitle("LABORATORIO PYTHON 13: DOS MANERAS DE OBSERVAR EL EFECTO TÚNEL (PALETA VIRIDIS)",
                          color='white', fontsize=12, fontweight='bold', y=0.98)
        self.fig.canvas.draw_idle()


def export_static_sample(output_path="/workspace/scratch/laboratorio13_efecto_tunel_viridis_sample.png"):
    """
    Genera una imagen estática de prueba para verificar ejecución headless.
    """
    run_sympy_analysis()
    res_stat = compute_stationary_state(0.8, 1.0, 1.5)

    dir_name = os.path.dirname(os.path.abspath(output_path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8), facecolor='#07151a')
    ax = fig.add_subplot(1, 1, 1, facecolor='#0e242b')
    ax.plot(res_stat['x_grid'], res_stat['prob_density'], color='#fde725', linewidth=2.5, label=r'$|\psi(x)|^2$')
    ax.set_title("Muestra Estática del Laboratorio 13: Efecto Túnel (Paleta Viridis)", color='white')
    ax.legend()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] Sample saved to {output_path}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        export_static_sample()
    else:
        app = TunnelingLabApp()
        plt.show()
