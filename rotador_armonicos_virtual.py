"""
================================================================================
LABORATORIO PYTHON 16: ARMÓNICOS ESFÉRICOS Y EVOLUCIÓN TEMPORAL DEL ROTADOR CUÁNTICO
================================================================================
Materia: Mecánica Cuántica
Tema: Armónicos Esféricos Y_l^m(theta, phi) y Rotador Rígido Cuántico H = L^2 / (2I)
Modos: Parte A (Explorador 3D de Y_l^m) y Parte B (Evolución Temporal del Rotador Rígido)
Paleta: Viridis Completo (Estética Unificada Teal/Verde/Amarillo/Morado)
Librerías: NumPy, Matplotlib, SciPy, SymPy
================================================================================
"""

import sys
import os
import numpy as np
import sympy as sp
import scipy.special as ssp
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons

# Forzado de Backend interactivo para ejecuciones en terminal / VS Code
if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] != '--test'):
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass


def run_sympy_analysis():
    """
    Análisis simbólico con SymPy de los armónicos esféricos Y_l^m y del Hamiltoniano
    del rotador rígido cuántico H = L^2 / (2I).
    """
    print("=" * 80)
    print(" 1. ANÁLISIS SIMBÓLICO CON SYMPY - LABORATORIO 16: ROTADOR CUÁNTICO Y Y_l^m")
    print("=" * 80)

    l, m, hbar, I_mom, t = sp.symbols('l m hbar I t', real=True, positive=True)

    # Autovalores del operador momento angular total L^2 y L_z
    L2_eigen = hbar**2 * l * (l + 1)
    Lz_eigen = m * hbar

    # Hamiltoniano del rotador rígido H = L^2 / (2I)
    H_rot = L2_eigen / (2 * I_mom)

    print("\n• Autovalor del Operador Momento Angular L²:")
    sp.pprint(sp.Eq(sp.Symbol('L^2 |l,m>'), L2_eigen * sp.Symbol('|l,m>')))

    print("\n• Autovalor del Operador L_z:")
    sp.pprint(sp.Eq(sp.Symbol('L_z |l,m>'), Lz_eigen * sp.Symbol('|l,m>')))

    print("\n• Energías Cuantizadas del Rotador Rígido E_l = L^2 / (2I):")
    sp.pprint(sp.Eq(sp.Symbol('E_l'), H_rot))

    print("\n• Frecuencia de Fase Temporal omega_l = E_l / hbar:")
    omega_l = H_rot / hbar
    sp.pprint(sp.Eq(sp.Symbol('omega_l'), omega_l))

    # Frecuencia de batido cuántico entre niveles l1 y l2
    l1, l2 = sp.symbols('l1 l2', integer=True, positive=True)
    E1 = hbar**2 * l1 * (l1 + 1) / (2 * I_mom)
    E2 = hbar**2 * l2 * (l2 + 1) / (2 * I_mom)
    delta_omega = sp.simplify((E2 - E1) / hbar)

    print("\n• Frecuencia de Batido Cuántico delta_omega = (E2 - E1) / hbar:")
    sp.pprint(sp.Eq(sp.Symbol('Delta omega_{l1,l2}'), delta_omega))

    print("=" * 80 + "\n")


def compute_Ylm(l, m, theta_grid, phi_grid):
    """
    Calcula el armónico esférico complejo Y_l^m(theta, phi).
    theta: colatitud [0, pi]
    phi: azimut [0, 2pi]
    """
    l = int(l)
    m = int(m)
    if hasattr(ssp, 'sph_harm_y'):
        # SciPy 1.15+
        return ssp.sph_harm_y(l, m, theta_grid, phi_grid)
    else:
        # Fallback manual con polinomios asociados de Legendre
        abs_m = abs(m)
        P_lm = ssp.lpmv(abs_m, l, np.cos(theta_grid))
        norm = np.sqrt((2 * l + 1) / (4.0 * np.pi) * ssp.factorial(l - abs_m) / ssp.factorial(l + abs_m))
        Y = norm * P_lm * np.exp(1j * m * phi_grid)
        if m < 0:
            Y = ((-1)**abs_m) * np.conj(Y)
        return Y


class QuantumRotatorApp:
    """
    Aplicación Interactiva Unificada en Paleta VIRIDIS para el Laboratorio 16.
    Armónicos Esféricos y Evolución Temporal del Rotador Cuántico H = L^2 / (2I).
    """
    def __init__(self):
        run_sympy_analysis()

        # Parámetros iniciales
        self.l1_init = 1
        self.m1_init = 0
        self.l2_init = 2
        self.m2_init = 1

        self.c1_init = 1.0 / np.sqrt(2.0)
        self.c2_init = 1.0 / np.sqrt(2.0)

        self.hbar = 1.0
        self.I_mom = 1.0
        self.t_val = 0.0

        self.mode = ['Parte A']  # 'Parte A' (Armónico 3D) o 'Parte B' (Evolución Temporal Rotador)
        self.superpos_type = ['Puro |l1,m1>']  # 'Puro', 'Superposición |l1,m1> + |l2,m2>', 'Estado Coherente'

        self.is_animating = False

        plt.style.use('dark_background')

        # Paleta Viridis Completa (Estética unificada)
        self.bg_fig = '#07151a'       # Fondo profundo viridis
        self.bg_card = '#0e242b'      # Violeta-teal oscuro
        self.bg_control = '#14353f'   # Verde-azul medio

        self.fig = plt.figure(figsize=(16, 9.8), facecolor=self.bg_fig)

        # Grilla de ángulos esféricos para renderizado 3D y mapas 2D
        self.n_theta = 90
        self.n_phi = 180
        self.theta = np.linspace(0, np.pi, self.n_theta)
        self.phi = np.linspace(0, 2 * np.pi, self.n_phi)
        self.THETA, self.PHI = np.meshgrid(self.theta, self.phi)

        # Subplots
        self.ax_3d   = self.fig.add_subplot(2, 2, 1, projection='3d', facecolor=self.bg_card)
        self.ax_2d   = self.fig.add_subplot(2, 2, 2, facecolor=self.bg_card)
        self.ax_time = self.fig.add_subplot(2, 2, 3, facecolor=self.bg_card)
        self.ax_info = self.fig.add_subplot(2, 2, 4, facecolor=self.bg_card)

        plt.subplots_adjust(left=0.06, right=0.95, top=0.91, bottom=0.28, wspace=0.25, hspace=0.34)

        # Controles y Sliders
        ax_l1    = self.fig.add_axes([0.10, 0.18, 0.30, 0.02], facecolor=self.bg_control)
        ax_m1    = self.fig.add_axes([0.10, 0.13, 0.30, 0.02], facecolor=self.bg_control)
        ax_l2    = self.fig.add_axes([0.10, 0.08, 0.30, 0.02], facecolor=self.bg_control)
        ax_m2    = self.fig.add_axes([0.10, 0.03, 0.30, 0.02], facecolor=self.bg_control)

        ax_t     = self.fig.add_axes([0.55, 0.18, 0.30, 0.02], facecolor=self.bg_control)
        ax_c1    = self.fig.add_axes([0.55, 0.13, 0.30, 0.02], facecolor=self.bg_control)
        ax_I     = self.fig.add_axes([0.55, 0.08, 0.30, 0.02], facecolor=self.bg_control)

        ax_radio_mode   = self.fig.add_axes([0.42, 0.11, 0.11, 0.09], facecolor=self.bg_card)
        ax_radio_state  = self.fig.add_axes([0.42, 0.02, 0.11, 0.08], facecolor=self.bg_card)

        ax_btn_play  = self.fig.add_axes([0.87, 0.13, 0.09, 0.045], facecolor='#21918c')
        ax_btn_reset = self.fig.add_axes([0.87, 0.06, 0.09, 0.045], facecolor='#3b528b')

        # Sliders en paleta Viridis (#21918c, #35b779, #fde725, #40e0d0, #3b528b)
        self.s_l1 = Slider(ax_l1, 'l₁', 0, 4, valinit=self.l1_init, valstep=1, color='#35b779')
        self.s_m1 = Slider(ax_m1, 'm₁', -self.l1_init, self.l1_init, valinit=self.m1_init, valstep=1, color='#21918c')

        self.s_l2 = Slider(ax_l2, 'l₂', 0, 4, valinit=self.l2_init, valstep=1, color='#40e0d0')
        self.s_m2 = Slider(ax_m2, 'm₂', -self.l2_init, self.l2_init, valinit=self.m2_init, valstep=1, color='#3b528b')

        self.s_t  = Slider(ax_t,  'Tiempo t', 0.0, 10.0, valinit=self.t_val, color='#fde725')
        self.s_c1 = Slider(ax_c1, 'Peso c₁', 0.0, 1.0, valinit=0.7071, color='#35b779')
        self.s_I  = Slider(ax_I,  'Inercia I', 0.5, 3.0, valinit=self.I_mom, color='#21918c')

        self.radio_mode = RadioButtons(ax_radio_mode, ('Parte A: Y_l^m', 'Parte B: Rotador'), active=0, activecolor='#21918c')
        for txt in self.radio_mode.labels:
            txt.set_color('white')
            txt.set_fontsize(7.5)

        self.radio_state = RadioButtons(ax_radio_state, ('Puro |l1,m1>', 'Superposición', 'Coherente'), active=0, activecolor='#35b779')
        for txt in self.radio_state.labels:
            txt.set_color('white')
            txt.set_fontsize(7.5)

        self.btn_play  = Button(ax_btn_play,  '▶ Animación', color='#21918c', hovercolor='#35b779')
        self.btn_reset = Button(ax_btn_reset, '↺ Reiniciar',  color='#3b528b', hovercolor='#21918c')

        self.btn_play.on_clicked(self.toggle_anim)
        self.btn_reset.on_clicked(self.reset_params)
        self.radio_mode.on_clicked(self.on_mode_changed)
        self.radio_state.on_clicked(self.on_state_changed)

        self.s_l1.on_changed(self.on_l1_changed)
        self.s_m1.on_changed(self.update_plots)
        self.s_l2.on_changed(self.on_l2_changed)
        self.s_m2.on_changed(self.update_plots)

        self.s_t.on_changed(self.update_plots)
        self.s_c1.on_changed(self.update_plots)
        self.s_I.on_changed(self.update_plots)

        # Historial temporal
        self.history_t = []
        self.history_Lz = []
        self.history_E = []
        self.history_norm = []

        self.update_plots()

    def on_l1_changed(self, val):
        l1_val = int(self.s_l1.val)
        self.s_m1.valmin = -l1_val
        self.s_m1.valmax = l1_val
        self.s_m1.ax.set_xlim(-l1_val, l1_val)
        if abs(self.s_m1.val) > l1_val:
            self.s_m1.set_val(0)
        self.update_plots()

    def on_l2_changed(self, val):
        l2_val = int(self.s_l2.val)
        self.s_m2.valmin = -l2_val
        self.s_m2.valmax = l2_val
        self.s_m2.ax.set_xlim(-l2_val, l2_val)
        if abs(self.s_m2.val) > l2_val:
            self.s_m2.set_val(0)
        self.update_plots()

    def toggle_anim(self, event=None):
        self.is_animating = not self.is_animating
        self.btn_play.label.set_text('❚❚ Pausa' if self.is_animating else '▶ Animación')
        self.btn_play.ax.set_facecolor('#fde725' if self.is_animating else '#21918c')
        if self.is_animating:
            self.run_animation()

    def run_animation(self):
        def animate(frame):
            if self.is_animating:
                t_curr = self.s_t.val + 0.05
                if t_curr > self.s_t.valmax:
                    t_curr = 0.0
                    self.history_t.clear()
                    self.history_Lz.clear()
                    self.history_E.clear()
                    self.history_norm.clear()
                self.s_t.set_val(t_curr)

        self.anim = FuncAnimation(self.fig, animate, interval=40, cache_frame_data=False)

    def reset_params(self, event=None):
        self.is_animating = False
        self.btn_play.label.set_text('▶ Animación')
        self.btn_play.ax.set_facecolor('#21918c')

        self.s_l1.set_val(self.l1_init)
        self.s_m1.set_val(self.m1_init)
        self.s_l2.set_val(self.l2_init)
        self.s_m2.set_val(self.m2_init)
        self.s_t.set_val(0.0)
        self.s_c1.set_val(0.7071)
        self.s_I.set_val(self.I_mom)

        self.history_t.clear()
        self.history_Lz.clear()
        self.history_E.clear()
        self.history_norm.clear()

        self.update_plots()

    def on_mode_changed(self, label):
        if 'Parte A' in label:
            self.mode[0] = 'Parte A'
        else:
            self.mode[0] = 'Parte B'
        self.update_plots()

    def on_state_changed(self, label):
        if 'Puro' in label:
            self.superpos_type[0] = 'Puro |l1,m1>'
        elif 'Superposición' in label:
            self.superpos_type[0] = 'Superposición'
        else:
            self.superpos_type[0] = 'Coherente'
        self.update_plots()

    def compute_wavefunction_and_metrics(self):
        l1 = int(self.s_l1.val)
        m1 = int(self.s_m1.val)
        l2 = int(self.s_l2.val)
        m2 = int(self.s_m2.val)

        t_val = self.s_t.val
        c1_val = self.s_c1.val
        c2_val = np.sqrt(max(0.0, 1.0 - c1_val**2))
        I_val = self.s_I.val

        # Energías E_l = hbar^2 l(l+1) / (2I)
        E1 = (self.hbar**2 * l1 * (l1 + 1)) / (2.0 * I_val)
        E2 = (self.hbar**2 * l2 * (l2 + 1)) / (2.0 * I_val)

        Y1 = compute_Ylm(l1, m1, self.THETA, self.PHI)
        Y2 = compute_Ylm(l2, m2, self.THETA, self.PHI)

        if self.superpos_type[0] == 'Puro |l1,m1>':
            psi = Y1 * np.exp(-1j * E1 * t_val / self.hbar)
            E_mean = E1
            Lz_mean = m1 * self.hbar
            L2_mean = self.hbar**2 * l1 * (l1 + 1)
            delta_omega = 0.0
            T_batido = np.inf
        elif self.superpos_type[0] == 'Superposición':
            psi = c1_val * Y1 * np.exp(-1j * E1 * t_val / self.hbar) + \
                  c2_val * Y2 * np.exp(-1j * E2 * t_val / self.hbar)
            E_mean = c1_val**2 * E1 + c2_val**2 * E2
            Lz_mean = (c1_val**2 * m1 + c2_val**2 * m2) * self.hbar
            L2_mean = self.hbar**2 * (c1_val**2 * l1 * (l1 + 1) + c2_val**2 * l2 * (l2 + 1))
            delta_omega = abs(E2 - E1) / self.hbar
            T_batido = (2.0 * np.pi / delta_omega) if delta_omega > 1e-6 else np.inf
        else:
            # Paquete Coherente (Superposición de l=0,1,2)
            coeffs = [0.5, 0.7071, 0.5]
            norm_c = np.sqrt(sum(c**2 for c in coeffs))
            coeffs = [c / norm_c for c in coeffs]
            ls = [0, 1, 2]
            ms = [0, 0, 0]

            psi = np.zeros_like(Y1, dtype=complex)
            E_mean = 0.0
            Lz_mean = 0.0
            L2_mean = 0.0

            for c_i, l_i, m_i in zip(coeffs, ls, ms):
                E_i = (self.hbar**2 * l_i * (l_i + 1)) / (2.0 * I_val)
                Y_i = compute_Ylm(l_i, m_i, self.THETA, self.PHI)
                psi += c_i * Y_i * np.exp(-1j * E_i * t_val / self.hbar)
                E_mean += c_i**2 * E_i
                L2_mean += self.hbar**2 * c_i**2 * l_i * (l_i + 1)

            delta_omega = (E1 - 0.0) / self.hbar
            T_batido = (2.0 * np.pi / delta_omega) if delta_omega > 1e-6 else np.inf

        prob = np.abs(psi)**2

        # Integración numérica en la esfera 2D: dOmega = sin(theta) dtheta dphi
        dtheta = self.theta[1] - self.theta[0]
        dphi = self.phi[1] - self.phi[0]
        SIN_THETA = np.sin(self.THETA)

        norm_total = np.sum(prob * SIN_THETA) * dtheta * dphi

        # Nodos latitudinales (theta) y longitudinales (phi)
        nodes_lat = l1 - abs(m1)
        nodes_long = 2 * abs(m1)

        return {
            'psi': psi,
            'prob': prob,
            'l1': l1,
            'm1': m1,
            'l2': l2,
            'm2': m2,
            'c1': c1_val,
            'c2': c2_val,
            'E1': E1,
            'E2': E2,
            'E_mean': E_mean,
            'Lz_mean': Lz_mean,
            'L2_mean': L2_mean,
            'norm_total': norm_total,
            'nodes_lat': nodes_lat,
            'nodes_long': nodes_long,
            'delta_omega': delta_omega,
            'T_batido': T_batido,
            't_val': t_val,
            'I_val': I_val
        }

    def update_plots(self, val=None):
        res = self.compute_wavefunction_and_metrics()

        self.history_t.append(res['t_val'])
        self.history_Lz.append(res['Lz_mean'])
        self.history_E.append(res['E_mean'])
        self.history_norm.append(res['norm_total'])

        # ---------------- 1. SUBPLOT 1: REPRESENTACIÓN SUPERFICIE 3D ----------------
        self.ax_3d.clear()
        self.ax_3d.set_facecolor(self.bg_card)

        # Radio r(theta, phi) proporcional a la densidad |psi|^2 o amplitud |psi|
        r_surf = np.abs(res['psi'])
        r_surf_norm = r_surf / np.max(r_surf) if np.max(r_surf) > 1e-12 else r_surf

        X = r_surf_norm * np.sin(self.THETA) * np.cos(self.PHI)
        Y = r_surf_norm * np.sin(self.THETA) * np.sin(self.PHI)
        Z = r_surf_norm * np.cos(self.THETA)

        # Mapeo de Fase complejas Arg(psi) en la superficie 3D (Paleta Viridis)
        phase_surf = np.angle(res['psi'])
        colors = plt.cm.viridis((phase_surf + np.pi) / (2 * np.pi))

        self.ax_3d.plot_surface(X, Y, Z, facecolors=colors, rstride=2, cstride=2, linewidth=0, antialiased=True, alpha=0.9)
        self.ax_3d.set_title(f'Superficie 3D: |\\Psi(\\theta,\\phi)| (Fase en Viridis)', color='#fde725', fontsize=9.5, fontweight='bold')

        self.ax_3d.set_axis_off()

        # ---------------- 2. SUBPLOT 2: MAPA 2D ANGULAR |psi(theta, phi)|^2 ----------------
        self.ax_2d.clear()
        self.ax_2d.set_facecolor(self.bg_card)

        mesh = self.ax_2d.pcolormesh(np.degrees(self.PHI), np.degrees(self.THETA), res['prob'], cmap='viridis', shading='auto')
        self.ax_2d.set_xlabel('Azimut \\phi (°)', color='white', fontsize=8)
        self.ax_2d.set_ylabel('Colatitud \\theta (°)', color='white', fontsize=8)
        self.ax_2d.set_title(r'Densidad Angular $|\Psi(\theta, \phi, t)|^2$', color='#35b779', fontsize=9.5, fontweight='bold')
        self.ax_2d.set_yticks([0, 45, 90, 135, 180])
        self.ax_2d.set_xticks([0, 90, 180, 270, 360])
        self.ax_2d.grid(True, color='#1f2937', linestyle='--', alpha=0.5)

        # ---------------- 3. SUBPLOT 3: EVOLUCIÓN TEMPORAL DE Lz Y ENERGÍA ----------------
        self.ax_time.clear()
        self.ax_time.set_facecolor(self.bg_card)

        if len(self.history_t) > 0:
            self.ax_time.plot(self.history_t, self.history_Lz, color='#40e0d0', linewidth=2.0, label=r'$\langle L_z \rangle(t)$')
            self.ax_time.plot(self.history_t, self.history_E, color='#fde725', linestyle='--', linewidth=1.5, label=r'$\langle H \rangle = \text{Const}$')
            self.ax_time.plot(self.history_t, self.history_norm, color='#35b779', linestyle=':', linewidth=1.2, label=r'$\int |\Psi|^2 d\Omega = 1.00$')

        self.ax_time.set_xlim([0, max(max(self.history_t) if len(self.history_t) > 0 else 10.0, 10.0)])
        self.ax_time.set_xlabel('Tiempo t (u.a.)', color='white', fontsize=8)
        self.ax_time.set_ylabel('Valores Esperados / Norma', color='white', fontsize=8)
        self.ax_time.set_title('Monitoreo Dinámico: Conservación y Batidos', color='#40e0d0', fontsize=9.5, fontweight='bold')
        self.ax_time.grid(True, color='#1f2937', linestyle='--', alpha=0.5)
        self.ax_time.legend(loc='upper right', facecolor=self.bg_card, edgecolor='#40e0d0', fontsize=7)

        # ---------------- 4. SUBPLOT 4: PANEL DE INFORMACIÓN Y MÉTRICAS CUÁNTICAS ----------------
        self.ax_info.clear()
        self.ax_info.axis('off')

        if self.mode[0] == 'Parte A':
            title_mode = "=== PARTE A: EXPLORADOR DE ARMÓNICOS ESFÉRICOS Y_l^m ==="
            info_text = (
                f"{title_mode}\n\n"
                f" • Estado Seleccionado   : Y_{res['l1']}^{{{res['m1']}}}(\\theta, \\phi)\n"
                f" • Grado l₁              : {res['l1']}\n"
                f" • Orden m₁              : {res['m1']}\n"
                f" • Autovalor L²          : {res['l1']*(res['l1']+1):.2f} ℏ²\n"
                f" • Autovalor L_z         : {res['m1']:.2f} ℏ\n\n"
                "--- Nodos Angulares ---\n"
                f" • Nodos Latitudinales (\\theta) : {res['nodes_lat']} (Conos / Paralelos)\n"
                f" • Nodos Longitudinales (\\phi): {res['nodes_long']} (Planos Meridianos)\n"
                f" • Total Nodos Angulares : {res['l1']}\n\n"
                "--- Normalización en la Esfera ---\n"
                f" • ∫ |Y_l^m|² d\\Omega    : {res['norm_total']:.5f} (Normalizada 1.00)\n"
            )
        else:
            title_mode = "=== PARTE B: ROTADOR RÍGIDO CUÁNTICO H = L² / (2I) ==="
            info_text = (
                f"{title_mode}\n\n"
                f" • Modo de Estado        : {self.superpos_type[0]}\n"
                f" • Estado 1 |l1,m1>      : |{res['l1']},{res['m1']}> (E₁ = {res['E1']:.4f})\n"
                f" • Estado 2 |l2,m2>      : |{res['l2']},{res['m2']}> (E₂ = {res['E2']:.4f})\n"
                f" • Momento de Inercia I  : {res['I_val']:.2f}\n\n"
                "--- Métricas Cuánticas y Conservación ---\n"
                f" • Energía Media <H>     : {res['E_mean']:.5f} (Constante)\n"
                f" • Moment. Angular <L_z>  : {res['Lz_mean']:.5f} ℏ\n"
                f" • Norma Total ∫|Ψ|² d\\Omega: {res['norm_total']:.5f} (Conservada)\n\n"
                "--- Frecuencia de Batido Cuántico ---\n"
                f" • Δ\\omega = (E₂ - E₁) / ℏ: {res['delta_omega']:.4f} rad/s\n"
                f" • Período Batido T      : {res['T_batido']:.4f} s\n"
            )

        self.ax_info.text(0.05, 0.95, info_text, color='#e2e8f0', fontsize=8.5, fontfamily='monospace',
                       verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.8', facecolor=self.bg_control, edgecolor='#35b779', alpha=0.95))

        self.fig.suptitle("LABORATORIO PYTHON 16: ARMÓNICOS ESFÉRICOS Y ROTADOR CUÁNTICO (PALETA VIRIDIS)",
                          color='white', fontsize=12, fontweight='bold', y=0.98)
        self.fig.canvas.draw_idle()


def export_static_sample(output_path="/workspace/scratch/rotador_armonicos_sample.png"):
    """
    Genera una imagen estática de prueba para verificar ejecución en entorno headless.
    """
    run_sympy_analysis()
    
    theta = np.linspace(0, np.pi, 60)
    phi = np.linspace(0, 2*np.pi, 120)
    THETA, PHI = np.meshgrid(theta, phi)
    Y21 = compute_Ylm(2, 1, THETA, PHI)
    prob = np.abs(Y21)**2

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#07151a')
    for ax in axes:
        ax.set_facecolor('#0e242b')

    mesh = axes[0].pcolormesh(np.degrees(PHI), np.degrees(THETA), prob, cmap='viridis', shading='auto')
    axes[0].set_title("Densidad Angular |Y_2^1|^2", color='white')
    axes[0].set_xlabel("Phi (°)")
    axes[0].set_ylabel("Theta (°)")

    axes[1].plot(np.degrees(theta), np.mean(prob, axis=0), color='#fde725', linewidth=2)
    axes[1].set_title("Perfil Latitudinal Integrado", color='white')
    axes[1].set_xlabel("Theta (°)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] Sample saved to {output_path}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        export_static_sample()
    else:
        app = QuantumRotatorApp()
        plt.show()
