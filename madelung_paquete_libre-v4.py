"""
================================================================================
LABORATORIO PYTHON 11: FORMULACIÓN HIDRODINÁMICA DE MADELUNG (PROPUESTA I)
================================================================================
Materia: Mecánica Cuántica
Tema: Dispersión y Flujo de un Paquete Libre en 1D (Sin Potencial V=0)
Paleta: Inferno Unificada
Librerías: NumPy, Matplotlib, SciPy, SymPy
================================================================================
"""

import sys
import os
import numpy as np
import sympy as sp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

# Forzado de Backend interactivo para ejecuciones en terminal / VS Code
if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] != '--test'):
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass


def run_madelung_sympy():
    """
    Análisis algebraico con SymPy de las ecuaciones de Madelung para el paquete libre.
    """
    print("=" * 80)
    print(" 1. ANÁLISIS SIMBÓLICO CON SYMPY - LABORATORIO 11: HIDRODINÁMICA DE MADELUNG")
    print("=" * 80)

    x, t, m, hbar, sigma0, p0, x0 = sp.symbols('x t m hbar sigma0 p0 x0', real=True, positive=True)

    # Densidad R(x,t) y Fase S(x,t)
    st = sigma0 * sp.sqrt(1 + (hbar * t / (2 * m * sigma0**2))**2)
    xc = x0 + (p0 / m) * t

    R = (1 / (2 * sp.pi * st**2)**(1/4)) * sp.exp(-(x - xc)**2 / (4 * st**2))
    
    # Velocidad v = grad(S)/m
    v = (p0 / m) + (hbar**2 * t / (4 * m**2 * sigma0**2 * st**2)) * (x - xc)

    # Potencial Cuántico Q = - (hbar^2 / 2m) * (d2R / R)
    dR = sp.diff(R, x)
    d2R = sp.diff(dR, x)
    Q = sp.simplify(- (hbar**2 / (2 * m)) * (d2R / R))

    print("\n• Expresión del Potencial Cuántico Q(x,t):")
    sp.pprint(Q)

    print("\n• Expresión de la Velocidad del Fluido v(x,t):")
    sp.pprint(v)
    print("=" * 80 + "\n")


def compute_madelung_state(x, t, x0=-10.0, p0=2.0, sigma0=2.0, hbar=1.0, m=1.0):
    """
    Calcula las magnitudes del paquete libre de onda y del fluido de Madelung.
    p0: Momento inicial del paquete libre.
    """
    k0 = p0 / hbar
    st = sigma0 * np.sqrt(1.0 + (hbar * t / (2.0 * m * sigma0**2))**2)
    xc = x0 + (p0 / m) * t

    norm = (1.0 / (2.0 * np.pi * sigma0**2))**0.25
    denom = np.sqrt(1.0 + 1j * hbar * t / (2.0 * m * sigma0**2))

    psi = (norm / denom) * np.exp(1j * k0 * x - 0.5j * (p0**2 / m) * t / hbar) * \
          np.exp(-(x - xc)**2 / (4.0 * sigma0**2 * (1.0 + 1j * hbar * t / (2.0 * m * sigma0**2))))

    rho = np.abs(psi)**2
    R = np.sqrt(rho)

    phase_mod = np.angle(psi)
    phase_unwrap = np.unwrap(phase_mod)

    # Velocidad del fluido v(x,t)
    v = (p0 / m) + (hbar**2 * t / (4.0 * m**2 * sigma0**2 * st**2)) * (x - xc)

    # Corriente de probabilidad j(x,t)
    j = rho * v

    # Potencial Cuántico Q(x,t)
    Q = (hbar**2 / (4.0 * m * st**2)) - (hbar**2 * (x - xc)**2 / (8.0 * m * st**4))

    # Fuerza Cuántica F_Q = - dQ/dx
    F_Q = (hbar**2 * (x - xc)) / (4.0 * m * st**4)

    # Energía Cinética Hidrodinámica T = 1/2 m v^2
    T_hydro = 0.5 * m * v**2

    # Energía Hidrodinámica Total E = T + Q (con V=0)
    E_hydro = T_hydro + Q

    return {
        'psi': psi,
        'rho': rho,
        'R': R,
        'phase_mod': phase_mod,
        'phase_unwrap': phase_unwrap,
        'v': v,
        'j': j,
        'Q': Q,
        'F_Q': F_Q,
        'T_hydro': T_hydro,
        'E_hydro': E_hydro,
        'xc': xc,
        'st': st
    }


class MadelungFreePacketApp:
    def __init__(self):
        run_madelung_sympy()

        # Parámetros iniciales
        self.x_min = -35.0
        self.x_max = 35.0
        self.Nx = 2049
        self.t_val = 0.0
        self.t_max = 6.0
        self.x0 = -10.0
        self.p0 = 2.0
        self.sigma0 = 2.0

        self.is_animating = False

        plt.style.use('dark_background')
        # Paleta Inferno Unificada - Colores de fondo y contenedores
        self.bg_fig = '#0a0612'       # Púrpura negro profundo
        self.bg_card = '#160c28'      # Violeta oscuro inferno
        self.bg_control = '#231138'   # Morado medio inferno

        self.fig = plt.figure(figsize=(16, 9.8), facecolor=self.bg_fig)

        # 8 Subplots unificados (2 filas x 4 columnas)
        self.ax_rho   = self.fig.add_subplot(2, 4, 1, facecolor=self.bg_card)
        self.ax_j     = self.fig.add_subplot(2, 4, 2, facecolor=self.bg_card)
        self.ax_phase = self.fig.add_subplot(2, 4, 3, facecolor=self.bg_card)
        self.ax_map   = self.fig.add_subplot(2, 4, 4, facecolor=self.bg_card)

        self.ax_v     = self.fig.add_subplot(2, 4, 5, facecolor=self.bg_card)
        self.ax_Q     = self.fig.add_subplot(2, 4, 6, facecolor=self.bg_card)
        self.ax_FQ    = self.fig.add_subplot(2, 4, 7, facecolor=self.bg_card)
        self.ax_E     = self.fig.add_subplot(2, 4, 8, facecolor=self.bg_card)

        self.spatial_axes = [self.ax_rho, self.ax_j, self.ax_phase, self.ax_v, self.ax_Q, self.ax_FQ, self.ax_E]

        plt.subplots_adjust(left=0.06, right=0.96, top=0.91, bottom=0.30, wspace=0.32, hspace=0.38)

        # Controles y Sliders
        ax_t     = self.fig.add_axes([0.10, 0.20, 0.32, 0.02], facecolor=self.bg_control)
        ax_sigma = self.fig.add_axes([0.10, 0.15, 0.32, 0.02], facecolor=self.bg_control)
        ax_p0    = self.fig.add_axes([0.10, 0.10, 0.32, 0.02], facecolor=self.bg_control)
        ax_x0    = self.fig.add_axes([0.10, 0.05, 0.32, 0.02], facecolor=self.bg_control)

        ax_xmin  = self.fig.add_axes([0.55, 0.20, 0.32, 0.02], facecolor=self.bg_control)
        ax_xmax  = self.fig.add_axes([0.55, 0.15, 0.32, 0.02], facecolor=self.bg_control)
        ax_tmax  = self.fig.add_axes([0.55, 0.10, 0.32, 0.02], facecolor=self.bg_control)

        # Botones con colores de la paleta inferno
        ax_btn_play  = self.fig.add_axes([0.55, 0.04, 0.09, 0.04], facecolor='#b5367a')
        ax_btn_reset = self.fig.add_axes([0.65, 0.04, 0.09, 0.04], facecolor='#781c6d')
        ax_btn_p1    = self.fig.add_axes([0.75, 0.04, 0.06, 0.04], facecolor='#ed6925')
        ax_btn_p2    = self.fig.add_axes([0.82, 0.04, 0.06, 0.04], facecolor='#fca50a')

        # Sliders utilizando tonos de la escala Inferno (#fca50a, #f8765c, #fde725, #ed6925, #b5367a, #fc2547)
        self.s_t     = Slider(ax_t,     'Tiempo t',       0.0, self.t_max, valinit=self.t_val, color='#f8765c')
        self.s_sigma = Slider(ax_sigma, 'Ancho σ₀',      0.5, 5.0, valinit=self.sigma0, color='#fca50a')
        self.s_p0    = Slider(ax_p0,    'Momento p₀',     0.0, 5.0, valinit=self.p0, color='#fde725')
        self.s_x0    = Slider(ax_x0,    'Posición x₀',  -25.0, 25.0, valinit=self.x0, color='#ed6925')

        self.s_xmin  = Slider(ax_xmin,  'Rango x_min',  -100.0, -10.0, valinit=self.x_min, color='#b5367a')
        self.s_xmax  = Slider(ax_xmax,  'Rango x_max',    10.0, 100.0, valinit=self.x_max, color='#f8765c')
        self.s_tmax  = Slider(ax_tmax,  'Tiempo t_max',   1.0,  20.0, valinit=self.t_max, color='#781c6d')

        self.btn_play  = Button(ax_btn_play,  '▶ Animación', color='#b5367a', hovercolor='#ed6925')
        self.btn_reset = Button(ax_btn_reset, '↺ Reiniciar', color='#781c6d', hovercolor='#b5367a')
        self.btn_p1    = Button(ax_btn_p1,    'Preset 1',    color='#ed6925', hovercolor='#fca50a')
        self.btn_p2    = Button(ax_btn_p2,    'Preset 2',    color='#fca50a', hovercolor='#fde725')

        self.btn_play.on_clicked(self.toggle_anim)
        self.btn_reset.on_clicked(self.reset_params)
        self.btn_p1.on_clicked(self.load_preset_1)
        self.btn_p2.on_clicked(self.load_preset_2)

        self.s_t.on_changed(self.update_plots)
        self.s_sigma.on_changed(self.recompute_all)
        self.s_p0.on_changed(self.recompute_all)
        self.s_x0.on_changed(self.recompute_all)

        self.s_xmin.on_changed(self.update_grid)
        self.s_xmax.on_changed(self.update_grid)
        self.s_tmax.on_changed(self.update_tmax)

        self.recompute_all()

    def update_grid(self, val=None):
        self.x_min = self.s_xmin.val
        self.x_max = self.s_xmax.val
        if self.x_max <= self.x_min + 5.0:
            self.x_max = self.x_min + 5.0
        self.recompute_all()

    def update_tmax(self, val=None):
        self.t_max = self.s_tmax.val
        self.s_t.valmax = self.t_max
        self.s_t.ax.set_xlim(0, self.t_max)
        self.recompute_all()

    def recompute_all(self, val=None):
        self.sigma0 = self.s_sigma.val
        self.p0 = self.s_p0.val
        self.x0 = self.s_x0.val

        self.x = np.linspace(self.x_min, self.x_max, self.Nx)
        self.update_plots()

    def update_plots(self, val=None):
        self.t_val = self.s_t.val
        res = compute_madelung_state(self.x, self.t_val, x0=self.x0, p0=self.p0, sigma0=self.sigma0)

        # ---------------- DASHBOARD UNIFICADO MADELUNG (PALETA INFERNO) ----------------

        # 1. Densidad rho(x,t) y Amplitud R(x,t)
        self.ax_rho.clear()
        self.ax_rho.set_facecolor(self.bg_card)
        self.ax_rho.plot(self.x, res['rho'], color='#fde725', linewidth=2, label=r'Densidad $\rho(x,t) = |\psi|^2$')
        self.ax_rho.plot(self.x, res['R'], color='#f8765c', linestyle='--', linewidth=1.5, label=r'Amplitud $R(x,t) = \sqrt{\rho}$')
        self.ax_rho.set_title(r'Densidad $\rho$ y Amplitud $R$', color='#fde725', fontsize=10, fontweight='bold')
        self.ax_rho.set_xlabel('x (u.a.)', color='white', fontsize=8)
        self.ax_rho.set_ylabel('Densidad', color='white', fontsize=8)

        # 2. Corriente de probabilidad j(x,t)
        self.ax_j.clear()
        self.ax_j.set_facecolor(self.bg_card)
        self.ax_j.plot(self.x, res['j'], color='#fca50a', linewidth=2, label=r'Corriente $j = \rho v$')
        self.ax_j.set_title(r'Corriente de Probabilidad $j(x,t)$', color='#fca50a', fontsize=10, fontweight='bold')
        self.ax_j.set_xlabel('x (u.a.)', color='white', fontsize=8)

        # 3. Fase theta(x,t)
        self.ax_phase.clear()
        self.ax_phase.set_facecolor(self.bg_card)
        self.ax_phase.plot(self.x, res['phase_mod'], color='#ed6925', linewidth=1.5, label=r'Fase Módulo $2\pi$')
        self.ax_phase.set_title(r'Fase $\theta(x,t) = \text{Arg}(\psi)$', color='#ed6925', fontsize=10, fontweight='bold')
        self.ax_phase.set_xlabel('x (u.a.)', color='white', fontsize=8)

        # 4. Mapa Espacio-Tiempo 2D (x,t) [Paleta Inferno]
        self.ax_map.clear()
        self.ax_map.set_facecolor(self.bg_card)
        t_grid_vals = np.linspace(0, self.t_max, 120)
        X_grid, T_grid = np.meshgrid(self.x, t_grid_vals)
        
        st_grid = self.sigma0 * np.sqrt(1.0 + (T_grid / (2.0 * self.sigma0**2))**2)
        xc_grid = self.x0 + self.p0 * T_grid
        Rho_grid = (1.0 / (np.sqrt(2 * np.pi) * st_grid)) * np.exp(-(X_grid - xc_grid)**2 / (2.0 * st_grid**2))

        mesh = self.ax_map.pcolormesh(X_grid, T_grid, Rho_grid, cmap='inferno', shading='auto')
        self.ax_map.axhline(self.t_val, color='#f8765c', linestyle='--', linewidth=1.5, label=f't = {self.t_val:.2f}')
        self.ax_map.plot(xc_grid[:, 0], t_grid_vals, color='#fde725', linestyle=':', linewidth=1.5, label=r'Centro $x_c(t)$')
        self.ax_map.axvline(res['xc'], color='#a0a0a0', linestyle=':', alpha=0.4, linewidth=1.2)
        self.ax_map.set_title(r'Mapa Espacio-Tiempo $(x,t)$', color='#f8765c', fontsize=10, fontweight='bold')
        self.ax_map.set_xlabel('x (u.a.)', color='white', fontsize=8)
        self.ax_map.set_ylabel('t (u.a.)', color='white', fontsize=8)

        # 5. Velocidad del fluido v(x,t)
        self.ax_v.clear()
        self.ax_v.set_facecolor(self.bg_card)
        self.ax_v.plot(self.x, res['v'], color='#fde725', linewidth=2, label=r'Velocidad $v = \frac{\nabla S}{m}$')
        self.ax_v.axhline(self.p0, color='#fca50a', linestyle='--', linewidth=1.2, label=f'v_grupo = p₀ = {self.p0:.2f}')
        self.ax_v.set_title(r'Velocidad del Fluido $v(x,t)$', color='#fde725', fontsize=10, fontweight='bold')
        self.ax_v.set_xlabel('x (u.a.)', color='white', fontsize=8)

        # 6. Potencial Cuántico Q(x,t)
        self.ax_Q.clear()
        self.ax_Q.set_facecolor(self.bg_card)
        self.ax_Q.plot(self.x, res['Q'], color='#b5367a', linewidth=2, label=r'Potencial Cuántico $Q = -\frac{\hbar^2}{2m}\frac{\nabla^2 R}{R}$')
        self.ax_Q.set_title(r'Potencial Cuántico $Q(x,t)$', color='#b5367a', fontsize=10, fontweight='bold')
        self.ax_Q.set_xlabel('x (u.a.)', color='white', fontsize=8)

        # 7. Fuerza Cuántica F_Q = - grad(Q)
        self.ax_FQ.clear()
        self.ax_FQ.set_facecolor(self.bg_card)
        self.ax_FQ.plot(self.x, res['F_Q'], color='#ed6925', linewidth=2, label=r'Fuerza Cuántica $F_Q = -\nabla Q$')
        self.ax_FQ.axhline(0, color='#a0a0a0', linestyle='-', linewidth=0.8, alpha=0.3)
        self.ax_FQ.set_title(r'Fuerza Cuántica $F_Q(x,t) = -\nabla Q$', color='#ed6925', fontsize=10, fontweight='bold')
        self.ax_FQ.set_xlabel('x (u.a.)', color='white', fontsize=8)

        # 8. Energía Hidrodinámica Total E = T + Q
        self.ax_E.clear()
        self.ax_E.set_facecolor(self.bg_card)
        self.ax_E.plot(self.x, res['E_hydro'], color='#fca50a', linewidth=2, label=r'$E = \frac{1}{2}m v^2 + Q$')
        self.ax_E.plot(self.x, res['T_hydro'], color='#fde725', linestyle='--', linewidth=1.5, label=r'$T = \frac{1}{2}m v^2$')
        self.ax_E.plot(self.x, res['Q'], color='#b5367a', linestyle=':', linewidth=1.5, label=r'$Q$')
        self.ax_E.set_title(r'Balance Energético $E = T + Q$', color='#fca50a', fontsize=10, fontweight='bold')
        self.ax_E.set_xlabel('x (u.a.)', color='white', fontsize=8)

        # DIBUJAR LÍNEA VERTICAL SUTIL EN x = x_c(t) Y FORMATO DE LEYENDAS / GRILLAS
        first_spatial = True
        for ax in self.spatial_axes:
            # Línea vertical punteada en x_c(t) atravesando todos los gráficos espaciales
            if first_spatial:
                ax.axvline(res['xc'], color='#a0a0a0', linestyle=':', alpha=0.4, linewidth=1.2, label=r'Centro $x_c(t)$')
                first_spatial = False
            else:
                ax.axvline(res['xc'], color='#a0a0a0', linestyle=':', alpha=0.4, linewidth=1.2)

            ax.grid(True, color='#28183d', linestyle='--', alpha=0.6)
            ax.legend(loc='upper right', facecolor=self.bg_card, edgecolor='#781c6d', fontsize=7)

        self.ax_map.grid(True, color='#28183d', linestyle='--', alpha=0.4)
        self.ax_map.legend(loc='upper right', facecolor=self.bg_card, edgecolor='#781c6d', fontsize=7)

        self.fig.suptitle("LABORATORIO PYTHON 11: FORMULACIÓN HIDRODINÁMICA DE MADELUNG (PAQUETE LIBRE EN 1D)",
                          color='white', fontsize=12, fontweight='bold', y=0.98)
        self.fig.canvas.draw_idle()

    def toggle_anim(self, event=None):
        self.is_animating = not self.is_animating
        self.btn_play.label.set_text('❚❚ Pausa' if self.is_animating else '▶ Animación')
        if self.is_animating:
            self.run_animation()

    def run_animation(self):
        def animate(frame):
            if not self.is_animating:
                return
            t_next = self.s_t.val + 0.08
            if t_next > self.t_max:
                t_next = 0.0
            self.s_t.set_val(t_next)

        self.anim = FuncAnimation(self.fig, animate, interval=50, cache_frame_data=False)

    def reset_params(self, event=None):
        self.is_animating = False
        self.btn_play.label.set_text('▶ Animación')
        self.s_t.set_val(0.0)
        self.s_sigma.set_val(2.0)
        self.s_p0.set_val(2.0)
        self.s_x0.set_val(-10.0)
        self.s_xmin.set_val(-35.0)
        self.s_xmax.set_val(35.0)
        self.s_tmax.set_val(6.0)

    def load_preset_1(self, event=None):
        """Preset 1: Paquete Rápido y Concentrado"""
        self.s_sigma.set_val(1.0)
        self.s_p0.set_val(4.0)
        self.s_x0.set_val(-15.0)

    def load_preset_2(self, event=None):
        """Preset 2: Paquete Lento y Ancho"""
        self.s_sigma.set_val(3.5)
        self.s_p0.set_val(1.0)
        self.s_x0.set_val(-5.0)


def export_static_sample(output_path="/workspace/scratch/madelung_paquete_libre_inferno_sample.png"):
    """
    Genera una imagen de muestra para testing.
    """
    run_madelung_sympy()
    x = np.linspace(-35, 35, 2049)
    res = compute_madelung_state(x, t=2.0, x0=-10.0, p0=2.0, sigma0=2.0)

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 4, figsize=(16, 9), facecolor='#0a0612')
    fig.subplots_adjust(hspace=0.35, wspace=0.32, left=0.06, right=0.96, top=0.91, bottom=0.08)

    # Subplots
    axes[0, 0].plot(x, res['rho'], color='#fde725', label=r'$\rho$')
    axes[0, 0].plot(x, res['R'], color='#f8765c', linestyle='--', label=r'$R$')
    axes[0, 0].set_title(r'Densidad $\rho$ y Amplitud $R$', color='#fde725')
    axes[0, 0].legend()

    axes[0, 1].plot(x, res['j'], color='#fca50a', label=r'$j$')
    axes[0, 1].set_title(r'Corriente $j(x,t)$', color='#fca50a')
    axes[0, 1].legend()

    axes[0, 2].plot(x, res['phase_mod'], color='#ed6925', label=r'$\theta$')
    axes[0, 2].set_title(r'Fase $\theta(x,t)$', color='#ed6925')
    axes[0, 2].legend()

    # Map Espacio-Tiempo con inferno
    t_arr = np.linspace(0, 6.0, 100)
    X_grid, T_grid = np.meshgrid(x, t_arr)
    st_grid = 2.0 * np.sqrt(1.0 + (T_grid / (2.0 * 2.0**2))**2)
    xc_grid = -10.0 + 2.0 * T_grid
    Rho_grid = (1.0 / (np.sqrt(2 * np.pi) * st_grid)) * np.exp(-(X_grid - xc_grid)**2 / (2.0 * st_grid**2))
    axes[0, 3].pcolormesh(X_grid, T_grid, Rho_grid, cmap='inferno', shading='auto')
    axes[0, 3].set_title(r'Espacio-Tiempo $(x,t)$ [Inferno]', color='#f8765c')

    axes[1, 0].plot(x, res['v'], color='#fde725', label=r'$v$')
    axes[1, 0].set_title(r'Velocidad $v(x,t)$', color='#fde725')
    axes[1, 0].legend()

    axes[1, 1].plot(x, res['Q'], color='#b5367a', label=r'$Q$')
    axes[1, 1].set_title(r'Potencial Cuántico $Q(x,t)$', color='#b5367a')
    axes[1, 1].legend()

    axes[1, 2].plot(x, res['F_Q'], color='#ed6925', label=r'$F_Q$')
    axes[1, 2].set_title(r'Fuerza Cuántica $F_Q$', color='#ed6925')
    axes[1, 2].legend()

    axes[1, 3].plot(x, res['E_hydro'], color='#fca50a', label=r'$E$')
    axes[1, 3].plot(x, res['T_hydro'], color='#fde725', linestyle='--', label=r'$T$')
    axes[1, 3].plot(x, res['Q'], color='#b5367a', linestyle=':', label=r'$Q$')
    axes[1, 3].set_title(r'Balance Energético', color='#fca50a')
    axes[1, 3].legend()

    for ax in axes.flat:
        ax.set_facecolor('#160c28')
        ax.axvline(res['xc'], color='#a0a0a0', linestyle=':', alpha=0.4, linewidth=1.2)
        ax.grid(True, color='#28183d', linestyle='--')

    plt.suptitle("LABORATORIO PYTHON 11: FORMULACIÓN HIDRODINÁMICA DE MADELUNG (PROPUESTA I: INFERNO)", color='white')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] Sample saved to {output_path}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        export_static_sample()
    else:
        app = MadelungFreePacketApp()
        plt.show()
