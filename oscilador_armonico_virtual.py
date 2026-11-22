"""
================================================================================
LABORATORIO PYTHON 15: OSCILADOR ARMÓNICO CUÁNTICO
================================================================================
Materia: Mecánica Cuántica
Tema: Operadores Escalera, Estados Coherentes y Evolución Temporal
Paleta: Viridis Completa (Estética Unificada Teal/Verde/Amarillo/Morado)
Librerías: NumPy, Matplotlib, SciPy
================================================================================
"""

import sys
import os
import math
import numpy as np
from scipy.special import eval_hermite
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons

# Forzado de Backend interactivo para ejecuciones en terminal / VS Code
if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] != '--test'):
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass


def qho_psi_n(n, x, m=1.0, omega=1.0, hbar=1.0):
    """Calcula la autofunción del oscilador armónico psi_n(x)."""
    alpha = np.sqrt(m * omega / hbar)
    norm = 1.0 / np.sqrt((2.0**n) * math.factorial(n) * np.sqrt(np.pi / (alpha**2)))
    xi = alpha * x
    return norm * eval_hermite(n, xi) * np.exp(-0.5 * xi**2)


def qho_energy(n, omega=1.0, hbar=1.0):
    """Energía del nivel n: E_n = hbar * omega * (n + 0.5)."""
    return hbar * omega * (n + 0.5)


class QHOLabApp:
    """
    Aplicación Interactiva Unificada en Paleta VIRIDIS para el Laboratorio 15.
    Oscilador Armónico Cuántico: Operadores Escalera y Evolución Temporal.
    """
    def __init__(self):
        # Parámetros iniciales
        self.m = 1.0
        self.omega = 1.0
        self.hbar = 1.0

        self.n_puro = 1
        self.n1_sup = 0
        self.n2_sup = 1
        self.c1_val = 1.0 / np.sqrt(2.0)
        self.c2_val = 1.0 / np.sqrt(2.0)
        self.alpha_coh = 2.0
        self.ladder_n = 2

        self.mode_str = 'Puro |n>'
        self.is_animating = False

        self.t_val = 0.0
        self.dt = 0.05
        self.t_max = 20.0

        self.x_max = 6.0
        self.Nx = 1000
        self.x_grid = np.linspace(-self.x_max, self.x_max, self.Nx)
        self.dx = self.x_grid[1] - self.x_grid[0]

        # Historial de trayectoria <x>(t)
        self.history_t = []
        self.history_x_exp = []
        self.history_x_clas = []

        plt.style.use('dark_background')

        # Paleta Viridis Completa
        self.bg_fig = '#07151a'
        self.bg_card = '#0e242b'
        self.bg_control = '#14353f'

        self.fig = plt.figure(figsize=(16, 9.8), facecolor=self.bg_fig)

        # 4 Subplots unificados (2x2)
        self.ax_wave = self.fig.add_subplot(2, 2, 1, facecolor=self.bg_card)
        self.ax_traj = self.fig.add_subplot(2, 2, 2, facecolor=self.bg_card)
        self.ax_ladder = self.fig.add_subplot(2, 2, 3, facecolor=self.bg_card)
        self.ax_info = self.fig.add_subplot(2, 2, 4, facecolor=self.bg_card)

        plt.subplots_adjust(left=0.07, right=0.95, top=0.92, bottom=0.28, wspace=0.25, hspace=0.34)

        # Controles y Sliders
        ax_t     = self.fig.add_axes([0.12, 0.18, 0.32, 0.022], facecolor=self.bg_control)
        ax_n1    = self.fig.add_axes([0.12, 0.13, 0.32, 0.022], facecolor=self.bg_control)
        ax_n2    = self.fig.add_axes([0.12, 0.08, 0.32, 0.022], facecolor=self.bg_control)

        ax_alpha = self.fig.add_axes([0.58, 0.18, 0.32, 0.022], facecolor=self.bg_control)
        ax_omega = self.fig.add_axes([0.58, 0.13, 0.32, 0.022], facecolor=self.bg_control)

        ax_radio     = self.fig.add_axes([0.52, 0.02, 0.18, 0.09], facecolor=self.bg_card)
        ax_btn_play  = self.fig.add_axes([0.72, 0.05, 0.10, 0.045], facecolor='#21918c')
        ax_btn_reset = self.fig.add_axes([0.83, 0.05, 0.10, 0.045], facecolor='#3b528b')

        # Botones de Operadores Escalera
        ax_btn_annih = self.fig.add_axes([0.12, 0.025, 0.16, 0.038], facecolor='#35b779')
        ax_btn_creat = self.fig.add_axes([0.30, 0.025, 0.16, 0.038], facecolor='#fde725')

        self.s_t     = Slider(ax_t,     'Tiempo t',        0.0, self.t_max, valinit=self.t_val, color='#f8765c')
        self.s_n1    = Slider(ax_n1,    'Nivel n / n₁',    0, 10, valinit=self.n_puro, valstep=1, color='#35b779')
        self.s_n2    = Slider(ax_n2,    'Nivel n₂',        0, 10, valinit=self.n2_sup, valstep=1, color='#21918c')

        self.s_alpha = Slider(ax_alpha, 'Coherente α',    0.0, 4.0, valinit=self.alpha_coh, color='#40e0d0')
        self.s_omega = Slider(ax_omega, 'Frecuencia ω',  0.5, 3.0, valinit=self.omega, color='#3b528b')

        self.radio_mode = RadioButtons(
            ax_radio,
            ('Puro |n>', 'Superposición c₁|n₁>+c₂|n₂>', 'Coherente |α>', 'Operadores Escalera'),
            active=0, activecolor='#21918c'
        )
        for txt in self.radio_mode.labels:
            txt.set_color('white')
            txt.set_fontsize(8.0)

        self.btn_play  = Button(ax_btn_play,  '▶ Evolucionar', color='#21918c', hovercolor='#35b779')
        self.btn_reset = Button(ax_btn_reset, '↺ Reiniciar', color='#3b528b', hovercolor='#21918c')

        self.btn_annih = Button(ax_btn_annih, 'â |n> (Aniquilación)', color='#35b779', hovercolor='#21918c')
        self.btn_creat = Button(ax_btn_creat, 'â† |n> (Creación)', color='#fde725', hovercolor='#35b779')

        self.btn_play.on_clicked(self.toggle_anim)
        self.btn_reset.on_clicked(self.reset_params)
        self.radio_mode.on_clicked(self.on_mode_changed)

        self.btn_annih.on_clicked(self.apply_annihilation)
        self.btn_creat.on_clicked(self.apply_creation)

        self.s_t.on_changed(self.on_slider_t_changed)
        self.s_n1.on_changed(self.on_params_changed)
        self.s_n2.on_changed(self.on_params_changed)
        self.s_alpha.on_changed(self.on_params_changed)
        self.s_omega.on_changed(self.on_params_changed)

        self.update_plots()

    def toggle_anim(self, event=None):
        self.is_animating = not self.is_animating
        self.btn_play.label.set_text('❚❚ Pausa' if self.is_animating else '▶ Evolucionar')
        self.btn_play.ax.set_facecolor('#fde725' if self.is_animating else '#21918c')
        if self.is_animating:
            self.run_animation()

    def run_animation(self):
        def animate(frame):
            if self.is_animating:
                self.t_val += self.dt
                if self.t_val > self.t_max:
                    self.t_val = 0.0
                    self.history_t.clear()
                    self.history_x_exp.clear()
                    self.history_x_clas.clear()

                self.s_t.set_val(self.t_val)

        self.anim = FuncAnimation(self.fig, animate, interval=30, cache_frame_data=False)

    def reset_params(self, event=None):
        self.is_animating = False
        self.btn_play.label.set_text('▶ Evolucionar')
        self.btn_play.ax.set_facecolor('#21918c')

        self.t_val = 0.0
        self.s_t.set_val(0.0)
        self.s_n1.set_val(1)
        self.s_n2.set_val(2)
        self.s_alpha.set_val(2.0)
        self.s_omega.set_val(1.0)

        self.history_t.clear()
        self.history_x_exp.clear()
        self.history_x_clas.clear()

        self.update_plots()

    def on_mode_changed(self, label):
        self.mode_str = label
        self.history_t.clear()
        self.history_x_exp.clear()
        self.history_x_clas.clear()
        self.update_plots()

    def on_slider_t_changed(self, val):
        self.t_val = val
        if not self.is_animating:
            self.update_plots()

    def on_params_changed(self, val=None):
        self.omega = self.s_omega.val
        self.n_puro = int(self.s_n1.val)
        self.n1_sup = int(self.s_n1.val)
        self.n2_sup = int(self.s_n2.val)
        self.alpha_coh = self.s_alpha.val

        if not self.is_animating:
            self.history_t.clear()
            self.history_x_exp.clear()
            self.history_x_clas.clear()
            self.update_plots()

    def apply_annihilation(self, event=None):
        n = int(self.s_n1.val)
        if n > 0:
            self.s_n1.set_val(n - 1)
        self.mode_str = 'Operadores Escalera'
        self.radio_mode.set_active(3)
        self.update_plots()

    def apply_creation(self, event=None):
        n = int(self.s_n1.val)
        if n < 10:
            self.s_n1.set_val(n + 1)
        self.mode_str = 'Operadores Escalera'
        self.radio_mode.set_active(3)
        self.update_plots()

    def compute_state(self):
        t = self.t_val
        x = self.x_grid
        dx = self.dx
        omega = self.omega
        hbar = self.hbar
        m = self.m

        V_pot = 0.5 * m * (omega**2) * (x**2)

        if 'Puro' in self.mode_str:
            n = int(self.s_n1.val)
            E_n = qho_energy(n, omega=omega, hbar=hbar)
            psi_0 = qho_psi_n(n, x, m=m, omega=omega, hbar=hbar)
            psi_t = psi_0 * np.exp(-1j * E_n * t / hbar)
            prob_density = np.abs(psi_t)**2
            E_mean = E_n
            x_exp = np.sum(prob_density * x) * dx
            x_clas = 0.0

        elif 'Superposición' in self.mode_str:
            n1 = int(self.s_n1.val)
            n2 = int(self.s_n2.val)
            c1 = 1.0 / np.sqrt(2.0)
            c2 = 1.0 / np.sqrt(2.0)

            E1 = qho_energy(n1, omega=omega, hbar=hbar)
            E2 = qho_energy(n2, omega=omega, hbar=hbar)

            psi1 = qho_psi_n(n1, x, m=m, omega=omega, hbar=hbar)
            psi2 = qho_psi_n(n2, x, m=m, omega=omega, hbar=hbar)

            psi_t = c1 * psi1 * np.exp(-1j * E1 * t / hbar) + c2 * psi2 * np.exp(-1j * E2 * t / hbar)
            prob_density = np.abs(psi_t)**2
            E_mean = (c1**2) * E1 + (c2**2) * E2
            x_exp = np.sum(prob_density * x) * dx

            # Trayectoria teórica de superposición
            delta_n = abs(n2 - n1)
            x_clas = 0.0
            if delta_n == 1:
                n_min = min(n1, n2)
                x_scale = np.sqrt(hbar / (m * omega))
                x_clas = 2.0 * c1 * c2 * np.sqrt((n_min + 1) / 2.0) * x_scale * np.cos((E2 - E1) * t / hbar)

        elif 'Coherente' in self.mode_str:
            alpha = self.s_alpha.val
            N_max = 15
            psi_t = np.zeros_like(x, dtype=complex)
            E_mean = 0.0

            for n in range(N_max):
                cn = np.exp(-0.5 * (alpha**2)) * (alpha**n) / np.sqrt(math.factorial(n))
                En = qho_energy(n, omega=omega, hbar=hbar)
                psi_n = qho_psi_n(n, x, m=m, omega=omega, hbar=hbar)
                psi_t += cn * psi_n * np.exp(-1j * En * t / hbar)
                E_mean += (cn**2) * En

            prob_density = np.abs(psi_t)**2
            x_exp = np.sum(prob_density * x) * dx

            # Trayectoria clásica exacta del estado coherente
            x0_scale = np.sqrt(hbar / (m * omega))
            x_clas = np.sqrt(2.0) * alpha * x0_scale * np.cos(omega * t)

        else:  # Operadores Escalera
            n = int(self.s_n1.val)
            E_n = qho_energy(n, omega=omega, hbar=hbar)
            psi_0 = qho_psi_n(n, x, m=m, omega=omega, hbar=hbar)
            psi_t = psi_0 * np.exp(-1j * E_n * t / hbar)
            prob_density = np.abs(psi_t)**2
            E_mean = E_n
            x_exp = np.sum(prob_density * x) * dx
            x_clas = 0.0

        norm_val = np.sum(prob_density) * dx

        return {
            'psi_t': psi_t,
            'prob_density': prob_density,
            'V_pot': V_pot,
            'E_mean': E_mean,
            'norm_val': norm_val,
            'x_exp': x_exp,
            'x_clas': x_clas
        }

    def update_plots(self):
        res = self.compute_state()

        x = self.x_grid
        psi_t = res['psi_t']
        prob = res['prob_density']
        V_pot = res['V_pot']
        E_mean_val = res['E_mean']

        # Actualizar historial de trayectoria
        self.history_t.append(self.t_val)
        self.history_x_exp.append(res['x_exp'])
        self.history_x_clas.append(res['x_clas'])

        # ---------------- 1. SUBPLOT 1: POTENCIAL V(x), NIVELES Y PAQUETE |Psi|^2 ----------------
        self.ax_wave.clear()
        self.ax_wave.set_facecolor(self.bg_card)

        self.ax_wave.plot(x, V_pot, color='#b5367a', linewidth=2.2, label=r'Potencial $V(x) = \frac{1}{2}m\omega^2 x^2$')
        self.ax_wave.axhline(E_mean_val, color='#fde725', linestyle='--', linewidth=1.5,
                             label=f'Energía Media <H> = {E_mean_val:.3f}')

        self.ax_wave.plot(x, prob + E_mean_val, color='#fde725', linewidth=2.5, label=r'Densidad $|\Psi(x,t)|^2$')
        self.ax_wave.fill_between(x, E_mean_val, prob + E_mean_val, color='#fde725', alpha=0.25)
        self.ax_wave.plot(x, np.real(psi_t) + E_mean_val, color='#40e0d0', linewidth=1.2, alpha=0.7, label=r'$\text{Re}[\Psi]$')

        self.ax_wave.set_xlim([-self.x_max, self.x_max])
        self.ax_wave.set_ylim([-0.2, max(V_pot[0], E_mean_val + 2.0)])
        self.ax_wave.set_xlabel('Posición x (u.a.)', color='white', fontsize=8)
        self.ax_wave.set_ylabel('Energía / Amplitud', color='white', fontsize=8)
        self.ax_wave.set_title(f'Evolución Temporal en el Oscilador ({self.mode_str})', color='#35b779', fontsize=10, fontweight='bold')
        self.ax_wave.grid(True, color='#1f2937', linestyle='--', alpha=0.5)
        self.ax_wave.legend(loc='upper right', facecolor=self.bg_card, edgecolor='#21918c', fontsize=7)

        # ---------------- 2. SUBPLOT 2: TRAYECTORIA <x>(t) Y COMPARACIÓN CLÁSICA ----------------
        self.ax_traj.clear()
        self.ax_traj.set_facecolor(self.bg_card)

        if len(self.history_t) > 0:
            self.ax_traj.plot(self.history_t, self.history_x_exp, color='#40e0d0', linewidth=2.2, label=r'Valor Esperado $\langle x \rangle(t)$')
            if 'Coherente' in self.mode_str or 'Superposición' in self.mode_str:
                self.ax_traj.plot(self.history_t, self.history_x_clas, color='#fde725', linestyle='--', linewidth=1.5, label=r'Trayectoria Clásica $x_{\text{cl}}(t)$')

        self.ax_traj.axhline(0, color='#a0a0a0', linestyle=':', alpha=0.5)
        self.ax_traj.set_xlim([0, max(self.t_max, max(self.history_t) if len(self.history_t) > 0 else 1.0)])
        self.ax_traj.set_ylim([-self.x_max * 0.8, self.x_max * 0.8])
        self.ax_traj.set_xlabel('Tiempo t (u.a.)', color='white', fontsize=8)
        self.ax_traj.set_ylabel('Posición (u.a.)', color='white', fontsize=8)
        self.ax_traj.set_title(r'Comportamiento de $\langle x \rangle(t)$ vs Clásico', color='#40e0d0', fontsize=10, fontweight='bold')
        self.ax_traj.grid(True, color='#1f2937', linestyle='--', alpha=0.5)
        self.ax_traj.legend(loc='upper right', facecolor=self.bg_card, edgecolor='#40e0d0', fontsize=7)

        # ---------------- 3. SUBPLOT 3: ACCIÓN DE OPERADORES ESCALERA ----------------
        self.ax_ladder.clear()
        self.ax_ladder.set_facecolor(self.bg_card)

        n_curr = int(self.s_n1.val)
        psi_n = qho_psi_n(n_curr, x, m=self.m, omega=self.omega, hbar=self.hbar)

        self.ax_ladder.plot(x, psi_n, color='#35b779', linewidth=2.0, label=f'Estado Original |{n_curr}>')

        # Operador Aniquilación a|n> = sqrt(n)|n-1>
        if n_curr > 0:
            psi_annih = np.sqrt(n_curr) * qho_psi_n(n_curr - 1, x, m=self.m, omega=self.omega, hbar=self.hbar)
            self.ax_ladder.plot(x, psi_annih, color='#40e0d0', linestyle='--', linewidth=1.5,
                                label=f'a|{n_curr}> = sqrt({n_curr})|{n_curr-1}>')
        else:
            self.ax_ladder.axhline(0, color='#40e0d0', linestyle='--', label=r'a|0> = 0')

        # Operador Creación a^\dagger|n> = sqrt(n+1)|n+1>
        psi_creat = np.sqrt(n_curr + 1) * qho_psi_n(n_curr + 1, x, m=self.m, omega=self.omega, hbar=self.hbar)
        self.ax_ladder.plot(x, psi_creat, color='#fde725', linestyle=':', linewidth=1.8,
                            label=f'a†|{n_curr}> = sqrt({n_curr+1})|{n_curr+1}>')

        self.ax_ladder.set_xlim([-self.x_max, self.x_max])
        self.ax_ladder.set_xlabel('Posición x (u.a.)', color='white', fontsize=8)
        self.ax_ladder.set_ylabel('Amplitud de Estado', color='white', fontsize=8)
        self.ax_ladder.set_title(r'Acción de Operadores Escalera $\hat{a}$ y $\hat{a}^\dagger$', color='#fde725', fontsize=10, fontweight='bold')
        self.ax_ladder.grid(True, color='#1f2937', linestyle='--', alpha=0.5)
        self.ax_ladder.legend(loc='upper right', facecolor=self.bg_card, edgecolor='#fde725', fontsize=7)

        # ---------------- 4. SUBPLOT 4: PANEL DE MÉTRICAS Y VERIFICACIONES ----------------
        self.ax_info.clear()
        self.ax_info.axis('off')

        info_text = (
            f"=== VERIFICACIÓN DEL OSCILADOR ARMÓNICO ({self.mode_str}) ===\n\n"
            f" • Tiempo t              : {self.t_val:.3f} s\n"
            f" • Frecuencia ω         : {self.omega:.2f} rad/s\n"
            f" • Norma Total ∫|Ψ|² dx  : {res['norm_val']:.6f} (Conservación 1.0000)\n"
            f" • Energía Media ⟨H⟩     : {res['E_mean']:.5f} (Conservada)\n\n"
            "--- Propiedades del Estado ---\n"
            f" • Valor Esperado ⟨x⟩    : {res['x_exp']:.4f}\n"
            f" • Trayectoria Clásica   : {res['x_clas']:.4f}\n"
        )

        if 'Superposición' in self.mode_str:
            n1 = int(self.s_n1.val)
            n2 = int(self.s_n2.val)
            delta_E = abs(qho_energy(n2, self.omega) - qho_energy(n1, self.omega))
            info_text += (
                f"\n--- Batido Cuántico ---\n"
                f" • Frecuencia de Batido  : Δω = {delta_E:.3f} rad/s\n"
                f" • Período de Oscilación : T = {2*np.pi/delta_E:.3f} s\n"
            )
        elif 'Coherente' in self.mode_str:
            alpha = self.s_alpha.val
            info_text += (
                f"\n--- Estado Coherente |α⟩ ---\n"
                f" • Parámetro α          : {alpha:.2f}\n"
                f" • Fotones/Quantos <n>   : |α|² = {alpha**2:.2f}\n"
                f" • Indeterminación Δx    : Constante = 0.7071 (Mínima)\n"
            )

        self.ax_info.text(0.05, 0.95, info_text, color='#e2e8f0', fontsize=8.5, fontfamily='monospace',
                          verticalalignment='top',
                          bbox=dict(boxstyle='round,pad=0.8', facecolor=self.bg_control, edgecolor='#35b779', alpha=0.95))

        self.fig.suptitle("LABORATORIO PYTHON 15: OSCILADOR ARMÓNICO CUÁNTICO (PALETA VIRIDIS)",
                          color='white', fontsize=12, fontweight='bold', y=0.98)
        self.fig.canvas.draw_idle()


def export_static_sample(output_path="/workspace/scratch/oscilador_armonico_sample.png"):
    """Genera una imagen estática para verificación headless."""
    x = np.linspace(-5, 5, 500)
    psi0 = qho_psi_n(0, x)
    psi1 = qho_psi_n(1, x)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#07151a')
    ax.set_facecolor('#0e242b')
    ax.plot(x, psi0**2, color='#fde725', label=r'$|\psi_0(x)|^2$')
    ax.plot(x, psi1**2, color='#35b779', label=r'$|\psi_1(x)|^2$')
    ax.set_title("Muestra del Oscilador Armónico Cuántico (Viridis)", color='white')
    ax.legend()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] Sample saved to {output_path}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        export_static_sample()
    else:
        app = QHOLabApp()
        plt.show()
