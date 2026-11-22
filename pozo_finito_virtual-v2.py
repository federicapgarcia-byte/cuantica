"""
================================================================================
LABORATORIO PYTHON 14: CUANDO LAS PAREDES PERMITEN PENETRACIÓN (POZO FINITO)
================================================================================
Materia: Mecánica Cuántica
Tema: Pozo Cuadrado Finito de Potencial (Estados Ligados, Penetrabilidad y Comparación con Pozo Infinito)
Paleta: Viridis Completa (Estética Unificada Teal/Verde/Amarillo/Morado)
Librerías: NumPy, Matplotlib, SciPy
================================================================================
"""

import sys
import os
import numpy as np
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Forzado de Backend interactivo para ejecuciones en terminal / VS Code
if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] != '--test'):
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass


def compute_finite_well_states(V0_ratio=10.0, a_ratio=1.0, hbar=1.0, m=1.0):
    """
    Calcula los estados ligados del pozo cuadrado finito de ancho 2a (-a <= x <= a).
    Parámetros:
      V0_ratio: Profundidad del pozo en unidades de E_ref = hbar^2 pi^2 / (8 m a^2).
      a_ratio: Semiancho a del pozo.
    Devuelve:
      diccionario con estados z_n, w_n, E_n, paridades y métricas.
    """
    a = a_ratio
    E_ref = (hbar**2 * np.pi**2) / (8.0 * m * a**2)
    V0 = V0_ratio * E_ref
    z0 = (a / hbar) * np.sqrt(2.0 * m * V0)

    states = []
    max_candidates = int(np.floor(z0 / (np.pi / 2.0))) + 1

    for n in range(1, max_candidates + 1):
        z_min = (n - 1) * np.pi / 2.0 + 1e-6
        z_max = min(n * np.pi / 2.0 - 1e-6, z0 - 1e-6)
        if z_min >= z_max:
            continue

        is_even = (n % 2 == 1)
        if is_even:
            f = lambda z: np.sqrt(z0**2 - z**2) - z * np.tan(z)
        else:
            f = lambda z: np.sqrt(z0**2 - z**2) + z / np.tan(z)

        try:
            if f(z_min) * f(z_max) <= 0:
                z_root = opt.brentq(f, z_min, z_max)
                w_root = np.sqrt(z0**2 - z_root**2)
                E_n = (hbar**2 * z_root**2) / (2.0 * m * a**2)
                E_inf = (hbar**2 * n**2 * np.pi**2) / (8.0 * m * a**2)

                states.append({
                    'n': n,
                    'z': z_root,
                    'w': w_root,
                    'E': E_n,
                    'E_inf': E_inf,
                    'is_even': is_even,
                    'parity': 'Par' if is_even else 'Impar',
                    'nodes': n - 1
                })
        except Exception:
            pass

    return {
        'states': states,
        'z0': z0,
        'V0': V0,
        'E_ref': E_ref,
        'a': a,
        'num_states': len(states)
    }


def compute_wavefunctions(state, a, x_grid):
    """
    Calcula la función de onda psi_n(x) y densidad |psi_n(x)|^2 normalizadas
    para un estado ligado del pozo finito y la compara con el pozo infinito.
    """
    z = state['z']
    w = state['w']
    n = state['n']
    k = z / a
    kappa = w / a
    is_even = state['is_even']

    # --- Pozo Finito ---
    psi_unnorm = np.zeros_like(x_grid)

    mask_in = np.abs(x_grid) <= a
    x_in = x_grid[mask_in]

    if is_even:
        psi_unnorm[mask_in] = np.cos(k * x_in)
        boundary_val = np.cos(z)
    else:
        psi_unnorm[mask_in] = np.sin(k * x_in)
        boundary_val = np.sin(z)

    mask_right = x_grid > a
    x_r = x_grid[mask_right]
    psi_unnorm[mask_right] = boundary_val * np.exp(-kappa * (x_r - a))

    mask_left = x_grid < -a
    x_l = x_grid[mask_left]
    if is_even:
        psi_unnorm[mask_left] = boundary_val * np.exp(-kappa * (-x_l - a))
    else:
        psi_unnorm[mask_left] = -boundary_val * np.exp(-kappa * (-x_l - a))

    dx = x_grid[1] - x_grid[0]
    norm = np.sqrt(np.sum(psi_unnorm**2) * dx)
    psi_fin = psi_unnorm / norm
    prob_fin = psi_fin**2

    P_int = np.sum(prob_fin[mask_in]) * dx
    P_ext = np.sum(prob_fin[~mask_in]) * dx

    # --- Pozo Infinito de Ancho 2a ---
    psi_inf = np.zeros_like(x_grid)
    if is_even:
        psi_inf[mask_in] = (1.0 / np.sqrt(a)) * np.cos((n * np.pi * x_in) / (2.0 * a))
    else:
        psi_inf[mask_in] = (1.0 / np.sqrt(a)) * np.sin((n * np.pi * x_in) / (2.0 * a))

    prob_inf = psi_inf**2

    return {
        'psi_fin': psi_fin,
        'prob_fin': prob_fin,
        'psi_inf': psi_inf,
        'prob_inf': prob_inf,
        'P_int': P_int,
        'P_ext': P_ext,
        'mask_in': mask_in,
        'mask_ext': ~mask_in
    }


class FiniteSquareWellApp:
    """
    Aplicación Interactiva Unificada en Paleta VIRIDIS para el Laboratorio 14.
    Pozo de Potencial Finito: "Cuando las paredes permiten penetración".
    """
    def __init__(self):
        # Parámetros iniciales
        self.V0_ratio_init = 10.0
        self.a_ratio_init = 1.0
        self.n_select_init = 1
        self.x_range_init = 4.0   # x_max = 4.0 * a
        self.show_inf_compare = True

        plt.style.use('dark_background')

        # Paleta Viridis Completa (Estética unificada)
        self.bg_fig = '#07151a'       # Fondo profundo viridis
        self.bg_card = '#0e242b'      # Violeta-teal oscuro
        self.bg_control = '#14353f'   # Verde-azul medio

        self.fig = plt.figure(figsize=(16, 9.8), facecolor=self.bg_fig)

        # 4 Subplots unificados (2 filas x 2 columnas)
        self.ax_pot  = self.fig.add_subplot(2, 2, 1, facecolor=self.bg_card)
        self.ax_eq   = self.fig.add_subplot(2, 2, 2, facecolor=self.bg_card)
        self.ax_psi  = self.fig.add_subplot(2, 2, 3, facecolor=self.bg_card)
        self.ax_prob = self.fig.add_subplot(2, 2, 4, facecolor=self.bg_card)

        plt.subplots_adjust(left=0.07, right=0.95, top=0.92, bottom=0.28, wspace=0.25, hspace=0.34)

        # Controles y Sliders
        ax_V0    = self.fig.add_axes([0.12, 0.18, 0.32, 0.022], facecolor=self.bg_control)
        ax_a     = self.fig.add_axes([0.12, 0.13, 0.32, 0.022], facecolor=self.bg_control)
        ax_n     = self.fig.add_axes([0.12, 0.08, 0.32, 0.022], facecolor=self.bg_control)

        ax_xrange = self.fig.add_axes([0.58, 0.18, 0.32, 0.022], facecolor=self.bg_control)

        ax_btn_comp  = self.fig.add_axes([0.58, 0.08, 0.18, 0.045], facecolor='#21918c')
        ax_btn_reset = self.fig.add_axes([0.78, 0.08, 0.12, 0.045], facecolor='#3b528b')

        self.s_V0     = Slider(ax_V0,     'Profundidad V₀/E_ref', 0.5, 30.0, valinit=self.V0_ratio_init, color='#35b779')
        self.s_a      = Slider(ax_a,      'Semiancho a/a_ref',   0.5, 3.0,  valinit=self.a_ratio_init,  color='#21918c')
        self.s_n      = Slider(ax_n,      'Estado Ligado n',      1, 8,      valinit=self.n_select_init, valstep=1, color='#fde725')

        self.s_xrange = Slider(ax_xrange, 'Ventana Espacial x_max/a', 1.5, 8.0, valinit=self.x_range_init, color='#40e0d0')

        self.btn_comp  = Button(ax_btn_comp,  'Pozo Infinito: ON', color='#21918c', hovercolor='#35b779')
        self.btn_reset = Button(ax_btn_reset, '↺ Reiniciar',      color='#3b528b', hovercolor='#21918c')

        self.btn_comp.on_clicked(self.toggle_inf_compare)
        self.btn_reset.on_clicked(self.reset_params)

        self.s_V0.on_changed(self.update_plots)
        self.s_a.on_changed(self.update_plots)
        self.s_n.on_changed(self.update_plots)
        self.s_xrange.on_changed(self.update_plots)

        self.update_plots()

    def toggle_inf_compare(self, event=None):
        self.show_inf_compare = not self.show_inf_compare
        self.btn_comp.label.set_text(f'Pozo Infinito: {"ON" if self.show_inf_compare else "OFF"}')
        self.btn_comp.ax.set_facecolor('#21918c' if self.show_inf_compare else '#3b528b')
        self.update_plots()

    def reset_params(self, event=None):
        self.s_V0.set_val(self.V0_ratio_init)
        self.s_a.set_val(self.a_ratio_init)
        self.s_n.set_val(self.n_select_init)
        self.s_xrange.set_val(self.x_range_init)
        self.show_inf_compare = True
        self.btn_comp.label.set_text('Pozo Infinito: ON')
        self.btn_comp.ax.set_facecolor('#21918c')
        self.update_plots()

    def update_plots(self, val=None):
        V0_ratio = self.s_V0.val
        a_ratio = self.s_a.val
        n_user = int(self.s_n.val)
        x_mult = self.s_xrange.val

        # Recalcular estados
        well_data = compute_finite_well_states(V0_ratio=V0_ratio, a_ratio=a_ratio)
        states = well_data['states']
        num_states = well_data['num_states']

        # Ajustar slider n si sobrepasa los estados ligados disponibles
        if num_states > 0 and n_user > num_states:
            n_select = num_states
        else:
            n_select = max(1, min(n_user, num_states if num_states > 0 else 1))

        # Seleccionar el estado actual
        current_state = states[n_select - 1] if num_states >= n_select else None

        a = well_data['a']
        V0 = well_data['V0']
        z0 = well_data['z0']
        E_ref = well_data['E_ref']

        x_max = x_mult * a
        x_grid = np.linspace(-x_max, x_max, 1600)

        # Etiqueta indicadora del modo activo
        mode_str = "[Finito + Infinito]" if self.show_inf_compare else "[Solo Pozo Finito]"

        # ---------------- 1. SUBPLOT 1: POTENCIAL Y NIVELES DE ENERGÍA ----------------
        self.ax_pot.clear()
        self.ax_pot.set_facecolor(self.bg_card)

        # Perfil de Potencial V(x)
        V_grid = np.where(np.abs(x_grid) <= a, 0.0, V0)
        self.ax_pot.plot(x_grid, V_grid, color='#b5367a', linewidth=2.5, label=f'Potencial Finito V₀ = {V0:.2f}')

        # Marcar interfaces x = -a y x = +a
        self.ax_pot.axvline(-a, color='#a0a0a0', linestyle=':', linewidth=1.2, alpha=0.6)
        self.ax_pot.axvline(a, color='#a0a0a0', linestyle=':', linewidth=1.2, alpha=0.6)

        # Niveles de Energía Discretos
        for st in states:
            n_st = st['n']
            E_st = st['E']
            is_curr = (n_st == n_select)

            col = '#fde725' if is_curr else '#21918c'
            lw = 2.2 if is_curr else 1.2
            lbl_fin = f'E_{n_st} (Finito) = {E_st:.2f}' if is_curr else None

            self.ax_pot.hlines(E_st, -a * 1.5, a * 1.5, colors=col, linewidth=lw, label=lbl_fin)

            if self.show_inf_compare:
                E_inf_st = st['E_inf']
                lbl_inf = f'E_{n_st}^∞ (Infinito) = {E_inf_st:.2f}' if is_curr else None
                self.ax_pot.hlines(E_inf_st, -a, a, colors='#40e0d0', linestyles='--', linewidth=1.4, label=lbl_inf)

        self.ax_pot.set_xlim([-x_max, x_max])
        self.ax_pot.set_ylim([-0.05 * V0, max(V0 * 1.25, 2.0)])
        self.ax_pot.set_xlabel('Posición x (u.a.)', color='white', fontsize=8)
        self.ax_pot.set_ylabel('Energía / Potencial', color='white', fontsize=8)
        self.ax_pot.set_title(f'Potencial V(x) y Niveles {mode_str}', color='#35b779', fontsize=10, fontweight='bold')
        self.ax_pot.grid(True, color='#1f2937', linestyle='--', alpha=0.5)
        self.ax_pot.legend(loc='upper right', facecolor=self.bg_card, edgecolor='#21918c', fontsize=7)

        # ---------------- 2. SUBPLOT 2: ECUACIÓN TRASCENDENTE Y ECUACIÓN DE ESTADOS ----------------
        self.ax_eq.clear()
        self.ax_eq.set_facecolor(self.bg_card)

        # Arco circular w = sqrt(z0^2 - z^2)
        z_circ = np.linspace(0, z0, 500)
        w_circ = np.sqrt(np.maximum(z0**2 - z_circ**2, 0.0))
        self.ax_eq.plot(z_circ, w_circ, color='#fde725', linewidth=2.5, label=r'Arco $w = \sqrt{z_0^2 - z^2}$')

        # Ramas pares w = z tan(z)
        max_k = int(np.ceil(z0 / np.pi)) + 1
        for k_idx in range(max_k):
            z_start = k_idx * np.pi + 1e-4
            z_end = min((k_idx + 0.5) * np.pi - 1e-4, z0)
            if z_start < z_end:
                z_branch = np.linspace(z_start, z_end, 300)
                w_branch = z_branch * np.tan(z_branch)
                lbl = r'Par: $w = z \tan z$' if k_idx == 0 else ""
                self.ax_eq.plot(z_branch, w_branch, color='#40e0d0', linestyle='--', linewidth=1.8, label=lbl)

        # Ramas impares w = -z cot(z)
        for k_idx in range(max_k):
            z_start = (k_idx + 0.5) * np.pi + 1e-4
            z_end = min((k_idx + 1) * np.pi - 1e-4, z0)
            if z_start < z_end:
                z_branch = np.linspace(z_start, z_end, 300)
                w_branch = -z_branch / np.tan(z_branch)
                lbl = r'Impar: $w = -z \cot z$' if k_idx == 0 else ""
                self.ax_eq.plot(z_branch, w_branch, color='#ff007f', linestyle=':', linewidth=1.8, label=lbl)

        # Intersecciones
        for st in states:
            n_st = st['n']
            z_st = st['z']
            w_st = st['w']
            is_curr = (n_st == n_select)

            if is_curr:
                self.ax_eq.plot(z_st, w_st, 'o', color='#fde725', markersize=9, markeredgecolor='white',
                                label=f'Estado n={n_st} (z={z_st:.2f})')
            else:
                self.ax_eq.plot(z_st, w_st, 'o', color='#35b779', markersize=5)

        self.ax_eq.set_xlim([0, z0 * 1.1])
        self.ax_eq.set_ylim([0, z0 * 1.1])
        self.ax_eq.set_xlabel(r'z = k a = $\frac{a}{\hbar}\sqrt{2mE}$', color='white', fontsize=8)
        self.ax_eq.set_ylabel(r'w = $\kappa a = \frac{a}{\hbar}\sqrt{2m(V_0-E)}$', color='white', fontsize=8)
        self.ax_eq.set_title(f'Ecuación Trascendente (z₀ = {z0:.2f})', color='#fde725', fontsize=10, fontweight='bold')
        self.ax_eq.grid(True, color='#1f2937', linestyle='--', alpha=0.5)
        self.ax_eq.legend(loc='upper right', facecolor=self.bg_card, edgecolor='#fde725', fontsize=7)

        # ---------------- 3. SUBPLOT 3: AUTOFUNCIÓN DE ONDA psi_n(x) ----------------
        self.ax_psi.clear()
        self.ax_psi.set_facecolor(self.bg_card)

        if current_state is not None:
            wf_data = compute_wavefunctions(current_state, a, x_grid)
            psi_fin = wf_data['psi_fin']
            psi_inf = wf_data['psi_inf']

            self.ax_psi.axhline(0, color='#a0a0a0', linestyle='-', linewidth=0.8, alpha=0.4)
            self.ax_psi.axvline(-a, color='#a0a0a0', linestyle=':', linewidth=1.2, alpha=0.6, label='x = -a')
            self.ax_psi.axvline(a, color='#a0a0a0', linestyle=':', linewidth=1.2, alpha=0.6, label='x = +a')

            self.ax_psi.plot(x_grid, psi_fin, color='#35b779', linewidth=2.2,
                             label=f'$\\psi_{n_select}(x)$ Pozo Finito ({current_state["parity"]})')

            if self.show_inf_compare:
                self.ax_psi.plot(x_grid, psi_inf, color='#40e0d0', linestyle='--', linewidth=1.5,
                                 label=f'$\\psi_{n_select}^\\infty(x)$ Pozo Infinito (V₀=∞)')

            self.ax_psi.set_xlim([-x_max, x_max])
            self.ax_psi.set_xlabel('Posición x (u.a.)', color='white', fontsize=8)
            self.ax_psi.set_ylabel('Amplitud de Onda', color='white', fontsize=8)
            self.ax_psi.set_title(f'Autofunción $\\psi_{n_select}(x)$ {mode_str} ({current_state["parity"]}, {current_state["nodes"]} nodos)',
                                  color='#35b779', fontsize=10, fontweight='bold')
            self.ax_psi.grid(True, color='#1f2937', linestyle='--', alpha=0.5)
            self.ax_psi.legend(loc='upper right', facecolor=self.bg_card, edgecolor='#35b779', fontsize=7)

        # ---------------- 4. SUBPLOT 4: DENSIDAD DE PROBABILIDAD |psi_n(x)|^2 ----------------
        self.ax_prob.clear()
        self.ax_prob.set_facecolor(self.bg_card)

        if current_state is not None:
            prob_fin = wf_data['prob_fin']
            prob_inf = wf_data['prob_inf']
            P_ext = wf_data['P_ext']
            P_int = wf_data['P_int']

            self.ax_prob.axvline(-a, color='#a0a0a0', linestyle=':', linewidth=1.2, alpha=0.6)
            self.ax_prob.axvline(a, color='#a0a0a0', linestyle=':', linewidth=1.2, alpha=0.6)

            self.ax_prob.plot(x_grid, prob_fin, color='#fde725', linewidth=2.2,
                             label=f'|$\\psi_{n_select}(x)$|² Pozo Finito (P_ext={P_ext*100:.1f}%)')

            # Destacar el área exterior (penetrabilidad en región prohibida)
            mask_ext = wf_data['mask_ext']
            self.ax_prob.fill_between(x_grid, 0, prob_fin, where=mask_ext, color='#b5367a', alpha=0.4,
                                      label=f'Penetración Finito: {P_ext*100:.2f}%')

            if self.show_inf_compare:
                self.ax_prob.plot(x_grid, prob_inf, color='#40e0d0', linestyle='--', linewidth=1.5,
                                 label=f'|$\\psi_{n_select}^\\infty(x)$|² Pozo Infinito (P_ext=0%)')

            self.ax_prob.set_xlim([-x_max, x_max])
            self.ax_prob.set_xlabel('Posición x (u.a.)', color='white', fontsize=8)
            self.ax_prob.set_ylabel('Densidad de Probabilidad', color='white', fontsize=8)
            self.ax_prob.set_title(f'Densidad de Probabilidad {mode_str} (P_int={P_int*100:.1f}%, P_ext={P_ext*100:.1f}%)',
                                  color='#fde725', fontsize=10, fontweight='bold')
            self.ax_prob.grid(True, color='#1f2937', linestyle='--', alpha=0.5)
            self.ax_prob.legend(loc='upper right', facecolor=self.bg_card, edgecolor='#fde725', fontsize=7)

        self.fig.suptitle("LABORATORIO PYTHON 14: POZO CUADRADO FINITO (CUANDO LAS PAREDES PERMITEN PENETRACIÓN)",
                          color='white', fontsize=12, fontweight='bold', y=0.98)
        self.fig.canvas.draw_idle()


def export_static_sample(output_path="/workspace/scratch/pozo_finito_sample.png"):
    """
    Genera una imagen de muestra estática para verificar ejecución en entrono headless.
    """
    well_data = compute_finite_well_states(10.0, 1.0)
    current_state = well_data['states'][0]
    x_grid = np.linspace(-4.0, 4.0, 1000)
    wf_data = compute_wavefunctions(current_state, 1.0, x_grid)

    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor='#07151a')
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    for ax in axes.flat:
        ax.set_facecolor('#0e242b')

    axes[0, 0].plot(x_grid, wf_data['psi_fin'], color='#35b779', label=r'$\psi_1(x)$')
    axes[0, 0].set_title("Autofunción Pozo Finito", color='white')
    axes[0, 0].legend()

    axes[0, 1].plot(x_grid, wf_data['prob_fin'], color='#fde725', label=r'$|\psi_1(x)|^2$')
    axes[0, 1].set_title("Densidad de Probabilidad", color='white')
    axes[0, 1].legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] Sample saved to {output_path}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        export_static_sample()
    else:
        app = FiniteSquareWellApp()
        plt.show()
