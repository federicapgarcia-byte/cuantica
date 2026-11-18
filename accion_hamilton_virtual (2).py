"""
================================================================================
LABORATORIO PYTHON: ACCIÓN DE HAMILTON-JACOBI Y POTENCIALES (CAPÍTULO 2)
================================================================================
Materia: Mecánica Cuántica
Entorno: Visual Studio Code / Python 3
Librerías: SymPy, NumPy, Matplotlib (Paleta 'viridis' de alto contraste), SciPy
================================================================================
"""

import sys
import os
import numpy as np
import sympy as sp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# Forzado de Backend interactivo para ejecuciones locales en VS Code / Tkinter
if '--test' not in sys.argv and 'pytest' not in sys.modules:
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass


def run_sympy_analysis():
    """
    Análisis algebraico con SymPy: Obtención simbólica de p(x) y W(x)
    para los cuatro potenciales del Capítulo 2 de QuantumM.
    """
    print("=" * 80)
    print(" 1. ANÁLISIS SIMBÓLICO CON SYMPY - CAPÍTULO 2: HAMILTON-JACOBI Y ACCIÓN W(x)")
    print("=" * 80)

    x, m, E, F, k, V0 = sp.symbols('x m E F k V0', real=True, positive=True)

    # 1. Partícula libre V(x) = 0
    p_libre = sp.sqrt(2 * m * E)
    W_libre = sp.integrate(p_libre, x)
    print("\n1. PARTÍCULA LIBRE (V(x) = 0):")
    print("   • Momento p(x) =")
    sp.pprint(p_libre)
    print("   • Acción W(x) = ∫ p dx =")
    sp.pprint(W_libre)

    # 2. Potencial lineal V(x) = a*x + b
    p_lineal = sp.sqrt(2 * m * (E - F * x))
    W_lineal = sp.integrate(p_lineal, x)
    print("\n2. POTENCIAL LINEAL (V(x) = a*x + b):")
    print("   • Punto de retorno clásico x_retorno = E / F")
    print("   • Momento p(x) =")
    sp.pprint(p_lineal)
    print("   • Acción W(x) = ∫ p dx =")
    sp.pprint(sp.simplify(W_lineal))

    # 3. Oscilador armónico V(x) = 1/2 * k * x^2
    p_osc = sp.sqrt(2 * m * (E - sp.Rational(1, 2) * k * x**2))
    print("\n3. OSCILADOR ARMÓNICO / POTENCIAL ELÁSTICO (V(x) = 1/2 * k * x^2):")
    print("   • Puntos de retorno clásicos x_retorno = ± √(2E / k)")
    print("   • Momento p(x) =")
    sp.pprint(p_osc)

    # 4. Relación fundamental del gradiente: dW/dx = p(x)
    print("\n• RELACIÓN FUNDAMENTAL DE HAMILTON-JACOBI (Gradiente de Acción):")
    print("  ∇W(x) = dW/dx = p(x) = √(2m[E - V(x)])")
    print("  Físicamente: El gradiente de la acción W da el campo de momentos clásicos.")
    print("=" * 80 + "\n")


def compute_potential_and_action(mode, E_val, x_grid, m_val=1.0, param_k=0.6, param_a=0.5, param_b=1.5, param_V0=1.5, barrier_w=1.4):
    """
    Calcula V(x), p(x) y W(x) numéricamente sobre la malla x_grid de forma exacta y pura.
    """
    if mode == 'Libre (V=0)':
        V_grid = np.zeros_like(x_grid)
        x_turn = []
    elif mode == 'Potencial Lineal':
        # Potencial lineal puro V(x) = a*x + b (sin recortar a 0 en el borde)
        V_grid = param_a * x_grid + param_b
        # Punto de retorno E = a*x + b  =>  x_turn = (E - b) / a
        x_turn = [(E_val - param_b) / param_a]
    elif mode == 'Oscilador Armónico':
        V_grid = 0.5 * param_k * (x_grid**2)
        x_turn_val = np.sqrt(max(2.0 * E_val / param_k, 0.0))
        x_turn = [-x_turn_val, x_turn_val]
    elif mode == 'Barrera de Potencial':
        V_grid = np.where(np.abs(x_grid) <= barrier_w / 2.0, param_V0, 0.0)
        if E_val < param_V0:
            x_turn = [-barrier_w / 2.0, barrier_w / 2.0]
        else:
            x_turn = []

    # Momento clásico p(x) = sqrt(2m(E - V(x)))
    diff = 2.0 * m_val * (E_val - V_grid)
    allowed_mask = diff >= 0.0

    p_grid = np.zeros_like(x_grid)
    p_grid[allowed_mask] = np.sqrt(diff[allowed_mask])

    # Acción reducida W(x) = integral(p(x') dx')
    dx = x_grid[1] - x_grid[0]
    W_grid = np.cumsum(p_grid) * dx
    # Normalizar W(x) para que empiece en 0 en el centro del dominio
    W_grid -= W_grid[len(W_grid) // 2]

    return V_grid, p_grid, W_grid, allowed_mask, x_turn


def launch_interactive_simulation():
    """Simulación interactiva completa con alta visibilidad y contraste."""
    run_sympy_analysis()

    # Parámetros iniciales
    x_part_init = 0.0
    E_init = 1.0

    x_grid = np.linspace(-4.0, 4.0, 800)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 9.5), facecolor='#0b0f19')

    # Configuración de Subplots
    ax_particle  = fig.add_subplot(4, 1, 1, facecolor='#111827')
    ax_potential = fig.add_subplot(4, 1, 2, facecolor='#111827')
    ax_momentum  = fig.add_subplot(4, 1, 3, facecolor='#111827')
    ax_action    = fig.add_subplot(4, 1, 4, facecolor='#111827')

    plt.subplots_adjust(left=0.08, right=0.72, top=0.94, bottom=0.08, hspace=0.48)

    # Controles Interactivos (Panel Derecho)
    ax_x_slider  = fig.add_axes([0.76, 0.82, 0.18, 0.025], facecolor='#1f2937')
    ax_E_slider  = fig.add_axes([0.76, 0.74, 0.18, 0.025], facecolor='#1f2937')
    ax_radio     = fig.add_axes([0.76, 0.48, 0.18, 0.20], facecolor='#111827')
    ax_info      = fig.add_axes([0.76, 0.08, 0.18, 0.34], facecolor='#111827')
    ax_info.axis('off')

    s_x = Slider(ax_x_slider, 'Posición x', -3.8, 3.8, valinit=x_part_init, color='#38b6ff')
    s_E = Slider(ax_E_slider, 'Energía E', 0.1, 3.0, valinit=E_init, color='#fde725')

    radio_mode = RadioButtons(ax_radio, ('Libre (V=0)', 'Potencial Lineal', 'Oscilador Armónico', 'Barrera de Potencial'),
                             active=2, activecolor='#00f5d4')

    for txt in radio_mode.labels:
        txt.set_color('white')
        txt.set_fontsize(9)

    s_x.label.set_color('white')
    s_x.valtext.set_color('white')
    s_E.label.set_color('white')
    s_E.valtext.set_color('white')

    fig._widgets = [s_x, s_E, radio_mode]

    def update_plots(val=None):
        x_part = s_x.val
        E_val = s_E.val
        mode = radio_mode.value_selected

        V_grid, p_grid, W_grid, allowed_mask, x_turn = compute_potential_and_action(mode, E_val, x_grid)

        # Interpolación de valores en la posición actual de la partícula x_part
        idx_part = np.argmin(np.abs(x_grid - x_part))
        V_part = V_grid[idx_part]
        p_part = p_grid[idx_part]
        W_part = W_grid[idx_part]
        is_allowed = allowed_mask[idx_part]

        # ---------------- 1. Esquema 1D de la Partícula ----------------
        ax_particle.clear()
        ax_particle.set_facecolor('#111827')
        ax_particle.axhline(0, color='#6b7280', linestyle='--', linewidth=1.2)

        ax_particle.set_xlim([-4.0, 4.0])
        ax_particle.set_ylim([-1.2, 1.2])
        ax_particle.set_ylabel('Eje X', color='white', fontsize=9)
        ax_particle.set_title('Desplazamiento 1D de la Partícula sobre el Eje X', color='#00f5d4', fontsize=11, fontweight='bold')

        # Dibujar partícula
        color_part = '#fde725' if is_allowed else '#ef4444'
        ax_particle.plot(x_part, 0, 'o', color=color_part, markersize=14, zorder=5, label=f'Partícula en x = {x_part:.2f}')

        # Vector momento p(x)
        if is_allowed and p_part > 0.05:
            ax_particle.annotate('', xy=(x_part + p_part * 0.4, 0), xytext=(x_part, 0),
                                 arrowprops=dict(arrowstyle="->", color='#38b6ff', lw=3.0))
            ax_particle.text(x_part + p_part * 0.2, 0.45, f'p = {p_part:.2f}', color='#38b6ff', fontsize=9, ha='center', fontweight='bold')

        # Marcadores de Puntos de Retorno Clásicos (E = V)
        for xt in x_turn:
            if -4.0 <= xt <= 4.0:
                ax_particle.axvline(xt, color='#f59e0b', linestyle=':', linewidth=1.8)
                ax_particle.plot(xt, 0, 'x', color='#f59e0b', markersize=10, markeredgewidth=2.5)

        ax_particle.legend(loc='upper right', facecolor='#111827', edgecolor='#00f5d4', fontsize=8)
        ax_particle.grid(True, color='#374151', linestyle='--', alpha=0.5)

        # ---------------- 2. Perfil del Potencial V(x) y Nivel de Energía E ----------------
        ax_potential.clear()
        ax_potential.set_facecolor('#111827')

        # Sombra de región permitida vs prohibida
        y_max_pot = max(np.max(V_grid) * 1.1, E_val * 1.3, 3.5)
        ax_potential.fill_between(x_grid, 0, y_max_pot, where=allowed_mask, color='#21918c', alpha=0.22, label='Región Permitida (E ≥ V)')
        ax_potential.fill_between(x_grid, 0, y_max_pot, where=~allowed_mask, color='#ef4444', alpha=0.20, label='Región Prohibida (E < V)')

        # Curva de V(x) y línea de Energía E
        ax_potential.plot(x_grid, V_grid, color='#35b779', linewidth=2.8, label='Potencial V(x)')
        ax_potential.axhline(E_val, color='#fde725', linewidth=2.2, linestyle='--', label=f'Energía Total E = {E_val:.2f}')

        # Marcador de la partícula en la curva de V(x)
        ax_potential.plot(x_part, V_part, 'o', color=color_part, markersize=10, zorder=5)
        ax_potential.vlines(x_part, 0, V_part, colors=color_part, linestyles=':', linewidth=1.4)

        # Puntos de Retorno (E = V)
        for xt in x_turn:
            if -4.0 <= xt <= 4.0:
                idx_t = np.argmin(np.abs(x_grid - xt))
                ax_potential.plot(xt, V_grid[idx_t], 'o', color='#f59e0b', markersize=8)
                ax_potential.axvline(xt, color='#f59e0b', linestyle=':', linewidth=1.4)
                ax_potential.text(xt, E_val + y_max_pot * 0.05, f'E=V\n({xt:.2f})', color='#f59e0b', fontsize=8, ha='center', fontweight='bold')

        ax_potential.set_xlim([-4.0, 4.0])
        ax_potential.set_ylim([-0.2, y_max_pot])
        ax_potential.set_ylabel('V(x), E', color='white', fontsize=9)
        ax_potential.set_title(f'Perfil del Potencial V(x) [{mode}] y Puntos de Retorno (E = V)', color='#35b779', fontsize=10, fontweight='bold')
        ax_potential.legend(loc='upper right', facecolor='#111827', edgecolor='#35b779', fontsize=8)
        ax_potential.grid(True, color='#374151', linestyle='--', alpha=0.5)

        # ---------------- 3. Campo de Momento Clásico p(x) ----------------
        ax_momentum.clear()
        ax_momentum.set_facecolor('#111827')

        # Usar color Celeste brillante (#38b6ff) para máxima visibilidad
        ax_momentum.plot(x_grid, p_grid, color='#38b6ff', linewidth=2.8, label=r'Momento Clásico $p(x) = \sqrt{2m(E - V(x))}$')
        ax_momentum.plot(x_part, p_part, 'o', color=color_part, markersize=9, zorder=5)
        ax_momentum.vlines(x_part, 0, p_part, colors=color_part, linestyles=':', linewidth=1.4)

        p_max_vis = max(np.max(p_grid) * 1.15, 1.0)
        ax_momentum.set_xlim([-4.0, 4.0])
        ax_momentum.set_ylim([-0.1, p_max_vis])
        ax_momentum.set_ylabel('p(x)', color='white', fontsize=9)
        ax_momentum.set_title(r'Campo de Momento Clásico $p(x) = \nabla W(x)$', color='#38b6ff', fontsize=10, fontweight='bold')
        ax_momentum.legend(loc='upper right', facecolor='#111827', edgecolor='#38b6ff', fontsize=8)
        ax_momentum.grid(True, color='#374151', linestyle='--', alpha=0.5)

        # ---------------- 4. Función Característica de Hamilton W(x) ----------------
        ax_action.clear()
        ax_action.set_facecolor('#111827')

        # Usar color Lima / Ciénaga brillante (#00f5d4) para alta visibilidad
        ax_action.plot(x_grid, W_grid, color='#00f5d4', linewidth=2.8, label=r'Acción Reducida $W(x) = \int p(x) dx$')
        ax_action.plot(x_part, W_part, 'o', color=color_part, markersize=9, zorder=5)
        ax_action.vlines(x_part, np.min(W_grid) - 0.2, W_part, colors=color_part, linestyles=':', linewidth=1.4)
        ax_action.axhline(0, color='#6b7280', linestyle=':', linewidth=1)

        w_min, w_max = np.min(W_grid), np.max(W_grid)
        w_margin = max((w_max - w_min) * 0.1, 0.5)
        ax_action.set_xlim([-4.0, 4.0])
        ax_action.set_ylim([w_min - w_margin, w_max + w_margin])
        ax_action.set_xlabel('Posición x', color='white', fontsize=9)
        ax_action.set_ylabel('W(x)', color='white', fontsize=9)
        ax_action.set_title(r'Función Característica de Hamilton $W(x)$ [Gradiente $\nabla W = p(x)$]', color='#00f5d4', fontsize=10, fontweight='bold')
        ax_action.legend(loc='upper right', facecolor='#111827', edgecolor='#00f5d4', fontsize=8)
        ax_action.grid(True, color='#374151', linestyle='--', alpha=0.5)

        # ---------------- 5. Panel de Información Cuántica y Dinámica ----------------
        ax_info.clear()
        ax_info.axis('off')

        info_text = (
            "=== MÉTRICAS HAMILTON-JACOBI ===\n\n"
            f" • Modo          : {mode}\n"
            f" • Posición x    : {x_part:.3f}\n"
            f" • Energía E     : {E_val:.3f}\n"
            f" • Potencial V(x): {V_part:.3f}\n\n"
            "--- Estado de Movimiento ---\n"
            f" • Condición     : {'PERMITIDO (E ≥ V)' if is_allowed else 'PROHIBIDO (E < V)'}\n"
            f" • Momento p(x)  : {p_part:.3f}\n"
            f" • Acción W(x)   : {W_part:.3f}\n\n"
            "--- Relación Gradiente ---\n"
            " • ∇W = dW/dx = p(x)\n"
            " • Eikonal vs HJ:\n"
            "   |∇Φ| = k  <-->  |∇W| = p\n"
        )

        ax_info.text(0.05, 0.95, info_text, color='#e2e8f0', fontsize=8.8, fontfamily='monospace',
                     verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.7', facecolor='#1f2937', edgecolor='#00f5d4', alpha=0.95))

        if '--test' not in sys.argv:
            fig.canvas.draw_idle()

    s_x.on_changed(update_plots)
    s_E.on_changed(update_plots)
    radio_mode.on_clicked(update_plots)

    update_plots()
    if '--test' not in sys.argv:
        plt.show()


if __name__ == '__main__':
    launch_interactive_simulation()
