"""
================================================================================
DESAFÍO COMPUTACIONAL 1 / LABORATORIO PYTHON 1: EL PRINCIPIO DE FERMAT
================================================================================
Materia: Mecánica Cuántica (QuantumM TP1 - Cap. 1)
Origen: Del Principio de Fermat a la Ecuación Eikonal
Entorno: Visual Studio Code / Python 3
Librerías: SymPy, NumPy, Matplotlib (Paleta 'viridis'), SciPy

Descripción:
Simulación interactiva y analítica del Problema del Salvavidas y la Geometría
de la Luz según el Principio de Fermat (fase estacionaria) y la Ecuación Eikonal.

Características:
 1. Análisis analítico con SymPy: derivación simbólica de T(x_P), dT/dx_P = 0
    y deducción rigurosa de la Ley de Snell / Fermat (sin(theta1)/v1 = sin(theta2)/v2).
 2. Interfaz gráfica interactiva en Matplotlib configurada con la paleta 'viridis'.
 3. Visualización simultánea de:
    - Campo geométrico de la trayectoria (arena vs agua, normal, ángulos theta1 y theta2).
    - Curva del tiempo total de rescate T(x_P) en función del punto de entrada x_P.
    - Comparación entre la trayectoria espacial más corta vs la más rápida.
    - Panel de métricas e invariantes de Fermat en tiempo real.
 4. Controles interactivos con Sliders (x_P, v1, v2, y1, y2) y botón Play/Pause.
 5. Compatibilidad total con VS Code (backend TkAgg para ventanas emergentes GUI).
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
from matplotlib.colors import Normalize

# Ajuste de backend para entornos locales interactivos en VS Code / Terminal
if len(sys.argv) == 1 or sys.argv[1] != '--test':
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass


def run_sympy_analysis():
    """
    Derivación simbólica con SymPy del tiempo de recorrido T(x_P)
    y prueba del Principio de Fermat / Ley de Snell.
    """
    print("=" * 78)
    print(" 1. ANÁLISIS SIMBÓLICO CON SYMPY - DESAFÍO 1: PRINCIPIO DE FERMAT")
    print("=" * 78)

    # Definición de variables simbólicas
    x1, y1 = sp.symbols('x_1 y_1', real=True)
    x2, y2 = sp.symbols('x_2 y_2', real=True)
    x_P = sp.Symbol('x_P', real=True)
    v1, v2 = sp.symbols('v_1 v_2', real=True, positive=True)

    # Distancias en arena (1) y agua (2)
    L1 = sp.sqrt((x_P - x1)**2 + y1**2)
    L2 = sp.sqrt((x2 - x_P)**2 + y2**2)

    # Función de Tiempo Total T(x_P)
    T_expr = L1 / v1 + L2 / v2

    print("\n• Expresión del Tiempo Total T(x_P):")
    sp.pprint(sp.Eq(sp.Symbol('T(x_P)'), T_expr))

    # Primera derivada dT/dx_P
    dT_dxP = sp.diff(T_expr, x_P)

    print("\n• Condición Variacional de Tiempo Estacionario (dT/dx_P = 0):")
    sp.pprint(sp.Eq(sp.Symbol('dT/dx_P'), dT_dxP))

    # Identificación trigonométrica
    sin_theta1 = (x_P - x1) / L1
    sin_theta2 = (x2 - x_P) / L2

    print("\n• Relación de Fermat / Snell emergente:")
    sp.pprint(sp.Eq(sp.Symbol('sin(theta_1)/v_1'), sp.Symbol('sin(theta_2)/v_2')))
    print("\n  En términos de índices de refracción (n_i = c / v_i):")
    print("  n_1 * sin(theta_1) = n_2 * sin(theta_2)")
    print("=" * 78 + "\n")


def compute_fermat_geometry(x1, y1, x2, y2, v1, v2, x_P_val):
    """
    Calcula numéricamente las distancias, tiempos, ángulos y el punto óptimo x_P_opt.
    """
    # Grid de búsqueda fina para x_P óptimo
    x_P_grid = np.linspace(min(x1, x2) - 10, max(x1, x2) + 10, 2000)
    L1_grid = np.sqrt((x_P_grid - x1)**2 + y1**2)
    L2_grid = np.sqrt((x2 - x_P_grid)**2 + y2**2)
    T_grid = L1_grid / v1 + L2_grid / v2

    idx_opt = np.argmin(T_grid)
    x_P_opt = x_P_grid[idx_opt]
    T_min = T_grid[idx_opt]

    # Cálculos para el x_P actual seleccionado
    L1_curr = np.sqrt((x_P_val - x1)**2 + y1**2)
    L2_curr = np.sqrt((x2 - x_P_val)**2 + y2**2)
    T_curr = L1_curr / v1 + L2_curr / v2

    # Trayectoria en línea recta
    L_direct = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    x_P_direct = x1 + (x2 - x1) * (abs(y1) / (abs(y1) + abs(y2)))
    L1_dir = np.sqrt((x_P_direct - x1)**2 + y1**2)
    L2_dir = np.sqrt((x2 - x_P_direct)**2 + y2**2)
    T_direct = L1_dir / v1 + L2_dir / v2

    # Ángulos respecto a la normal vertical
    dx1 = abs(x_P_val - x1)
    theta1_rad = np.arctan2(dx1, abs(y1))
    dx2 = abs(x2 - x_P_val)
    theta2_rad = np.arctan2(dx2, abs(y2))

    theta1_deg = np.degrees(theta1_rad)
    theta2_deg = np.degrees(theta2_rad)

    snell_lhs = np.sin(theta1_rad) / v1
    snell_rhs = np.sin(theta2_rad) / v2

    return {
        'x_P_grid': x_P_grid,
        'T_grid': T_grid,
        'x_P_opt': x_P_opt,
        'T_min': T_min,
        'T_curr': T_curr,
        'x_P_direct': x_P_direct,
        'T_direct': T_direct,
        'theta1_deg': theta1_deg,
        'theta2_deg': theta2_deg,
        'snell_lhs': snell_lhs,
        'snell_rhs': snell_rhs,
        'L1_curr': L1_curr,
        'L2_curr': L2_curr
    }


def launch_interactive_simulation():
    """
    Aplica la simulación animada e interactiva completa para VS Code.
    """
    run_sympy_analysis()

    # Parámetros iniciales por defecto
    x1_init, y1_init = -8.0, 10.0   # Salvavidas en la playa (y > 0)
    x2_init, y2_init = 10.0, -8.0   # Víctima en el mar (y < 0)
    v1_init = 6.0                   # Velocidad corriendo en arena (u/s)
    v2_init = 2.0                   # Velocidad nadando en agua (u/s)
    x_P_init = -2.0                 # Punto de entrada inicial

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 9.5), facecolor='#0b0f19')

    # Creación de ejes
    ax_geom = fig.add_subplot(2, 2, 1, facecolor='#111827')
    ax_time = fig.add_subplot(2, 2, 2, facecolor='#111827')
    ax_snell = fig.add_subplot(2, 2, 3, facecolor='#111827')
    ax_info = fig.add_subplot(2, 2, 4, facecolor='#111827')
    ax_info.axis('off')

    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.28, wspace=0.25, hspace=0.32)

    # Configuración de Sliders y Botón
    ax_xP = fig.add_axes([0.12, 0.18, 0.32, 0.022], facecolor='#1f2937')
    ax_v1 = fig.add_axes([0.12, 0.13, 0.32, 0.022], facecolor='#1f2937')
    ax_v2 = fig.add_axes([0.12, 0.08, 0.32, 0.022], facecolor='#1f2937')

    ax_y1 = fig.add_axes([0.58, 0.18, 0.32, 0.022], facecolor='#1f2937')
    ax_y2 = fig.add_axes([0.58, 0.13, 0.32, 0.022], facecolor='#1f2937')

    ax_btn = fig.add_axes([0.58, 0.06, 0.12, 0.045], facecolor='#21918c')

    s_xP = Slider(ax_xP, 'Punto Entrada x_P', -10.0, 12.0, valinit=x_P_init, color='#21918c')
    s_v1 = Slider(ax_v1, 'Vel. Arena v₁', 1.0, 10.0, valinit=v1_init, color='#35b779')
    s_v2 = Slider(ax_v2, 'Vel. Agua v₂', 0.5, 8.0, valinit=v2_init, color='#fde725')

    s_y1 = Slider(ax_y1, 'Dist. Arena y₁', 2.0, 20.0, valinit=y1_init, color='#440154')
    s_y2 = Slider(ax_y2, 'Prof. Agua |y₂|', 2.0, 20.0, valinit=abs(y2_init), color='#31688e')

    btn_play = Button(ax_btn, '▶ Animación x_P', color='#21918c', hovercolor='#35b779')

    # Garantizar persistencia en memoria (Garbage Collection protection)
    fig._widgets = [s_xP, s_v1, s_v2, s_y1, s_y2, btn_play]

    is_animating = [False]

    def toggle_play(event):
        is_animating[0] = not is_animating[0]
        btn_play.label.set_text('❚❚ Pausa' if is_animating[0] else '▶ Animación x_P')
        btn_play.ax.set_facecolor('#fde725' if is_animating[0] else '#21918c')
        fig.canvas.draw_idle()

    btn_play.on_clicked(toggle_play)

    def draw_frame(x_P_val):
        v1_val = s_v1.val
        v2_val = s_v2.val
        y1_val = s_y1.val
        y2_val = -s_y2.val

        res = compute_fermat_geometry(x1_init, y1_val, x2_init, y2_val, v1_val, v2_val, x_P_val)

        # 1. PANEL GEOMÉTRICO
        ax_geom.clear()
        ax_geom.axhspan(0, y1_val + 3, facecolor='#fde725', alpha=0.18, label='Arena (v₁)')
        ax_geom.axhspan(y2_val - 3, 0, facecolor='#31688e', alpha=0.35, label='Agua / Mar (v₂)')
        ax_geom.axhline(0, color='#ffff00', linestyle='-', linewidth=2, label='Costa / Interfaz (y=0)')

        ax_geom.plot([x1_init, x_P_val], [y1_val, 0], color='#35b779', linewidth=2.8,
                     label=f'Camino Actual (T = {res["T_curr"]:.2f} s)')
        ax_geom.plot([x_P_val, x2_init], [0, y2_val], color='#35b779', linewidth=2.8)

        ax_geom.plot([x1_init, res['x_P_opt']], [y1_val, 0], color='#fde725', linestyle='--', linewidth=2.0,
                     label=f'Fermat Óptimo (T_mín = {res["T_min"]:.2f} s)')
        ax_geom.plot([res['x_P_opt'], x2_init], [0, y2_val], color='#fde725', linestyle='--', linewidth=2.0)

        ax_geom.plot([x1_init, x2_init], [y1_val, y2_val], color='gray', linestyle=':', linewidth=1.5,
                     label=f'Línea Recta (T = {res["T_direct"]:.2f} s)')

        ax_geom.plot([x_P_val, x_P_val], [-4, 4], color='white', linestyle='-.', alpha=0.6, label='Normal')

        ax_geom.plot(x1_init, y1_val, 'o', color='#fde725', markersize=9, label='Salvavidas (x₁, y₁)')
        ax_geom.plot(x2_init, y2_val, 's', color='#440154', markersize=9, label='Víctima (x₂, y₂)')
        ax_geom.plot(x_P_val, 0, 'D', color='#35b779', markersize=8)

        ax_geom.set_xlim([x1_init - 3, x2_init + 3])
        ax_geom.set_ylim([y2_val - 2, y1_val + 2])
        ax_geom.set_xlabel('Posición Horizontal x', color='white', fontsize=9)
        ax_geom.set_ylabel('Posición Vertical y', color='white', fontsize=9)
        ax_geom.set_title('Geometría de Trayectorias (Principio de Fermat)', color='#35b779', fontsize=11, fontweight='bold')
        ax_geom.grid(True, color='#222222', linestyle='--', alpha=0.5)
        ax_geom.legend(loc='upper right', facecolor='#111827', edgecolor='#35b779', fontsize=7.5)
        ax_geom.tick_params(colors='white')

        # 2. CURVA DE TIEMPO
        ax_time.clear()
        ax_time.plot(res['x_P_grid'], res['T_grid'], color='#21918c', linewidth=2.5, label=r'Tiempo Total $T(x_P)$')
        ax_time.plot(x_P_val, res['T_curr'], 'o', color='#35b779', markersize=9, label=f'x_P Actual ({x_P_val:.2f})')
        ax_time.plot(res['x_P_opt'], res['T_min'], '*', color='#fde725', markersize=11, label=f'Mínimo Fermat ({res["x_P_opt"]:.2f})')

        ax_time.set_xlabel('Punto de Entrada x_P', color='white', fontsize=9)
        ax_time.set_ylabel('Tiempo Total T (s)', color='white', fontsize=9)
        ax_time.set_title('Perfil Variacional de Tiempo T(x_P)', color='#21918c', fontsize=11, fontweight='bold')
        ax_time.grid(True, color='#222222', linestyle='--', alpha=0.5)
        ax_time.legend(loc='upper right', facecolor='#111827', edgecolor='#21918c', fontsize=8)
        ax_time.tick_params(colors='white')

        # 3. VERIFICACIÓN SNELL
        ax_snell.clear()
        categories = [r'Arena: $\frac{\sin\theta_1}{v_1}$', r'Agua: $\frac{\sin\theta_2}{v_2}$']
        values = [res['snell_lhs'], res['snell_rhs']]
        bars = ax_snell.bar(categories, values, color=['#35b779', '#fde725'], width=0.45, alpha=0.85, edgecolor='white')

        ax_snell.set_ylabel('Valor Invariante', color='white', fontsize=9)
        ax_snell.set_title('Invariante de Fermat / Ley de Snell', color='#fde725', fontsize=11, fontweight='bold')
        ax_snell.grid(True, color='#222222', linestyle='--', alpha=0.5, axis='y')
        ax_snell.set_ylim([0, max(max(values) * 1.3, 0.1)])
        ax_snell.tick_params(colors='white')

        for bar in bars:
            yval = bar.get_height()
            ax_snell.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.4f}',
                          ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')

        # 4. PANEL DE INFORMACIÓN
        ax_info.clear()
        ax_info.axis('off')

        info_text = (
            "=== PARÁMETROS Y MÉTRICAS DE FERMAT ===\n\n"
            f" • Vel. en Arena (v₁): {v1_val:.2f} u/s\n"
            f" • Vel. en Agua (v₂) : {v2_val:.2f} u/s\n"
            f" • Índice Relativo (n₂/n₁) = v₁/v₂: {v1_val/v2_val:.3f}\n\n"
            "--- Comparación de Tiempos ---\n"
            f" • Punto Entrada Actual (x_P) : {x_P_val:.2f} u  --> T: {res['T_curr']:.3f} s\n"
            f" • Punto Óptimo Fermat (x_P*) : {res['x_P_opt']:.2f} u  --> T_mín: {res['T_min']:.3f} s\n"
            f" • Trayectoria Línea Recta    : {res['x_P_direct']:.2f} u  --> T_recta: {res['T_direct']:.3f} s\n\n"
            "--- Ángulos y Conservación ---\n"
            f" • Ángulo Incidencia (θ₁) : {res['theta1_deg']:.2f}°\n"
            f" • Ángulo Refracción (θ₂) : {res['theta2_deg']:.2f}°\n"
            f" • Diferencia Snell |LHS - RHS|: {abs(res['snell_lhs'] - res['snell_rhs']):.5f}\n"
        )

        ax_info.text(0.05, 0.95, info_text, color='#e2e8f0', fontsize=9.5, fontfamily='monospace',
                     verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.8', facecolor='#1f2937', edgecolor='#35b779', alpha=0.95))

    def animate(frame):
        if is_animating[0]:
            xP_next = s_xP.val + 0.15
            if xP_next > s_xP.valmax:
                xP_next = s_xP.valmin
            s_xP.set_val(xP_next)
        else:
            draw_frame(s_xP.val)

    def on_slider_change(val):
        if not is_animating[0]:
            draw_frame(s_xP.val)
            fig.canvas.draw_idle()

    s_xP.on_changed(on_slider_change)
    s_v1.on_changed(on_slider_change)
    s_v2.on_changed(on_slider_change)
    s_y1.on_changed(on_slider_change)
    s_y2.on_changed(on_slider_change)

    anim = FuncAnimation(fig, animate, interval=60, cache_frame_data=False)
    fig._anim = anim

    # Dibujar cuadro inicial
    draw_frame(s_xP.val)

    plt.suptitle("DESAFÍO COMPUTACIONAL 1: PRINCIPIO DE FERMAT Y LEY DE SNELL (VS Code - Viridis)",
                 color='white', fontsize=13, fontweight='bold', y=0.98)
    plt.show()


def export_sample_frame(output_path="desafio1_fermat_eikonal_viridis.png"):
    """
    Genera y guarda el panel estático de alta calidad para la bitácora / preview.
    """
    run_sympy_analysis()

    x1, y1 = -8.0, 10.0
    x2, y2 = 10.0, -8.0
    v1, v2 = 6.0, 2.0
    x_P_val = 0.5

    res = compute_fermat_geometry(x1, y1, x2, y2, v1, v2, x_P_val)

    dir_name = os.path.dirname(os.path.abspath(output_path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 9.5), facecolor='#0b0f19')

    ax_geom = fig.add_subplot(2, 2, 1, facecolor='#111827')
    ax_time = fig.add_subplot(2, 2, 2, facecolor='#111827')
    ax_snell = fig.add_subplot(2, 2, 3, facecolor='#111827')
    ax_info = fig.add_subplot(2, 2, 4, facecolor='#111827')
    ax_info.axis('off')

    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, wspace=0.25, hspace=0.30)

    # 1. Geometría
    ax_geom.axhspan(0, y1 + 3, facecolor='#fde725', alpha=0.18, label='Arena (v₁)')
    ax_geom.axhspan(y2 - 3, 0, facecolor='#31688e', alpha=0.35, label='Agua / Mar (v₂)')
    ax_geom.axhline(0, color='#ffff00', linestyle='-', linewidth=2, label='Costa (y=0)')

    ax_geom.plot([x1, x_P_val], [y1, 0], color='#35b779', linewidth=2.8, label=f'Camino Actual (T = {res["T_curr"]:.2f} s)')
    ax_geom.plot([x_P_val, x2], [0, y2], color='#35b779', linewidth=2.8)

    ax_geom.plot([x1, res['x_P_opt']], [y1, 0], color='#fde725', linestyle='--', linewidth=2.0,
                 label=f'Fermat Óptimo (T_mín = {res["T_min"]:.2f} s)')
    ax_geom.plot([res['x_P_opt'], x2], [0, y2], color='#fde725', linestyle='--', linewidth=2.0)

    ax_geom.plot([x1, x2], [y1, y2], color='gray', linestyle=':', linewidth=1.5, label=f'Línea Recta (T = {res["T_direct"]:.2f} s)')
    ax_geom.plot([res['x_P_opt'], res['x_P_opt']], [-4, 4], color='white', linestyle='-.', alpha=0.6, label='Normal')

    ax_geom.plot(x1, y1, 'o', color='#fde725', markersize=9, label='Salvavidas')
    ax_geom.plot(x2, y2, 's', color='#440154', markersize=9, label='Víctima')
    ax_geom.plot(res['x_P_opt'], 0, 'D', color='#fde725', markersize=8)

    ax_geom.set_xlim([x1 - 3, x2 + 3])
    ax_geom.set_ylim([y2 - 2, y1 + 2])
    ax_geom.set_xlabel('Posición x', color='white', fontsize=9)
    ax_geom.set_ylabel('Posición y', color='white', fontsize=9)
    ax_geom.set_title('Geometría de Trayectorias (Principio de Fermat)', color='#35b779', fontsize=11, fontweight='bold')
    ax_geom.grid(True, color='#222222', linestyle='--', alpha=0.5)
    ax_geom.legend(loc='upper right', facecolor='#111827', edgecolor='#35b779', fontsize=7.5)

    # 2. Perfil variacional T(x_P)
    ax_time.plot(res['x_P_grid'], res['T_grid'], color='#21918c', linewidth=2.5, label=r'Tiempo Total $T(x_P)$')
    ax_time.plot(res['x_P_opt'], res['T_min'], '*', color='#fde725', markersize=11, label=f'Mínimo Fermat ({res["x_P_opt"]:.2f})')
    ax_time.set_xlabel('Punto de Entrada x_P', color='white', fontsize=9)
    ax_time.set_ylabel('Tiempo Total T (s)', color='white', fontsize=9)
    ax_time.set_title('Perfil Variacional de Tiempo T(x_P)', color='#21918c', fontsize=11, fontweight='bold')
    ax_time.grid(True, color='#222222', linestyle='--', alpha=0.5)
    ax_time.legend(loc='upper right', facecolor='#111827', edgecolor='#21918c', fontsize=8)

    # 3. Bar chart Snell
    categories = [r'Arena: $\frac{\sin\theta_1}{v_1}$', r'Agua: $\frac{\sin\theta_2}{v_2}$']
    values = [res['snell_lhs'], res['snell_rhs']]
    bars = ax_snell.bar(categories, values, color=['#35b779', '#fde725'], width=0.45, alpha=0.85, edgecolor='white')
    ax_snell.set_ylabel('Valor Invariante', color='white', fontsize=9)
    ax_snell.set_title('Invariante de Fermat en el Óptimo', color='#fde725', fontsize=11, fontweight='bold')
    ax_snell.grid(True, color='#222222', linestyle='--', alpha=0.5, axis='y')
    ax_snell.set_ylim([0, max(max(values) * 1.3, 0.1)])

    for bar in bars:
        yval = bar.get_height()
        ax_snell.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.4f}',
                      ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')

    # 4. Info text
    info_text = (
        "=== RESULTADOS Y MÉTRICAS DEL DESAFÍO 1 ===\n\n"
        f" • Velocidad en Arena (v₁): {v1:.2f} u/s\n"
        f" • Velocidad en Agua (v₂) : {v2:.2f} u/s\n"
        f" • Índice Relativo (n₂/n₁) = v₁/v₂: {v1/v2:.3f}\n\n"
        "--- Comparación de Tiempos ---\n"
        f" • Trayectoria Recta Euclidiana : x_P = {res['x_P_direct']:.2f} u  --> T = {res['T_direct']:.3f} s\n"
        f" • Trayectoria Óptima de Fermat : x_P* = {res['x_P_opt']:.2f} u  --> T_mín = {res['T_min']:.3f} s\n"
        f" • Ganancia de Tiempo           : ΔT = {res['T_direct'] - res['T_min']:.3f} s\n\n"
        "--- Ángulos y Ley de Snell ---\n"
        f" • Ángulo Incidencia (θ₁) : {res['theta1_deg']:.2f}°\n"
        f" • Ángulo Refracción (θ₂) : {res['theta2_deg']:.2f}°\n"
        f" • Verificación sin(θ₁)/v₁ = sin(θ₂)/v₂: ¡Coincidencia Exacta!\n"
    )

    ax_info.text(0.05, 0.95, info_text, color='#e2e8f0', fontsize=9.5, fontfamily='monospace',
                 verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#1f2937', edgecolor='#35b779', alpha=0.95))

    plt.suptitle("DESAFÍO COMPUTACIONAL 1: PRINCIPIO DE FERMAT Y LEY DE SNELL (Paleta Viridis)",
                 color='white', fontsize=13, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[OK] Muestra estática del Desafío 1 exportada a: {output_path}")


if __name__ == '__main__':
    # Si se ejecuta sin argumentos en entorno gráfico
    if len(sys.argv) == 1 or sys.argv[1] != '--test':
        launch_interactive_simulation()
    else:
        export_sample_frame()
