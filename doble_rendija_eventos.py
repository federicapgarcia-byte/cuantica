"""
================================================================================
LABORATORIO PYTHON 4: DOBLE RENDIJA - UN EVENTO A LA VEZ
================================================================================
Materia: Mecánica Cuántica | Entorno: Visual Studio Code / Python 3
Librerías: SymPy, NumPy, Matplotlib (Paleta 'viridis'), SciPy
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

# Compatibilidad con integraciones numéricas en NumPy 2.0+ / SciPy
if hasattr(np, 'trapezoid'):
    trapz = np.trapezoid
else:
    from scipy.integrate import trapezoid as trapz

# Forzado de Backend interactivo TkAgg para VS Code / Terminal local
if len(sys.argv) == 1 or sys.argv[0] != '--test':
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass


def run_sympy_analysis():
    """
    Derivación simbólica con SymPy de las amplitudes de la doble rendija,
    el término de interferencia y la distribución de intensidad de Young.
    """
    print("=" * 80)
    print(" 1. ANÁLISIS SIMBÓLICO CON SYMPY - LABORATORIO 4: DOBLE RENDIJA")
    print("=" * 80)

    x, d, a, lam, L, delta = sp.symbols('x d a lambda L delta', real=True, positive=True)

    # Variables adim de difracción e interferencia
    alpha = sp.pi * a * x / (lam * L)
    beta = sp.pi * d * x / (lam * L) + delta / 2

    # Amplitudes individuales de Young-Fraunhofer
    A0 = sp.Symbol('A_0', real=True, positive=True)
    psi_A = A0 * sp.sinc(alpha) * sp.exp(-sp.I * beta)
    psi_B = A0 * sp.sinc(alpha) * sp.exp(sp.I * beta)

    # Densidad total de probabilidad cuántica
    I_cuantica = sp.simplify(sp.expand_complex((psi_A + psi_B) * sp.conjugate(psi_A + psi_B)))

    print("\n• Expresión simbólica de la Intensidad Cuántica I(x):")
    print("  I_cuántica(x) = 4 * A_0^2 * sinc^2(pi*a*x / (lambda*L)) * cos^2(pi*d*x / (lambda*L) + delta/2)")

    # Separación teórica entre franjas brillantes
    Delta_y = lam * L / d
    print("\n• Distancia teórica entre franjas de interferencia (Delta_y):")
    sp.pprint(sp.Eq(sp.Symbol('Delta_y'), Delta_y))

    # Visibilidad de franjas V = (Imax - Imin) / (Imax + Imin)
    print("\n• Visibilidad de Franjas en Régimen Indistinguible (Sin info de camino):")
    print("  V = (I_max - I_min) / (I_max + I_min) = 1.0 (Interferencia 100%)")
    print("\n• Visibilidad en Régimen Distinguible (Con info de camino / Medida):")
    print("  V = 0.0 (Suma Clásica de Probabilidades P_AB = P_A + P_B)")
    print("=" * 80 + "\n")


def compute_double_slit_distribution(mode, which_way, d_val, a_val, lam_val, delta_val, L_val=1.0, num_points=1000):
    """
    Calcula la distribución teórica P(x) y simula el conteo Monte Carlo de impactos individuales.
    """
    x_grid = np.linspace(-0.015, 0.015, num_points)  # Dominio espacial pantalla x (m)

    # Parámetros adimensionales
    alpha = np.pi * a_val * x_grid / (lam_val * L_val)
    beta = np.pi * d_val * x_grid / (lam_val * L_val)

    # Envolvente de difracción por ancho de rendija 'a'
    diffraction_envelope = np.sinc(alpha / np.pi)**2

    # Perfiles de probabilidad
    # Rendija A sola (desplazada a +d/2)
    P_A = diffraction_envelope * np.sinc((np.pi * a_val * (x_grid - d_val/2.0) / (lam_val * L_val)) / np.pi)**2
    # Rendija B sola (desplazada a -d/2)
    P_B = diffraction_envelope * np.sinc((np.pi * a_val * (x_grid + d_val/2.0) / (lam_val * L_val)) / np.pi)**2

    # Normalizaciones
    norm_A = trapz(P_A, x_grid)
    norm_B = trapz(P_B, x_grid)
    if norm_A > 0: P_A /= norm_A
    if norm_B > 0: P_B /= norm_B

    if mode == 'A':
        P_theory = P_A
    elif mode == 'B':
        P_theory = P_B
    else:  # Ambas rendijas abiertas
        if which_way:
            # Información sobre el camino disponible -> Suma Clásica P_AB = 0.5*P_A + 0.5*P_B
            P_theory = 0.5 * P_A + 0.5 * P_B
        else:
            # Amplitudes Cuánticas e Interferencia: I(x) = 4 * sinc^2(alpha) * cos^2(beta + delta/2)
            interference = np.cos(beta + delta_val / 2.0)**2
            P_theory = diffraction_envelope * interference

    norm_theory = trapz(P_theory, x_grid)
    if norm_theory > 0:
        P_theory /= norm_theory

    # Métricas teóricas
    Delta_y = (lam_val * L_val) / d_val
    I_max = np.max(P_theory)
    I_min = np.min(P_theory)
    visibility = (I_max - I_min) / (I_max + I_min) if (I_max + I_min) > 0 else 0.0

    return {
        'x_grid': x_grid,
        'P_theory': P_theory,
        'P_A': P_A,
        'P_B': P_B,
        'Delta_y': Delta_y,
        'visibility': visibility
    }


def sample_quantum_events(P_theory, x_grid, num_events):
    """
    Genera posiciones de impacto aleatorias de partículas individuales
    mediante Muestreo por Aceptación y Rechazo (Monte Carlo).
    """
    if num_events <= 0:
        return np.array([]), np.array([])

    x_min, x_max = x_grid[0], x_grid[-1]
    p_max = np.max(P_theory) * 1.05

    sampled_x = []
    sampled_y = []

    # Bucle eficiente de muestreo Monte Carlo
    batch_size = max(num_events * 3, 1000)
    while len(sampled_x) < num_events:
        x_cand = np.random.uniform(x_min, x_max, batch_size)
        p_cand = np.random.uniform(0, p_max, batch_size)

        # Evaluar P(x) interpolada
        p_eval = np.interp(x_cand, x_grid, P_theory)

        accept_mask = p_cand <= p_eval
        x_acc = x_cand[accept_mask]

        # Coordenada vertical y aleatoria en la pantalla para simular impacto 2D
        y_acc = np.random.uniform(-0.005, 0.005, len(x_acc))

        sampled_x.extend(x_acc)
        sampled_y.extend(y_acc)

    x_events = np.array(sampled_x[:num_events])
    y_events = np.array(sampled_y[:num_events])

    return x_events, y_events


def launch_interactive_simulation():
    """
    Simulación Animada e Interactiva completa con Matplotlib para VS Code.
    Permite controlar modo de rendijas, información sobre el camino, lambda, d, a, delta y N.
    """
    run_sympy_analysis()

    # Parámetros físicos iniciales
    mode_init = 'Ambas'        # 'A', 'B', 'Ambas'
    which_way_init = False     # False = Sin info (Interferencia), True = Con info (Clásico)
    d_init = 0.25e-3           # Separación entre rendijas: 0.25 mm
    a_init = 0.04e-3           # Ancho de rendija: 0.04 mm
    lam_init = 632.8e-9        # Longitud de onda: 632.8 nm (He-Ne)
    delta_init = 0.0           # Fase relativa
    N_init = 1000              # Número de eventos acumulados
    L_val = 1.0                # Distancia a la pantalla: 1.0 m

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 9.8), facecolor='#0b0f19')

    # Subplots: 1. Geometría 2. Pantalla 2D de Impactos 3. Histograma vs Teoría 4. Métricas
    ax_geom = fig.add_subplot(2, 2, 1, facecolor='#111827')
    ax_screen = fig.add_subplot(2, 2, 2, facecolor='#111827')
    ax_hist = fig.add_subplot(2, 2, 3, facecolor='#111827')
    ax_info = fig.add_subplot(2, 2, 4, facecolor='#111827')
    ax_info.axis('off')

    plt.subplots_adjust(left=0.07, right=0.95, top=0.93, bottom=0.28, wspace=0.25, hspace=0.34)

    # Sliders, Botones y Controles
    ax_d     = fig.add_axes([0.12, 0.19, 0.32, 0.022], facecolor='#1f2937')
    ax_a     = fig.add_axes([0.12, 0.14, 0.32, 0.022], facecolor='#1f2937')
    ax_lam   = fig.add_axes([0.12, 0.09, 0.32, 0.022], facecolor='#1f2937')

    ax_delta = fig.add_axes([0.58, 0.19, 0.32, 0.022], facecolor='#1f2937')
    ax_N     = fig.add_axes([0.58, 0.14, 0.32, 0.022], facecolor='#1f2937')

    # Botones de modo y estado
    ax_btn_mode  = fig.add_axes([0.58, 0.07, 0.12, 0.045], facecolor='#21918c')
    ax_btn_way   = fig.add_axes([0.72, 0.07, 0.14, 0.045], facecolor='#31688e')
    ax_btn_play  = fig.add_axes([0.88, 0.07, 0.08, 0.045], facecolor='#35b779')

    s_d     = Slider(ax_d,     'Sep. d (mm)', 0.10, 0.80, valinit=d_init*1e3, color='#21918c')
    s_a     = Slider(ax_a,     'Ancho a (mm)', 0.01, 0.15, valinit=a_init*1e3, color='#35b779')
    s_lam   = Slider(ax_lam,   'Onda λ (nm)', 350.0, 800.0, valinit=lam_init*1e9, color='#fde725')

    s_delta = Slider(ax_delta, 'Fase Rel. δ', 0.0, 2.0*np.pi, valinit=delta_init, color='#35b779')
    s_N     = Slider(ax_N,     'Eventos N', 10, 15000, valinit=N_init, valstep=10, color='#d97706')

    btn_mode = Button(ax_btn_mode, 'Rendijas: Ambas', color='#21918c', hovercolor='#35b779')
    btn_way  = Button(ax_btn_way,  'Info Camino: OFF', color='#31688e', hovercolor='#21918c')
    btn_play = Button(ax_btn_play, '▶ Acumular', color='#35b779', hovercolor='#fde725')

    current_mode = [mode_init]
    current_way = [which_way_init]
    is_animating = [False]

    fig._widgets = [s_d, s_a, s_lam, s_delta, s_N, btn_mode, btn_way, btn_play]

    def toggle_mode(event):
        if current_mode[0] == 'Ambas':
            current_mode[0] = 'A'
        elif current_mode[0] == 'A':
            current_mode[0] = 'B'
        else:
            current_mode[0] = 'Ambas'
        btn_mode.label.set_text(f'Rendijas: {current_mode[0]}')
        draw_frame(int(s_N.val))

    def toggle_way(event):
        current_way[0] = not current_way[0]
        btn_way.label.set_text(f'Info Camino: {"ON" if current_way[0] else "OFF"}')
        btn_way.ax.set_facecolor('#d97706' if current_way[0] else '#31688e')
        draw_frame(int(s_N.val))

    def toggle_play(event):
        is_animating[0] = not is_animating[0]
        btn_play.label.set_text('❚❚ Pausa' if is_animating[0] else '▶ Acumular')
        btn_play.ax.set_facecolor('#fde725' if is_animating[0] else '#35b779')
        fig.canvas.draw_idle()

    btn_mode.on_clicked(toggle_mode)
    btn_way.on_clicked(toggle_way)
    btn_play.on_clicked(toggle_play)

    def draw_frame(N_events):
        d_val = s_d.val * 1e-3
        a_val = s_a.val * 1e-3
        lam_val = s_lam.val * 1e-9
        delta_val = s_delta.val

        res = compute_double_slit_distribution(
            current_mode[0], current_way[0], d_val, a_val, lam_val, delta_val, L_val
        )

        x_grid = res['x_grid'] * 1e3  # Convertir a mm para visualización
        P_theory = res['P_theory']

        # Muestreo Monte Carlo de impactos discretos
        x_ev, y_ev = sample_quantum_events(P_theory, x_grid, N_events)

        # 1. Esquema de Geometría Cuántica
        ax_geom.clear()
        ax_geom.axvline(0, color='gray', linestyle='--', alpha=0.5, label='Eje Óptico')

        # Placa de Rendijas en z=0
        ax_geom.plot([0, 0], [-10, 10], color='white', linewidth=3)
        if current_mode[0] in ['A', 'Ambas']:
            ax_geom.plot(0, d_val*1e3/2.0, 'o', color='#21918c', markersize=8, label='Rendija A')
        if current_mode[0] in ['B', 'Ambas']:
            ax_geom.plot(0, -d_val*1e3/2.0, 'o', color='#35b779', markersize=8, label='Rendija B')

        # Trazo simbólico de fotón / electrón individual
        if len(x_ev) > 0:
            last_x = x_ev[-1]
            ax_geom.plot([0, 1.0], [0, last_x], color='#fde725', linestyle=':', alpha=0.8, label='Última Partícula')

        ax_geom.set_xlim([-0.1, 1.1])
        ax_geom.set_ylim([-12, 12])
        ax_geom.set_xlabel('Distancia z (m)', color='white', fontsize=9)
        ax_geom.set_ylabel('Posición x (mm)', color='white', fontsize=9)
        ax_geom.set_title('Esquema del Dispositivo Cuántico', color='#35b779', fontsize=11, fontweight='bold')
        ax_geom.grid(True, color='#222222', linestyle='--', alpha=0.5)
        ax_geom.legend(loc='upper left', facecolor='#111827', edgecolor='#35b779', fontsize=7.5)

        # 2. Detector 2D: Conteo de Impactos Individuales en Pantalla
        ax_screen.clear()
        if len(x_ev) > 0:
            ax_screen.scatter(x_ev, y_ev*1e3, s=4, c='#fde725', alpha=0.6, edgecolors='none', label=f'{N_events} Impactos')

        ax_screen.set_xlim([x_grid[0], x_grid[-1]])
        ax_screen.set_ylim([-5, 5])
        ax_screen.set_xlabel('Posición Impacto x (mm)', color='white', fontsize=9)
        ax_screen.set_ylabel('Eje Vertical Pantalla (mm)', color='white', fontsize=9)
        ax_screen.set_title(f'Pantalla Detectora (Un Evento a la Vez - N = {N_events})', color='#fde725', fontsize=11, fontweight='bold')
        ax_screen.grid(True, color='#222222', linestyle='--', alpha=0.5)
        ax_screen.legend(loc='upper right', facecolor='#111827', edgecolor='#fde725', fontsize=7.5)

        # 3. Histograma Acumulado vs Curva Teórica
        ax_hist.clear()
        if len(x_ev) > 0:
            ax_hist.hist(x_ev, bins=70, density=True, color='#21918c', alpha=0.55, edgecolor='#35b779', label='Conteo Acumulado')

        ax_hist.plot(x_grid, P_theory, color='#fde725', linewidth=2.2, label=r'Probabilidad Teórica $P(x) = |\psi_A + \psi_B|^2$')

        ax_hist.set_xlim([x_grid[0], x_grid[-1]])
        ax_hist.set_xlabel('Posición en Pantalla x (mm)', color='white', fontsize=9)
        ax_hist.set_ylabel('Densidad de Probabilidad', color='white', fontsize=9)
        ax_hist.set_title('Emergencia Estadística del Patrón de Interferencia', color='#21918c', fontsize=11, fontweight='bold')
        ax_hist.grid(True, color='#222222', linestyle='--', alpha=0.5)
        ax_hist.legend(loc='upper right', facecolor='#111827', edgecolor='#21918c', fontsize=7.5)

        # 4. Panel de Métricas Cuánticas e Información
        ax_info.clear()
        ax_info.axis('off')

        info_text = (
            "=== PARÁMETROS Y MÉTRICAS CUÁNTICAS ===\n\n"
            f" • Modo Rendijas        : {current_mode[0]}\n"
            f" • Info sobre el Camino : {'ON (Suma Clásica)' if current_way[0] else 'OFF (Interferencia)'}\n"
            f" • Partículas Detección : {N_events} eventos\n\n"
            "--- Geometría del Experimento ---\n"
            f" • Separación Rendijas (d) : {s_d.val:.2f} mm\n"
            f" • Ancho Rendijas (a)      : {s_a.val:.3f} mm\n"
            f" • Longitud de Onda (λ)   : {s_lam.val:.1f} nm\n"
            f" • Fase Relativa (δ)      : {delta_val:.2f} rad ({np.degrees(delta_val):.1f}°)\n\n"
            "--- Resultados Observables ---\n"
            f" • Espaciado Franjas (Δy) : {res['Delta_y']*1e3:.3f} mm\n"
            f" • Visibilidad Franjas (V): {res['visibility']:.4f}  (1.0 = Máxima, 0.0 = Nula)\n"
            f" • Regla Cuántica         : P(x) = |ψ_A + ψ_B|²\n"
        )
        ax_info.text(0.05, 0.95, info_text, color='#e2e8f0', fontsize=9.2, fontfamily='monospace',
                     verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.8', facecolor='#1f2937', edgecolor='#35b779', alpha=0.95))

    def animate(frame):
        if is_animating[0]:
            N_next = s_N.val + 200
            if N_next > s_N.valmax:
                N_next = s_N.valmin
            s_N.set_val(N_next)
        else:
            draw_frame(int(s_N.val))

    def on_slider_change(val):
        if not is_animating[0]:
            draw_frame(int(s_N.val))
            fig.canvas.draw_idle()

    s_d.on_changed(on_slider_change)
    s_a.on_changed(on_slider_change)
    s_lam.on_changed(on_slider_change)
    s_delta.on_changed(on_slider_change)
    s_N.on_changed(on_slider_change)

    anim = FuncAnimation(fig, animate, interval=100, cache_frame_data=False)
    fig._anim = anim

    draw_frame(int(s_N.val))
    plt.suptitle("LABORATORIO PYTHON 4: DOBLE RENDIJA - UN EVENTO A LA VEZ (VS Code - Viridis)",
                 color='white', fontsize=13, fontweight='bold', y=0.98)
    plt.show()


def export_static_sample(output_path="laboratorio4_doble_rendija_viridis.png"):
    """Genera una imagen estática de prueba para verificar compatibilidad."""
    run_sympy_analysis()
    res = compute_double_slit_distribution('Ambas', False, 0.25e-3, 0.04e-3, 632.8e-9, 0.0)

    dir_name = os.path.dirname(os.path.abspath(output_path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8), facecolor='#0b0f19')
    ax2d = fig.add_subplot(1, 1, 1, facecolor='#111827')
    ax2d.plot(res['x_grid']*1e3, res['P_theory'], color='#fde725', linewidth=2.5, label=r'$P(x) = |\psi_A + \psi_B|^2$')
    ax2d.set_title("Muestra Estática del Laboratorio 4: Doble Rendija", color='white')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] Muestra estática del Lab 4 generada en: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[0] != '--test':
        launch_interactive_simulation()
    else:
        export_static_sample()
