"""
================================================================================
LABORATORIO PYTHON 5: INTERFERÓMETRO DE MACH-ZEHNDER VIRTUAL
(Teoría, Monte Carlo Estado por Estado y Efecto Elitzur-Vaidman)
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

# Forzado de Backend interactivo para ejecuciones en terminal / VS Code
if len(sys.argv) == 1 or sys.argv != '--test':
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass


def run_sympy_analysis():
    """
    Análisis algebraico estricto con SymPy para el Interferómetro de Mach-Zehnder:
    Matriz del divisor de haz 50:50, desfasador, opacidad en brazo B,
    probabilidades de detección y efecto Elitzur-Vaidman.
    """
    print("=" * 80)
    print(" 1. ANÁLISIS SIMBÓLICO CON SYMPY - LABORATORIO 5: MACH-ZEHNDER VIRTUAL")
    print("=" * 80)

    phi, eta = sp.symbols('phi eta', real=True, nonnegative=True)

    # 1. Matriz de Divisor de Haz simétrico 50:50 (Hecht Cap. 9)
    # Reflejado adquiere fase i = exp(i*pi/2)
    U_BS = (1 / sp.sqrt(2)) * sp.Matrix([[1, sp.I], [sp.I, 1]])

    # 2. Estado inicial: Entrada en Puerto 1 (Brazo A) |psi_in> = [1, 0]^T
    psi_in = sp.Matrix([1, 0])

    # Estado tras BS1
    psi_after_BS1 = U_BS * psi_in  # [1/sqrt(2), i/sqrt(2)]^T

    # Estado tras Desfasador en A y Atenuador/Bomba en B (factor sqrt(1-eta))
    a_A = (1 / sp.sqrt(2)) * sp.exp(sp.I * phi)
    a_B = (sp.I / sp.sqrt(2)) * sp.sqrt(1 - eta)
    psi_before_BS2 = sp.Matrix([a_A, a_B])

    # Estado final tras BS2
    psi_out = U_BS * psi_before_BS2

    # Probabilidades analíticas simplificadas
    P_D1_expr = sp.Rational(1, 2) - eta/4 - sp.Rational(1, 2) * sp.sqrt(1 - eta) * sp.cos(phi)
    P_D2_expr = sp.Rational(1, 2) - eta/4 + sp.Rational(1, 2) * sp.sqrt(1 - eta) * sp.cos(phi)
    P_bomb_expr = eta / 2

    print("\n• Estado de Amplitudes a la Salida [A_D1, A_D2]^T:")
    sp.pprint(psi_out)

    print("\n• Probabilidad Teórica en Detector D1 (Puerto Oscuro para phi=0):")
    sp.pprint(sp.Eq(sp.Symbol('P(D1)'), P_D1_expr))

    print("\n• Probabilidad Teórica en Detector D2 (Puerto Brillante para phi=0):")
    sp.pprint(sp.Eq(sp.Symbol('P(D2)'), P_D2_expr))

    print("\n• Probabilidad de Absorción / Bloqueo en Brazo B (P_bomba):")
    sp.pprint(sp.Eq(sp.Symbol('P_bomba'), P_bomb_expr))

    # Casos límite
    P_D1_coh = P_D1_expr.subs(eta, 0)
    P_D2_coh = P_D2_expr.subs(eta, 0)

    P_D1_ev = P_D1_expr.subs({eta: 1, phi: 0})
    P_D2_ev = P_D2_expr.subs({eta: 1, phi: 0})

    print("\n• Régimen Coherente Puro (eta = 0):")
    print(f"  • P(D1) = {P_D1_coh}")
    print(f"  • P(D2) = {P_D2_coh}")

    print("\n• Régimen Elitzur-Vaidman (eta = 1, phi = 0 - Bomba en Brazo B):")
    print(f"  • P(D1 - Detector Oscuro) : {P_D1_ev}  (25% --> ¡Detección Libre de Interacción!)")
    print(f"  • P(D2 - Detector Principal): {P_D2_ev}  (25%)")
    print(f"  • P(Bomba / Absorción B)  : {P_bomb_expr.subs(eta, 1)}  (50% --> ¡Explotó!)")
    print("=" * 80 + "\n")


def get_theoretical_probabilities(phi_val, eta_val):
    """
    Calcula las probabilidades teóricas exactas P(D1), P(D2) y P(bomba),
    así como la visibilidad V = sqrt(1 - eta).
    """
    V = np.sqrt(max(1.0 - eta_val, 0.0))
    p_d1 = 0.5 - 0.25 * eta_val - 0.5 * V * np.cos(phi_val)
    p_d2 = 0.5 - 0.25 * eta_val + 0.5 * V * np.cos(phi_val)
    p_bomb = 0.5 * eta_val

    p_d1 = max(0.0, min(1.0, p_d1))
    p_d2 = max(0.0, min(1.0, p_d2))
    p_bomb = max(0.0, min(1.0, p_bomb))

    total = p_d1 + p_d2 + p_bomb
    if total > 0:
        p_d1 /= total
        p_d2 /= total
        p_bomb /= total

    return p_d1, p_d2, p_bomb, V


class MachZehnderState:
    """
    Mantiene el estado acumulativo estocástico de fotones para evitar
    fluctuaciones caóticas y simular la acumulación realista evento a evento.
    """
    def __init__(self):
        self.count_D1 = 0
        self.count_D2 = 0
        self.count_bomb = 0
        self.total_photons = 0
        self.impacts_D1_y = []
        self.impacts_D1_z = []
        self.impacts_D2_y = []
        self.impacts_D2_z = []

    def reset(self):
        self.count_D1 = 0
        self.count_D2 = 0
        self.count_bomb = 0
        self.total_photons = 0
        self.impacts_D1_y = []
        self.impacts_D1_z = []
        self.impacts_D2_y = []
        self.impacts_D2_z = []

    def add_photons(self, batch_size, phi_val, eta_val):
        p_d1, p_d2, p_bomb, V = get_theoretical_probabilities(phi_val, eta_val)
        probs = [p_d1, p_d2, p_bomb]

        events = np.random.choice([0, 1, 2], size=batch_size, p=probs)

        n_d1 = np.sum(events == 0)
        n_d2 = np.sum(events == 1)
        n_bomb = np.sum(events == 2)

        self.count_D1 += n_d1
        self.count_D2 += n_d2
        self.count_bomb += n_bomb
        self.total_photons += batch_size

        # Muestreo espacial de posiciones de impacto en detectores 2D
        k_y = 4.0 * np.pi  # Frecuencia espacial de franjas

        # Para D1
        if n_d1 > 0:
            y_samples = []
            while len(y_samples) < n_d1:
                y_cand = np.random.uniform(-1.0, 1.0)
                prob_y = 0.5 * (1.0 - V * np.cos(k_y * y_cand + phi_val)) if V > 0 else 0.5
                if np.random.rand() <= prob_y:
                    y_samples.append(y_cand)
            z_samples = np.random.normal(0.0, 0.25, size=n_d1)
            self.impacts_D1_y.extend(y_samples)
            self.impacts_D1_z.extend(z_samples)

        # Para D2
        if n_d2 > 0:
            y_samples = []
            while len(y_samples) < n_d2:
                y_cand = np.random.uniform(-1.0, 1.0)
                prob_y = 0.5 * (1.0 + V * np.cos(k_y * y_cand + phi_val)) if V > 0 else 0.5
                if np.random.rand() <= prob_y:
                    y_samples.append(y_cand)
            z_samples = np.random.normal(0.0, 0.25, size=n_d2)
            self.impacts_D2_y.extend(y_samples)
            self.impacts_D2_z.extend(z_samples)

        # Limitar número de puntos visuales mantenidos para rendimiento
        max_points = 1500
        if len(self.impacts_D1_y) > max_points:
            self.impacts_D1_y = self.impacts_D1_y[-max_points:]
            self.impacts_D1_z = self.impacts_D1_z[-max_points:]
        if len(self.impacts_D2_y) > max_points:
            self.impacts_D2_y = self.impacts_D2_y[-max_points:]
            self.impacts_D2_z = self.impacts_D2_z[-max_points:]


def launch_interactive_simulation():
    """Simulación interactiva y animada completa para VS Code / Tkinter."""
    run_sympy_analysis()

    mz_state = MachZehnderState()

    phi_init = 0.0
    opacity_init = 0.0
    rate_init = 50

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 9.8), facecolor='#0b0f19')

    # Subplots
    ax_schema  = fig.add_subplot(2, 2, 1, facecolor='#111827')
    ax_curves  = fig.add_subplot(2, 2, 2, facecolor='#111827')
    ax_screens = fig.add_subplot(2, 2, 3, facecolor='#111827')
    ax_info    = fig.add_subplot(2, 2, 4, facecolor='#111827')
    ax_info.axis('off')

    plt.subplots_adjust(left=0.07, right=0.95, top=0.93, bottom=0.26, wspace=0.25, hspace=0.34)

    # Sliders y Botones
    ax_phi     = fig.add_axes([0.12, 0.16, 0.32, 0.022], facecolor='#1f2937')
    ax_opacity = fig.add_axes([0.12, 0.11, 0.32, 0.022], facecolor='#1f2937')
    ax_rate    = fig.add_axes([0.12, 0.06, 0.32, 0.022], facecolor='#1f2937')

    ax_btn_play  = fig.add_axes([0.58, 0.12, 0.14, 0.05], facecolor='#21918c')
    ax_btn_reset = fig.add_axes([0.76, 0.12, 0.14, 0.05], facecolor='#31688e')

    s_phi     = Slider(ax_phi,     'Fase φ (rad)', 0.0, 4.0 * np.pi, valinit=phi_init, color='#21918c')
    s_opacity = Slider(ax_opacity, 'Bloqueo B (η)', 0.0, 1.0, valinit=opacity_init, color='#35b779')
    s_rate    = Slider(ax_rate,    'Tasa (fot/frame)', 10, 200, valinit=rate_init, valstep=10, color='#fde725')

    btn_play  = Button(ax_btn_play,  '▶ Acumular Fotones', color='#21918c', hovercolor='#35b779')
    btn_reset = Button(ax_btn_reset, '↺ Reiniciar', color='#31688e', hovercolor='#21918c')

    fig._widgets = [s_phi, s_opacity, s_rate, btn_play, btn_reset]
    is_animating = [False]

    def toggle_play(event):
        is_animating[0] = not is_animating[0]
        btn_play.label.set_text('❚❚ Pausa' if is_animating[0] else '▶ Acumular Fotones')
        btn_play.ax.set_facecolor('#fde725' if is_animating[0] else '#21918c')
        fig.canvas.draw_idle()

    def reset_sim(event):
        mz_state.reset()
        draw_frame(s_phi.val)
        fig.canvas.draw_idle()

    btn_play.on_clicked(toggle_play)
    btn_reset.on_clicked(reset_sim)

    def draw_frame(phi_val):
        opacity_val = s_opacity.val
        p_d1, p_d2, p_bomb, V = get_theoretical_probabilities(phi_val, opacity_val)

        # ---------------- 1. Esquema Óptico 2D ----------------
        ax_schema.clear()
        ax_schema.set_facecolor('#111827')

        ax_schema.plot([0, 2], [1, 1], color='#35b779', linewidth=2.5)  # Entrada
        color_A = '#21918c'
        color_B = '#fde725' if opacity_val < 0.5 else '#d97706'

        # Brazo A
        ax_schema.plot([2, 2], [1, 4], color=color_A, linewidth=2.5)
        ax_schema.plot([2, 6], [4, 4], color=color_A, linewidth=2.5)

        # Brazo B
        ax_schema.plot([2, 6], [1, 1], color=color_B, linewidth=2.5)
        ax_schema.plot([6, 6], [1, 4], color=color_B, linewidth=2.5)

        # Salidas a Detectores
        color_out1 = '#35b779' if p_d1 > 0.01 else 'gray'
        color_out2 = '#fde725' if p_d2 > 0.01 else 'gray'
        ax_schema.plot([6, 8], [4, 4], color=color_out1, linewidth=2.2, linestyle='--')
        ax_schema.plot([6, 6], [4, 6], color=color_out2, linewidth=2.2, linestyle='--')

        # Componentes Ópticos (BS y Espejos)
        ax_schema.plot([1.7, 2.3], [0.7, 1.3], color='white', linewidth=3, label='BS1 (50:50)')
        ax_schema.plot([5.7, 6.3], [3.7, 4.3], color='white', linewidth=3, label='BS2 (50:50)')
        ax_schema.plot([1.7, 2.3], [4.3, 3.7], color='#94a3b8', linewidth=3)  # M1
        ax_schema.plot([5.7, 6.3], [1.3, 0.7], color='#94a3b8', linewidth=3)  # M2

        # Desfasador φ
        ax_schema.plot([4, 4], [3.7, 4.3], color='#21918c', linewidth=6)
        ax_schema.text(4, 4.5, f'φ = {phi_val:.2f} rad', color='#21918c', fontsize=9, ha='center', fontweight='bold')

        # Bloqueador / Bomba en Brazo B
        if opacity_val > 0.0:
            ax_schema.plot([4, 4], [0.7, 1.3], color='#ef4444', linewidth=8 * opacity_val)
            ax_schema.text(4, 0.3, f'Bomba B: {opacity_val*100:.0f}%', color='#ef4444', fontsize=8.5, ha='center', fontweight='bold')

        # Detectores D1 y D2
        ax_schema.plot(8, 4, 's', color=color_out1, markersize=12, label='Detector D1 (Oscuro para φ=0)')
        ax_schema.plot(6, 6, 's', color=color_out2, markersize=12, label='Detector D2 (Brillante para φ=0)')

        ax_schema.set_xlim([-0.5, 9.0])
        ax_schema.set_ylim([-0.5, 7.0])
        ax_schema.axis('off')
        ax_schema.set_title('Esquema Óptico del Interferómetro de Mach-Zehnder', color='#35b779', fontsize=11, fontweight='bold')
        ax_schema.legend(loc='lower right', facecolor='#111827', edgecolor='#21918c', fontsize=7.5)

        # ---------------- 2. Curvas Teóricas P(D1) y P(D2) vs φ ----------------
        ax_curves.clear()
        phi_grid = np.linspace(0, 4.0 * np.pi, 500)

        p_d1_grid = 0.5 - 0.25 * opacity_val - 0.5 * V * np.cos(phi_grid)
        p_d2_grid = 0.5 - 0.25 * opacity_val + 0.5 * V * np.cos(phi_grid)

        ax_curves.plot(phi_grid, p_d1_grid, color='#35b779', linewidth=2.2, label=r'$P(D_1) = \frac{1}{2}(1 - V\cos\phi)$')
        ax_curves.plot(phi_grid, p_d2_grid, color='#fde725', linewidth=2.2, label=r'$P(D_2) = \frac{1}{2}(1 + V\cos\phi)$')

        ax_curves.axvline(phi_val, color='#e2e8f0', linestyle=':', linewidth=1.8, label=f'Fase Actual φ = {phi_val:.2f}')
        ax_curves.plot(phi_val, p_d1, 'o', color='#35b779', markersize=8)
        ax_curves.plot(phi_val, p_d2, 'o', color='#fde725', markersize=8)

        ax_curves.set_xlim([0, 4.0 * np.pi])
        ax_curves.set_ylim([-0.05, 1.1])
        ax_curves.set_xlabel('Desfase φ (radianes)', color='white', fontsize=9)
        ax_curves.set_ylabel('Probabilidad de Detección', color='white', fontsize=9)
        ax_curves.set_title(f'Franjas de Interferencia (Visibilidad V = {V:.3f})', color='#21918c', fontsize=11, fontweight='bold')
        ax_curves.grid(True, color='#222222', linestyle='--', alpha=0.5)
        ax_curves.legend(loc='upper right', facecolor='#111827', edgecolor='#21918c', fontsize=7.5)

        # ---------------- 3. Pantallas 2D e Histograma Acumulado ----------------
        ax_screens.clear()

        # Placeholders invisibles para garantizar que la leyenda no falle antes de haber fotones
        ax_screens.scatter([], [], c='#fde725', s=8, alpha=0.6, label='Pantalla D2 (Fotones)')
        ax_screens.scatter([], [], c='#35b779', s=8, alpha=0.6, label='Pantalla D1 (Fotones)')

        if mz_state.total_photons > 0:
            if len(mz_state.impacts_D2_y) > 0:
                ax_screens.scatter(mz_state.impacts_D2_y, np.array(mz_state.impacts_D2_z) + 1.2,
                                   c='#fde725', s=8, alpha=0.6)

            if len(mz_state.impacts_D1_y) > 0:
                ax_screens.scatter(mz_state.impacts_D1_y, np.array(mz_state.impacts_D1_z) - 1.2,
                                   c='#35b779', s=8, alpha=0.6)

        ax_screens.axhline(1.2, color='#fde725', linestyle='--', alpha=0.3)
        ax_screens.axhline(-1.2, color='#35b779', linestyle='--', alpha=0.3)
        ax_screens.set_xlim([-1.2, 1.2])
        ax_screens.set_ylim([-2.5, 2.5])
        ax_screens.set_yticks([-1.2, 1.2])
        ax_screens.set_yticklabels(['Pantalla D1', 'Pantalla D2'], color='white', fontsize=9, fontweight='bold')
        ax_screens.set_xlabel('Posición Transversal y (mm)', color='white', fontsize=9)
        ax_screens.set_title(f'Impactos de Fotones en Detectores 2D (Acumulados N = {mz_state.total_photons})',
                             color='#fde725', fontsize=11, fontweight='bold')
        ax_screens.grid(True, color='#222222', linestyle='--', alpha=0.5)
        ax_screens.legend(loc='upper right', facecolor='#111827', edgecolor='#21918c', fontsize=7.5)

        # ---------------- 4. Panel de Métricas e Información ----------------
        ax_info.clear()
        ax_info.axis('off')

        is_ev_mode = (opacity_val > 0.8) and (abs(phi_val % (2*np.pi)) < 0.2)
        has_ev_event = is_ev_mode and (mz_state.count_D1 > 0)

        info_text = (
            "=== MÉTRICAS Y COHERENCIA CUÁNTICA ===\n\n"
            f" • Desfase Óptico (φ)     : {phi_val:.2f} rad ({np.degrees(phi_val):.1f}°)\n"
            f" • Opacidad Bomba B (η)   : {opacity_val*100:.1f}%\n"
            f" • Visibilidad Teórica (V): {V:.4f}\n\n"
            "--- Probabilidades Teóricas Exactas ---\n"
            f" • P(D1) [Puerto Oscuro]    : {p_d1*100:.2f}%\n"
            f" • P(D2) [Puerto Brillante] : {p_d2*100:.2f}%\n"
            f" • P(Bomba / Absorción B)  : {p_bomb*100:.2f}%\n\n"
            f"--- Conteo Acumulado Monte Carlo (N = {mz_state.total_photons}) ---\n"
            f" • Detecciones D1 : {mz_state.count_D1}  ({(mz_state.count_D1/max(mz_state.total_photons,1))*100:.1f}%)\n"
            f" • Detecciones D2 : {mz_state.count_D2}  ({(mz_state.count_D2/max(mz_state.total_photons,1))*100:.1f}%)\n"
            f" • Absorciones B : {mz_state.count_bomb}  ({(mz_state.count_bomb/max(mz_state.total_photons,1))*100:.1f}%)\n\n"
            "--- Efecto Elitzur-Vaidman (Medición Libre de Interacción) ---\n"
            f" • Modo Bomba B: {'ACTIVO (η=100%, φ=0)' if is_ev_mode else 'Inactivo (Ajuste η=1.0, φ=0)'}\n"
            f" • Estado: {'¡DETECCIÓN LIBRE DE INTERACCIÓN CONFIRMADA EN D1!' if has_ev_event else 'Esperando acumulación de fotones...' if is_ev_mode else 'Régimen Estándar'}\n"
        )

        ax_info.text(0.05, 0.95, info_text, color='#e2e8f0', fontsize=9.0, fontfamily='monospace',
                     verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.8', facecolor='#1f2937', edgecolor='#35b779' if has_ev_event else '#21918c', alpha=0.95))

    def animate(frame):
        if is_animating[0]:
            batch_size = int(s_rate.val)
            mz_state.add_photons(batch_size, s_phi.val, s_opacity.val)
            draw_frame(s_phi.val)
        else:
            draw_frame(s_phi.val)

    def on_slider_change(val):
        if not is_animating[0]:
            draw_frame(s_phi.val)
            fig.canvas.draw_idle()

    s_phi.on_changed(on_slider_change)
    s_opacity.on_changed(on_slider_change)
    s_rate.on_changed(on_slider_change)

    anim = FuncAnimation(fig, animate, interval=50, cache_frame_data=False)
    fig._anim = anim

    draw_frame(s_phi.val)
    plt.suptitle("LABORATORIO PYTHON 5: INTERFERÓMETRO DE MACH-ZEHNDER Y EFECTO ELITZUR-VAIDMAN",
                 color='white', fontsize=13, fontweight='bold', y=0.98)
    plt.show()


def export_static_sample(output_path="laboratorio5_mach_zehnder_viridis.png"):
    """Genera una imagen estática de prueba para verificar funcionamiento."""
    run_sympy_analysis()
    p_d1, p_d2, p_bomb, V = get_theoretical_probabilities(0.0, 0.0)

    dir_name = os.path.dirname(os.path.abspath(output_path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8), facecolor='#0b0f19')
    ax = fig.add_subplot(1, 1, 1, facecolor='#111827')
    phi_grid = np.linspace(0, 4*np.pi, 500)
    ax.plot(phi_grid, np.sin(phi_grid/2.0)**2, color='#35b779', linewidth=2.5, label='P(D1)')
    ax.plot(phi_grid, np.cos(phi_grid/2.0)**2, color='#fde725', linewidth=2.5, label='P(D2)')
    ax.set_title("Muestra Estática del Laboratorio 5", color='white')
    plt.legend()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] Muestra estática generada en: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv != '--test':
        launch_interactive_simulation()
    else:
        export_static_sample()
