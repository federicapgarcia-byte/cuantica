"""
================================================================================
LABORATORIO PYTHON 5: INTERFERÓMETRO DE MACH-ZEHNDER VIRTUAL
================================================================================
Materia: Mecánica Cuántica | Referencia Óptica: Eugene Hecht (Cap. 9: Interferencia)
Entorno: Visual Studio Code / Python 3
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

# Forzado de Backend interactivo para ejecuciones en terminal / VS Code
if len(sys.argv) == 1 or sys.argv[1] != '--test':
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass


def run_sympy_analysis():
    """
    Derivación simbólica con SymPy de la matriz de transferencia del interferómetro
    de Mach-Zehnder, amplitudes de probabilidad y efecto Elitzur-Vaidman.
    """
    print("=" * 80)
    print(" 1. ANÁLISIS SIMBÓLICO CON SYMPY - LABORATORIO 5: MACH-ZEHNDER VIRTUAL")
    print("=" * 80)

    phi = sp.Symbol('phi', real=True)
    
    # Amplitudes exactas del sistema
    # Puerto Brillante D1: A_D1 = 1/2 * (exp(i*phi) + 1)
    # Puerto Oscuro D2   : A_D2 = 1/2 * (exp(i*phi) - 1)
    A_D1 = (sp.exp(sp.I * phi) + 1) / 2
    A_D2 = (sp.exp(sp.I * phi) - 1) / 2

    # Probabilidades de detección P(D1) y P(D2)
    P_D1_expr = sp.simplify(sp.expand_complex(sp.Abs(A_D1)**2))
    P_D2_expr = sp.simplify(sp.expand_complex(sp.Abs(A_D2)**2))

    print("\n• Probabilidad en Detector D1 (Puerto Brillante / Constructivo en phi=0):")
    sp.pprint(sp.Eq(sp.Symbol('P(D1)'), P_D1_expr))

    print("\n• Probabilidad en Detector D2 (Puerto Oscuro / Destructivo en phi=0):")
    sp.pprint(sp.Eq(sp.Symbol('P(D2)'), P_D2_expr))

    print("\n• Evaluación para phi = 0 (Caminos de igual longitud):")
    print("  P(D1) = cos^2(0) = 1.0  (100% - Puerto Brillante / Interferencia Constructiva)")
    print("  P(D2) = sin^2(0) = 0.0  (0%   - Puerto Oscuro    / Interferencia Destructiva)")

    print("\n• Visibilidad Teórica de Interferencia V (Sin medición de camino):")
    print("  V = (P_max - P_min) / (P_max + P_min) = 1.0 (Coherencia Cuántica Total)")

    print("\n• Medición Libre de Interacción (Efecto Elitzur-Vaidman / Bloqueador en Brazo B):")
    print("  Si se coloca un objeto opaco en el Brazo B a phi = 0:")
    print("   - Probabilidad de Absorción en Objeto : P_absorbed = 0.50 (50%)")
    print("   - Probabilidad en Detector D1 (Brillante): P(D1) = 0.25 (25%)")
    print("   - Probabilidad en Detector D2 (Oscuro):    P(D2) = 0.25 (25%) --> ¡Detección sin interacción!")
    print("=" * 80 + "\n")


def compute_mach_zehnder(phi_val, opacity_B=0.0, num_photons=1000):
    """
    Calcula probabilidades teóricas exactas y ejecuta simulación Monte Carlo evento a evento.
    
    Convención Estándar de Mach-Zehnder (para phi = 0, opacity_B = 0):
      - Detector D1 (Puerto Brillante / Constructivo) : P(D1) = cos^2(phi/2) = 1.0
      - Detector D2 (Puerto Oscuro / Destructivo)     : P(D2) = sin^2(phi/2) = 0.0
    """
    eta = max(0.0, min(1.0, opacity_B))
    sqrt_1_eta = np.sqrt(1.0 - eta)

    # Amplitudes complejas
    # A_D1 = 1/2 * (exp(i*phi) + sqrt(1-eta))
    # A_D2 = 1/2 * (exp(i*phi) - sqrt(1-eta))
    A_D1 = 0.5 * (np.exp(1j * phi_val) + sqrt_1_eta)
    A_D2 = 0.5 * (np.exp(1j * phi_val) - sqrt_1_eta)

    P_D1_real = np.abs(A_D1)**2
    P_D2_real = np.abs(A_D2)**2
    P_block_real = 0.5 * eta

    # Normalización estricta
    total_P = P_D1_real + P_D2_real + P_block_real
    P_D1_real /= total_P
    P_D2_real /= total_P
    P_block_real /= total_P

    # Visibilidad V = sqrt(1 - eta)
    visibility = sqrt_1_eta

    # Muestreo Monte Carlo evento a evento
    probs = [P_D1_real, P_D2_real, P_block_real]
    events = np.random.choice([0, 1, 2], size=num_photons, p=probs)
    
    count_D1 = np.sum(events == 0)
    count_D2 = np.sum(events == 1)
    count_block = np.sum(events == 2)

    return {
        'phi': phi_val,
        'opacity_B': eta,
        'P_D1_real': P_D1_real,
        'P_D2_real': P_D2_real,
        'P_block_real': P_block_real,
        'count_D1': count_D1,
        'count_D2': count_D2,
        'count_block': count_block,
        'visibility': visibility,
        'num_photons': num_photons,
        'events': events
    }


def launch_interactive_simulation():
    """Simulación animada e interactiva para VS Code con interfaz Tkinter y Viridis."""
    run_sympy_analysis()

    phi_init = 0.0
    opacity_init = 0.0
    N_init = 1000

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 9.8), facecolor='#0b0f19')

    # Subplots
    ax_schema = fig.add_subplot(2, 2, 1, facecolor='#111827')
    ax_curves = fig.add_subplot(2, 2, 2, facecolor='#111827')
    ax_hist   = fig.add_subplot(2, 2, 3, facecolor='#111827')
    ax_info   = fig.add_subplot(2, 2, 4, facecolor='#111827')
    ax_info.axis('off')

    plt.subplots_adjust(left=0.07, right=0.95, top=0.93, bottom=0.26, wspace=0.25, hspace=0.34)

    # Sliders y Botones
    ax_phi     = fig.add_axes([0.12, 0.16, 0.32, 0.022], facecolor='#1f2937')
    ax_opacity = fig.add_axes([0.12, 0.11, 0.32, 0.022], facecolor='#1f2937')
    ax_N       = fig.add_axes([0.12, 0.06, 0.32, 0.022], facecolor='#1f2937')

    ax_btn_play  = fig.add_axes([0.58, 0.12, 0.14, 0.05], facecolor='#21918c')
    ax_btn_reset = fig.add_axes([0.76, 0.12, 0.14, 0.05], facecolor='#31688e')

    s_phi     = Slider(ax_phi,     'Fase φ (rad)', 0.0, 4.0 * np.pi, valinit=phi_init, color='#21918c')
    s_opacity = Slider(ax_opacity, 'Bloqueo B', 0.0, 1.0, valinit=opacity_init, color='#35b779')
    s_N       = Slider(ax_N,       'Eventos N', 100, 10000, valinit=N_init, valstep=100, color='#fde725')

    btn_play  = Button(ax_btn_play,  '▶ Acumular Fotones', color='#21918c', hovercolor='#35b779')
    btn_reset = Button(ax_btn_reset, '↺ Reiniciar', color='#31688e', hovercolor='#21918c')

    fig._widgets = [s_phi, s_opacity, s_N, btn_play, btn_reset]
    is_animating = [False]

    def toggle_play(event):
        is_animating[0] = not is_animating[0]
        btn_play.label.set_text('❚❚ Pausa' if is_animating[0] else '▶ Acumular Fotones')
        btn_play.ax.set_facecolor('#fde725' if is_animating[0] else '#21918c')
        fig.canvas.draw_idle()

    def reset_sim(event):
        is_animating[0] = False
        btn_play.label.set_text('▶ Acumular Fotones')
        btn_play.ax.set_facecolor('#21918c')
        draw_frame(s_phi.val)
        fig.canvas.draw_idle()

    btn_play.on_clicked(toggle_play)
    btn_reset.on_clicked(reset_sim)

    def draw_frame(phi_val):
        opacity_val = s_opacity.val
        N_val = int(s_N.val)

        res = compute_mach_zehnder(phi_val, opacity_val, N_val)

        # 1. Esquema Geométrico del Interferómetro (Hecht Cap. 9)
        ax_schema.clear()
        ax_schema.set_facecolor('#111827')
        
        # Posiciones del esquema
        ax_schema.plot([0, 2], [1, 1], color='#35b779', linewidth=2.5)  # Entrada
        
        # Arm A (Arriba: BS1 -> M1 -> BS2)
        color_A = '#21918c'
        ax_schema.plot([2, 2], [1, 4], color=color_A, linewidth=2.5, label='Brazo A (Desfase φ)')
        ax_schema.plot([2, 6], [4, 4], color=color_A, linewidth=2.5)

        # Arm B (Abajo: BS1 -> M2 -> BS2)
        color_B = '#fde725' if opacity_val < 0.5 else '#d97706'
        ax_schema.plot([2, 6], [1, 1], color=color_B, linewidth=2.5, label='Brazo B')
        ax_schema.plot([6, 6], [1, 4], color=color_B, linewidth=2.5)

        # Salidas a Detectores D1 y D2
        ax_schema.plot([6, 8], [4, 4], color='#35b779', linewidth=2.0, linestyle='--')
        ax_schema.plot([6, 6], [4, 6], color='#fde725', linewidth=2.0, linestyle='--')

        # Componentes Ópticos (BS y Espejos)
        ax_schema.plot([1.7, 2.3], [0.7, 1.3], color='white', linewidth=3, label='BS1 (50:50)')
        ax_schema.plot([5.7, 6.3], [3.7, 4.3], color='white', linewidth=3, label='BS2 (50:50)')
        ax_schema.plot([1.7, 2.3], [4.3, 3.7], color='#94a3b8', linewidth=3) # M1
        ax_schema.plot([5.7, 6.3], [1.3, 0.7], color='#94a3b8', linewidth=3) # M2

        # Desfasador φ en Brazo A
        ax_schema.plot([4, 4], [3.7, 4.3], color='#21918c', linewidth=6)
        ax_schema.text(4, 4.5, f'φ = {phi_val:.2f} rad', color='#21918c', fontsize=9, ha='center', fontweight='bold')

        # Bloqueador / Detector en Brazo B (si opacity > 0)
        if opacity_val > 0.0:
            ax_schema.plot([4, 4], [0.7, 1.3], color='#ef4444', linewidth=8 * opacity_val)
            ax_schema.text(4, 0.3, f'Bloqueo: {opacity_val*100:.0f}%', color='#ef4444', fontsize=8.5, ha='center', fontweight='bold')

        # Detectores
        # D1: Puerto Brillante / Constructivo (a φ=0, P(D1)=100%)
        ax_schema.plot(8, 4, 's', color='#35b779', markersize=14, label='D1 (Puerto Brillante)')
        ax_schema.text(8.2, 4, f"D1\n{res['P_D1_real']*100:.0f}%", color='#35b779', fontsize=8.5, fontweight='bold', va='center')

        # D2: Puerto Oscuro / Destructivo (a φ=0, P(D2)=0%)
        ax_schema.plot(6, 6, 's', color='#fde725', markersize=14, label='D2 (Puerto Oscuro)')
        ax_schema.text(6, 6.4, f"D2: {res['P_D2_real']*100:.0f}% (Oscuro)", color='#fde725', fontsize=8.5, fontweight='bold', ha='center')

        ax_schema.set_xlim([-0.5, 9.8])
        ax_schema.set_ylim([-0.5, 7.2])
        ax_schema.axis('off')
        ax_schema.set_title('Esquema Óptico del Interferómetro de Mach-Zehnder', color='#35b779', fontsize=11, fontweight='bold')
        ax_schema.legend(loc='lower right', facecolor='#111827', edgecolor='#21918c', fontsize=7.5)

        # 2. Curvas de Interferencia P(D1) y P(D2) vs Fase φ
        ax_curves.clear()
        phi_grid = np.linspace(0, 4.0 * np.pi, 500)
        
        # Curvas teóricas exactas
        # P_D1 = 1/4 * (1 + (1-eta) + 2*sqrt(1-eta)*cos(phi))
        # P_D2 = 1/4 * (1 + (1-eta) - 2*sqrt(1-eta)*cos(phi))
        eta = opacity_val
        sqrt_1_eta = np.sqrt(max(0.0, 1.0 - eta))
        
        P_D1_grid = 0.25 * (1.0 + (1.0 - eta) + 2.0 * sqrt_1_eta * np.cos(phi_grid))
        P_D2_grid = 0.25 * (1.0 + (1.0 - eta) - 2.0 * sqrt_1_eta * np.cos(phi_grid))

        ax_curves.plot(phi_grid, P_D1_grid, color='#35b779', linewidth=2.2, label=r'$P(D_1) = \frac{1}{2}(1 + V\cos\phi)$ [Brillante]')
        ax_curves.plot(phi_grid, P_D2_grid, color='#fde725', linewidth=2.2, label=r'$P(D_2) = \frac{1}{2}(1 - V\cos\phi)$ [Oscuro]')
        
        ax_curves.axvline(phi_val, color='#e2e8f0', linestyle=':', linewidth=1.8, label=f'Fase Actual φ = {phi_val:.2f}')
        ax_curves.plot(phi_val, res['P_D1_real'], 'o', color='#35b779', markersize=8)
        ax_curves.plot(phi_val, res['P_D2_real'], 'o', color='#fde725', markersize=8)

        ax_curves.set_xlim([0, 4.0 * np.pi])
        ax_curves.set_ylim([-0.05, 1.1])
        ax_curves.set_xlabel('Desfase φ (radianes)', color='white', fontsize=9)
        ax_curves.set_ylabel('Probabilidad de Detección', color='white', fontsize=9)
        ax_curves.set_title('Franjas de Interferencia en Función del Desfase φ', color='#21918c', fontsize=11, fontweight='bold')
        ax_curves.grid(True, color='#222222', linestyle='--', alpha=0.5)
        ax_curves.legend(loc='upper right', facecolor='#111827', edgecolor='#21918c', fontsize=7.5)

        # 3. Histograma de Eventos Monte Carlo (D1, D2, Absorbed)
        ax_hist.clear()
        categories = ['D1 (Brillante)', 'D2 (Oscuro)', 'Absorbed / Blocked']
        counts = [res['count_D1'], res['count_D2'], res['count_block']]
        theor_counts = [res['P_D1_real'] * N_val, res['P_D2_real'] * N_val, res['P_block_real'] * N_val]

        x_pos = np.arange(len(categories))
        width = 0.35

        rects1 = ax_hist.bar(x_pos - width/2, counts, width, label='Eventos Monte Carlo', color='#21918c', alpha=0.85, edgecolor='white')
        rects2 = ax_hist.bar(x_pos + width/2, theor_counts, width, label='Predicción Teórica', color='#fde725', alpha=0.5, edgecolor='#fde725', linestyle='--')

        ax_hist.set_xticks(x_pos)
        ax_hist.set_xticklabels(categories, color='white', fontsize=9)
        ax_hist.set_ylabel('Conteo de Eventos', color='white', fontsize=9)
        ax_hist.set_title(f'Conteo de Detecciones Cuánticas (N = {N_val})', color='#fde725', fontsize=11, fontweight='bold')
        ax_hist.grid(True, color='#222222', linestyle='--', alpha=0.5, axis='y')
        ax_hist.legend(loc='upper right', facecolor='#111827', edgecolor='#21918c', fontsize=8)

        for rect in rects1:
            h = rect.get_height()
            ax_hist.text(rect.get_x() + rect.get_width()/2.0, h + N_val*0.01, f'{int(h)}', ha='center', va='bottom', color='white', fontsize=8.5, fontweight='bold')

        # 4. Panel de Métricas e Información Cuántica
        ax_info.clear()
        ax_info.axis('off')

        is_interaction_free = (opacity_val > 0.8) and (res['count_D2'] > 0) and (np.abs(phi_val % (2*np.pi)) < 0.2)

        info_text = (
            "=== MÉTRICAS Y COHERENCIA CUÁNTICA ===\n\n"
            f" • Desfase Óptico (φ)     : {phi_val:.2f} rad ({np.degrees(phi_val):.1f}°)\n"
            f" • Opacidad Brazo B       : {opacity_val*100:.1f}%\n"
            f" • Visibilidad de Franjas  : V = {res['visibility']:.4f}\n\n"
            "--- Probabilidades Teóricas ---\n"
            f" • P(D1) [Puerto Brillante / Constructivo] : {res['P_D1_real']*100:.2f}%\n"
            f" • P(D2) [Puerto Oscuro / Destructivo]    : {res['P_D2_real']*100:.2f}%\n"
            f" • P(Bloqueado / Medido en Brazo B)       : {res['P_block_real']*100:.2f}%\n\n"
            "--- Conteo Monte Carlo Real (N = " + str(N_val) + ") ---\n"
            f" • Detecciones D1 (Brillante) : {res['count_D1']}  ({res['count_D1']/N_val*100:.1f}%)\n"
            f" • Detecciones D2 (Oscuro)    : {res['count_D2']}  ({res['count_D2']/N_val*100:.1f}%)\n"
            f" • Clics Bloqueador en B     : {res['count_block']}  ({res['count_block']/N_val*100:.1f}%)\n\n"
            "--- Medición Libre de Interacción (Elitzur-Vaidman) ---\n"
            f" • Estado: {'¡DETECCIÓN LIBRE DE INTERACCIÓN ACTIVA EN DETECTOR OSCURO D2!' if is_interaction_free else 'Régimen Estándar de Interferencia'}\n"
        )

        ax_info.text(0.05, 0.95, info_text, color='#e2e8f0', fontsize=9.0, fontfamily='monospace',
                     verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.8', facecolor='#1f2937', edgecolor='#35b779', alpha=0.95))

    def animate(frame):
        if is_animating[0]:
            phi_next = s_phi.val + 0.05
            if phi_next > s_phi.valmax:
                phi_next = s_phi.valmin
            s_phi.set_val(phi_next)
        else:
            draw_frame(s_phi.val)

    def on_slider_change(val):
        if not is_animating[0]:
            draw_frame(s_phi.val)
            fig.canvas.draw_idle()

    s_phi.on_changed(on_slider_change)
    s_opacity.on_changed(on_slider_change)
    s_N.on_changed(on_slider_change)

    anim = FuncAnimation(fig, animate, interval=50, cache_frame_data=False)
    fig._anim = anim

    draw_frame(s_phi.val)
    plt.suptitle("LABORATORIO PYTHON 5: INTERFERÓMETRO DE MACH-ZEHNDER VIRTUAL (Hecht Cap. 9 - Viridis)",
                 color='white', fontsize=13, fontweight='bold', y=0.98)
    plt.show()


def export_static_sample(output_path="laboratorio5_mach_zehnder_viridis.png"):
    """Genera una imagen estática de prueba para verificar compatibilidad."""
    run_sympy_analysis()
    res = compute_mach_zehnder(0.0, 0.0, 1000)

    dir_name = os.path.dirname(os.path.abspath(output_path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8), facecolor='#0b0f19')
    ax = fig.add_subplot(1, 1, 1, facecolor='#111827')
    
    phi_grid = np.linspace(0, 4 * np.pi, 500)
    ax.plot(phi_grid, 0.5 * (1.0 + np.cos(phi_grid)), color='#35b779', linewidth=2.5, label='P(D1) [Brillante]')
    ax.plot(phi_grid, 0.5 * (1.0 - np.cos(phi_grid)), color='#fde725', linewidth=2.5, label='P(D2) [Oscuro]')
    ax.set_title("Interferómetro de Mach-Zehnder - Muestra Estática (Lab 5)", color='white')
    ax.legend()

    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] Muestra estática del Lab 5 generada en: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] != '--test':
        launch_interactive_simulation()
    else:
        export_static_sample()
