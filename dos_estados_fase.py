"""
================================================================================
LABORATORIO PYTHON 3: DOS ESTADOS, UNA SUPERPOSICIÓN Y LA HUELLA DE LA FASE
================================================================================
Materia: Mecánica Cuántica
Entorno: Visual Studio Code / Python 3
Librerías: SymPy, NumPy, Matplotlib (Paleta 'viridis'), SciPy

Descripción:
Simulación animada e interactiva en tiempo real que resuelve el pozo infinito
unidimensional, construye la superposición coherente de los dos primeros autoestados:
    Ψ(x,t) = exp(i χ) [ c1 φ1(x) exp(-i E1 t / ℏ) + c2 exp(i δ) φ2(x) exp(-i E2 t / ℏ) ]
con c1 = cos(θ), c2 = sin(θ).

Demuestra:
 1. Análisis simbólico previo con SymPy del pozo, energías En y término cruzado I12.
 2. Estructura tridimensional 3D de la hélice compleja (x, Re[Ψ], Im[Ψ]) coloreada con 'viridis'.
 3. Comparación entre la densidad total |Ψ(x,t)|² y la suma sin interferencia (|c1|²φ1² + |c2|²φ2²).
 4. Respuesta temporal de un detector virtual en posición xD (demostrando la frecuencia de BOHR / Rabi Ω21 = (E2-E1)/ℏ).
 5. Diferencia entre fase global (χ) e inobservabilidad vs. fase relativa (δ) e interferencia observable.
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
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Compatibilidad con integraciones numéricas en NumPy 2.0+ / SciPy
if hasattr(np, 'trapezoid'):
    trapz = np.trapezoid
else:
    from scipy.integrate import trapezoid as trapz

# Forzado de Backend interactivo para ejecuciones en terminal / VS Code
if len(sys.argv) == 1 or sys.argv[1] != '--test':
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass


def run_sympy_analysis():
    """
    Derivación analítica con SymPy de los autoestados del pozo infinito,
    la energía de Bohr y el término de interferencia I12(x,t).
    """
    print("=" * 80)
    print(" 1. ANÁLISIS SIMBÓLICO CON SYMPY - LABORATORIO 3: DOS ESTADOS Y FASE")
    print("=" * 80)

    x, t, L, hbar, m = sp.symbols('x t L hbar m', real=True, positive=True)
    n = sp.Symbol('n', integer=True, positive=True)
    theta, delta = sp.symbols('theta delta', real=True)

    # 1. Autofunciones y Energías
    phi_n = sp.sqrt(2/L) * sp.sin(n * sp.pi * x / L)
    E_n = (n**2 * sp.pi**2 * hbar**2) / (2 * m * L**2)

    print("\n• Autoestado del pozo infinito phi_n(x):")
    sp.pprint(phi_n)

    print("\n• Autovalor de Energía E_n:")
    sp.pprint(sp.Eq(sp.Symbol('E_n'), E_n))

    # 2. Transición entre E1 y E2
    E1 = E_n.subs(n, 1)
    E2 = E_n.subs(n, 2)
    delta_E = E2 - E1
    omega_21 = delta_E / hbar

    print("\n• Diferencia de Energía Delta E = E2 - E1:")
    sp.pprint(sp.Eq(sp.Symbol('Delta_E'), delta_E))

    print("\n• Frecuencia de Frecuencia de Fases / Transición Omega_21 = Delta_E / hbar:")
    sp.pprint(sp.Eq(sp.Symbol('Omega_21'), omega_21))

    # 3. Término de Interferencia I_12
    c1 = sp.cos(theta)
    c2 = sp.sin(theta)
    phi1 = phi_n.subs(n, 1)
    phi2 = phi_n.subs(n, 2)

    I_12 = 2 * c1 * c2 * phi1 * phi2 * sp.cos(delta - omega_21 * t)
    print("\n• Término de Interferencia Cruzado I_12(x,t):")
    sp.pprint(sp.Eq(sp.Symbol('I_12(x,t)'), I_12))

    print("\nEn unidades adimensionales (hbar=1, m=1, L=1):")
    E1_num = float((1**2 * np.pi**2) / 2.0)
    E2_num = float((2**2 * np.pi**2) / 2.0)
    dE_num = E2_num - E1_num
    T_num = 2 * np.pi / dE_num

    print(f"  • E1 = pi^2 / 2 = {E1_num:.5f}")
    print(f"  • E2 = 2*pi^2   = {E2_num:.5f}")
    print(f"  • Delta E = 1.5*pi^2 = {dE_num:.5f}")
    print(f"  • Período de Oscilación Teórico T_21 = 2*pi / Delta E = {T_num:.5f} s")
    print("=" * 80 + "\n")


def compute_two_state_system(theta_val, delta_val, chi_val, t_val, xD_val, L_val=1.0, num_x=600):
    """
    Calcula la función de onda total, la densidad espacial y la señal del detector.
    Unidades adimensionales: hbar = 1, m = 1, L = 1.
    """
    x_grid = np.linspace(0, L_val, num_x)

    # Autoestados analíticos
    phi1 = np.sqrt(2.0 / L_val) * np.sin(np.pi * x_grid / L_val)
    phi2 = np.sqrt(2.0 / L_val) * np.sin(2.0 * np.pi * x_grid / L_val)

    # Energías E1 y E2
    E1 = (np.pi**2) / (2.0 * L_val**2)
    E2 = (4.0 * np.pi**2) / (2.0 * L_val**2)
    delta_E = E2 - E1
    omega_21 = delta_E
    T_21 = 2.0 * np.pi / omega_21

    # Coeficientes
    c1 = np.cos(theta_val)
    c2 = np.sin(theta_val)

    # Evolución temporal
    psi1_t = c1 * phi1 * np.exp(-1j * E1 * t_val)
    psi2_t = c2 * np.exp(1j * delta_val) * phi2 * np.exp(-1j * E2 * t_val)

    # Fase global chi
    psi_total = np.exp(1j * chi_val) * (psi1_t + psi2_t)

    # Densidad de probabilidad
    prob_density = np.abs(psi_total)**2

    # Densidad sin interferencia
    rho_no_interf = (c1**2) * (phi1**2) + (c2**2) * (phi2**2)

    # Término cruzado de interferencia
    I_12 = prob_density - rho_no_interf

    # Comprobaciones de Ortogonalidad y Normalización
    norm_val = trapz(prob_density, x_grid)
    ortho_val = trapz(phi1 * phi2, x_grid)

    # ---------------------------------------------------------
    # Generación de la señal del detector I_D(t') a lo largo del tiempo
    # ---------------------------------------------------------
    t_signal_grid = np.linspace(0, 2.5 * T_21, 500)
    phi1_xD = np.sqrt(2.0 / L_val) * np.sin(np.pi * xD_val / L_val)
    phi2_xD = np.sqrt(2.0 / L_val) * np.sin(2.0 * np.pi * xD_val / L_val)

    psi1_xD = c1 * phi1_xD * np.exp(-1j * E1 * t_signal_grid)
    psi2_xD = c2 * np.exp(1j * delta_val) * phi2_xD * np.exp(-1j * E2 * t_signal_grid)
    signal_xD = np.abs(np.exp(1j * chi_val) * (psi1_xD + psi2_xD))**2

    # Señal en el tiempo t_val actual
    current_signal_xD = np.abs(np.exp(1j * chi_val) * (
        c1 * phi1_xD * np.exp(-1j * E1 * t_val) +
        c2 * np.exp(1j * delta_val) * phi2_xD * np.exp(-1j * E2 * t_val)
    ))**2

    return {
        'x_grid': x_grid,
        'phi1': phi1,
        'phi2': phi2,
        'psi_total': psi_total,
        'prob_density': prob_density,
        'rho_no_interf': rho_no_interf,
        'I_12': I_12,
        'E1': E1,
        'E2': E2,
        'delta_E': delta_E,
        'omega_21': omega_21,
        'T_21': T_21,
        'norm_val': norm_val,
        'ortho_val': ortho_val,
        'c1': c1,
        'c2': c2,
        't_signal_grid': t_signal_grid,
        'signal_xD': signal_xD,
        'current_signal_xD': current_signal_xD
    }


def launch_interactive_simulation():
    """Simulación interactiva y animada completa para VS Code / Tkinter."""
    run_sympy_analysis()

    # Parámetros iniciales
    theta_init = np.pi / 4.0      # Superposición equilibrada c1^2 = c2^2 = 0.5
    delta_init = 0.0              # Fase relativa inicial
    chi_init = 0.0                # Fase global inicial
    xD_init = 0.333               # Posición del detector (L/3)
    t_init = 0.0

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 9.8), facecolor='#0b0f19')

    ax3d = fig.add_subplot(2, 2, 1, projection='3d', facecolor='#111827')
    ax2d = fig.add_subplot(2, 2, 2, facecolor='#111827')
    ax_det = fig.add_subplot(2, 2, 3, facecolor='#111827')
    ax_info = fig.add_subplot(2, 2, 4, facecolor='#111827')
    ax_info.axis('off')

    plt.subplots_adjust(left=0.07, right=0.95, top=0.93, bottom=0.28, wspace=0.25, hspace=0.34)

    # Sliders y Botones
    ax_theta = fig.add_axes([0.12, 0.19, 0.32, 0.022], facecolor='#1f2937')
    ax_delta = fig.add_axes([0.12, 0.14, 0.32, 0.022], facecolor='#1f2937')
    ax_chi   = fig.add_axes([0.12, 0.09, 0.32, 0.022], facecolor='#1f2937')

    ax_xD    = fig.add_axes([0.58, 0.19, 0.32, 0.022], facecolor='#1f2937')
    ax_t     = fig.add_axes([0.58, 0.14, 0.32, 0.022], facecolor='#1f2937')

    ax_btn   = fig.add_axes([0.58, 0.06, 0.12, 0.045], facecolor='#21918c')

    s_theta = Slider(ax_theta, 'Mezcla θ', 0.0, np.pi/2.0, valinit=theta_init, color='#21918c')
    s_delta = Slider(ax_delta, 'Fase Relat. δ', 0.0, 2.0*np.pi, valinit=delta_init, color='#35b779')
    s_chi   = Slider(ax_chi,   'Fase Glob. χ', 0.0, 2.0*np.pi, valinit=chi_init, color='#31688e')

    s_xD    = Slider(ax_xD,    'Detector x_D', 0.05, 0.95, valinit=xD_init, color='#fde725')
    s_t     = Slider(ax_t,     'Tiempo t (s)', 0.0, 2.0, valinit=t_init, color='#d97706')

    btn_play = Button(ax_btn, '▶ Animación', color='#21918c', hovercolor='#35b779')

    fig._widgets = [s_theta, s_delta, s_chi, s_xD, s_t, btn_play]
    is_animating = [False]

    def toggle_play(event):
        is_animating[0] = not is_animating[0]
        btn_play.label.set_text('❚❚ Pausa' if is_animating[0] else '▶ Animación')
        btn_play.ax.set_facecolor('#fde725' if is_animating[0] else '#21918c')
        fig.canvas.draw_idle()

    btn_play.on_clicked(toggle_play)

    def draw_frame(t_val):
        theta_val = s_theta.val
        delta_val = s_delta.val
        chi_val   = s_chi.val
        xD_val    = s_xD.val

        res = compute_two_state_system(theta_val, delta_val, chi_val, t_val, xD_val)
        x_grid = res['x_grid']
        psi = res['psi_total']
        prob = res['prob_density']
        re_psi = np.real(psi)
        im_psi = np.imag(psi)

        # 1. Gráfico 3D Hélice Compleja
        ax3d.clear()
        points = np.array([x_grid, re_psi, im_psi]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = Normalize(vmin=0, vmax=max(np.max(prob), 1e-4))
        cmap = plt.get_cmap('viridis')

        lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidth=2.2)
        lc.set_array(prob[:-1])
        ax3d.add_collection(lc)

        max_amp = 1.8
        ax3d.set_xlim([0, 1.0])
        ax3d.set_ylim([-max_amp, max_amp])
        ax3d.set_zlim([-max_amp, max_amp])
        ax3d.set_xlabel('x (Pozo)', color='white', fontsize=9)
        ax3d.set_ylabel('Re[Ψ]', color='#35b779', fontsize=9)
        ax3d.set_zlabel('Im[Ψ]', color='#fde725', fontsize=9)
        ax3d.set_title(f'Hélice Compleja 3D (x, Re, Im)\\nt = {t_val:.3f} s', color='#35b779', fontsize=11, fontweight='bold')

        # 2. Densidad Espacial 2D
        ax2d.clear()
        ax2d.plot(x_grid, prob, color='#fde725', linewidth=2.5, label=r'Densidad Total $|\Psi(x,t)|^2$')
        ax2d.fill_between(x_grid, 0, prob, color='#fde725', alpha=0.25)

        ax2d.plot(x_grid, res['rho_no_interf'], color='gray', linestyle='--', linewidth=1.5,
                   label=r'Sin Interferencia $|c_1|^2\phi_1^2 + |c_2|^2\phi_2^2$')
        
        ax2d.plot(x_grid, res['phi1'], color='#21918c', linestyle=':', alpha=0.6, label=r'Autoestado $\phi_1(x)$')
        ax2d.plot(x_grid, res['phi2'], color='#35b779', linestyle=':', alpha=0.6, label=r'Autoestado $\phi_2(x)$')

        ax2d.axvline(xD_val, color='#d97706', linestyle='--', linewidth=1.8, label=f'Detector $x_D = {xD_val:.2f}$')

        ax2d.set_xlim([0, 1.0])
        ax2d.set_ylim([-1.6, max(np.max(prob)*1.2, 2.5)])
        ax2d.set_xlabel('Posición x / L', color='white', fontsize=9)
        ax2d.set_ylabel('Amplitud / Densidad', color='white', fontsize=9)
        ax2d.set_title('Densidad de Probabilidad e Interferencia 2D', color='#21918c', fontsize=11, fontweight='bold')
        ax2d.grid(True, color='#222222', linestyle='--', alpha=0.5)
        ax2d.legend(loc='upper right', facecolor='#111827', edgecolor='#21918c', fontsize=7.5)

        # 3. Detector Virtual en Tiempo Real
        ax_det.clear()
        ax_det.plot(res['t_signal_grid'], res['signal_xD'], color='#35b779', linewidth=2.2, label=r'Señal $I_D(t) = |\Psi(x_D,t)|^2$')
        ax_det.plot(t_val, res['current_signal_xD'], 'o', color='#fde725', markersize=8, label=f't = {t_val:.3f} s')
        ax_det.axvline(t_val, color='#fde725', linestyle=':', linewidth=1.5)

        # Marcar periodo Teórico T_21
        ax_det.axvline(res['T_21'], color='#d97706', linestyle='--', alpha=0.8, label=f"Período $T_{{21}} = {res['T_21']:.3f}$ s")

        ax_det.set_xlim([0, max(res['t_signal_grid'])])
        ax_det.set_ylim([0, max(np.max(res['signal_xD'])*1.2, 0.5)])
        ax_det.set_xlabel('Tiempo t (s)', color='white', fontsize=9)
        ax_det.set_ylabel('Intensidad en Detector $I_D$', color='white', fontsize=9)
        ax_det.set_title(f'Señal del Detector Virtual en $x_D = {xD_val:.2f}$', color='#fde725', fontsize=11, fontweight='bold')
        ax_det.grid(True, color='#222222', linestyle='--', alpha=0.5)
        ax_det.legend(loc='upper right', facecolor='#111827', edgecolor='#35b779', fontsize=7.5)

        # 4. Métricas e Información
        ax_info.clear()
        ax_info.axis('off')

        info_text = (
            "=== ESTADO CUÁNTICO Y MÉTRICAS ===\n\n"
            f" • Coeficiente c₁ = cos({theta_val:.2f}) = {res['c1']:.3f}  --> |c₁|² = {res['c1']**2:.3f}\n"
            f" • Coeficiente c₂ = sin({theta_val:.2f}) = {res['c2']:.3f}  --> |c₂|² = {res['c2']**2:.3f}\n"
            f" • Fase Relativa (δ)  : {delta_val:.2f} rad ({np.degrees(delta_val):.1f}°)\n"
            f" • Fase Global   (χ)  : {chi_val:.2f} rad ({np.degrees(chi_val):.1f}°)\n\n"
            "--- Espectro y Dinámica de Fases ---\n"
            f" • Energía E₁ = π²/2      : {res['E1']:.4f} u.a.\n"
            f" • Energía E₂ = 2*π²     : {res['E2']:.4f} u.a.\n"
            f" • Frecuencia Ω₂₁ (ΔE/ℏ) : {res['omega_21']:.4f} rad/s\n"
            f" • Período Teórico T₂₁    : {res['T_21']:.4f} s\n\n"
            "--- Verificaciones del Espacio de Hilbert ---\n"
            f" • Normalización  <Ψ|Ψ>  : {res['norm_val']:.5f}  (Debe ser 1.0000)\n"
            f" • Ortogonalidad <φ₁|φ₂> : {res['ortho_val']:.5e} (Debe ser 0.0000)\n"
            f" • Intensidad en x_D actual: {res['current_signal_xD']:.4f}\n"
        )
        ax_info.text(0.05, 0.95, info_text, color='#e2e8f0', fontsize=9.2, fontfamily='monospace',
                     verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.8', facecolor='#1f2937', edgecolor='#21918c', alpha=0.95))

    def animate(frame):
        if is_animating[0]:
            t_next = (s_t.val + 0.008)
            if t_next > s_t.valmax:
                t_next = s_t.valmin
            s_t.set_val(t_next)
        else:
            draw_frame(s_t.val)

    def on_slider_change(val):
        if not is_animating[0]:
            draw_frame(s_t.val)
            fig.canvas.draw_idle()

    s_theta.on_changed(on_slider_change)
    s_delta.on_changed(on_slider_change)
    s_chi.on_changed(on_slider_change)
    s_xD.on_changed(on_slider_change)
    s_t.on_changed(on_slider_change)

    anim = FuncAnimation(fig, animate, interval=40, cache_frame_data=False)
    fig._anim = anim

    draw_frame(s_t.val)
    plt.suptitle("LABORATORIO PYTHON 3: DOS ESTADOS, UNA SUPERPOSICIÓN Y LA HUELLA DE LA FASE (VS Code - Viridis)",
                 color='white', fontsize=13, fontweight='bold', y=0.98)
    plt.show()


def export_static_sample(output_path="laboratorio3_dos_estados_viridis.png"):
    """Genera una imagen estática de prueba para verificar compatibilidad."""
    run_sympy_analysis()
    res = compute_two_state_system(np.pi/4.0, 0.0, 0.0, 0.1, 0.333)

    dir_name = os.path.dirname(os.path.abspath(output_path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8), facecolor='#0b0f19')
    ax2d = fig.add_subplot(1, 1, 1, facecolor='#111827')
    ax2d.plot(res['x_grid'], res['prob_density'], color='#fde725', linewidth=2.5, label=r'$|\Psi(x,t)|^2$')
    ax2d.set_title("Muestra Estática del Laboratorio 3", color='white')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] Muestra estática del Lab 3 generada en: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] != '--test':
        launch_interactive_simulation()
    else:
        export_static_sample()
