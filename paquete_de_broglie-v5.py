"""
================================================================================
LABORATORIO PYTHON 2: DE LAS ONDAS DE DE BROGLIE A UN PAQUETE DE ONDAS (V5)
================================================================================
Materia: Mecánica Cuántica
Entorno: Visual Studio Code / Python 3 (Windows / Linux / macOS)
Librerías: SymPy, NumPy, Matplotlib (Paleta 'viridis'), SciPy

Novedades en V5 (Solución de Interactividad en VS Code):
 1. Selección explícita del backend gráfico ('TkAgg') para garantizar una ventana
    GUI nativa independiente donde los clics y arrastres de ratón funcionen.
 2. Retención explícita de objetos de Matplotlib (Sliders, Buttons, Animation)
    en memoria para evitar que el Garbage Collector de Python anule los eventos.
 3. Paleta de colores 'viridis' optimizada para 3D y 2D.
 4. Respuesta instantánea al arrastrar los deslizadores N, k0, sigma_k, L y t.
================================================================================
"""

import sys
import os
import matplotlib

# Forzar backend interactivo TkAgg si está disponible para VS Code / Windows
try:
    matplotlib.use('TkAgg')
except Exception:
    pass

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Compatibilidad con integraciones numéricas en NumPy 2.0+
if hasattr(np, 'trapezoid'):
    trapz = np.trapezoid
else:
    from scipy.integrate import trapezoid as trapz


def run_sympy_analysis():
    """Derivación simbólica con SymPy para w(k), v_fase y v_grupo."""
    print("=" * 75)
    print(" 1. ANÁLISIS SIMBÓLICO CON SYMPY - LABORATORIO 2 DE MECÁNICA CUÁNTICA")
    print("=" * 75)
    
    x, t, k, k0, sig_k, hbar, m = sp.symbols('x t k k_0 sigma_k hbar m', real=True, positive=True)
    omega = hbar * k**2 / (2 * m)
    v_fase = omega / k
    v_grupo = sp.diff(omega, k)

    print("\n• Relación de Dispersión w(k):")
    sp.pprint(sp.Eq(sp.Symbol('omega(k)'), omega))
    print("\n• Velocidad de Fase v_f(k) = w / k:")
    sp.pprint(sp.Eq(sp.Symbol('v_f'), v_fase))
    print("\n• Velocidad de Grupo v_g(k) = dw / dk:")
    sp.pprint(sp.Eq(sp.Symbol('v_g'), v_grupo))

    print("\nEn unidades adimensionales (hbar=1, m=1) evaluadas en k_0:")
    print(f"  v_f(k_0) = {v_fase.subs({hbar:1, m:1, k:k0})}")
    print(f"  v_g(k_0) = {v_grupo.subs({hbar:1, m:1, k:k0})}  --> (Velocidad Clásica p_0/m)")
    print("=" * 75 + "\n")


def compute_wavepacket(N, k0, sig_k, t_val, L_val, num_points=800):
    """Calcula numéricamente la función de onda espacial compleja Ψ(x,t)."""
    x_grid = np.linspace(-L_val, L_val, num_points)

    if N == 1:
        k_vals = np.array([float(k0)])
        A_vals = np.array([1.0])
    else:
        k_vals = np.linspace(k0 - 3.5 * sig_k, k0 + 3.5 * sig_k, int(N))
        A_vals = np.exp(-((k_vals - k0) ** 2) / (2 * sig_k ** 2))

    A_norm = A_vals / np.sqrt(np.sum(A_vals**2)) if np.sum(A_vals**2) > 0 else A_vals
    omega_vals = 0.5 * k_vals**2

    phases = k_vals[:, np.newaxis] * x_grid[np.newaxis, :] - omega_vals[:, np.newaxis] * t_val
    components = A_norm[:, np.newaxis] * np.exp(1j * phases)
    psi = np.sum(components, axis=0)

    # Normalización espacial \int |Ψ|^2 dx = 1
    norm_factor = np.sqrt(trapz(np.abs(psi)**2, x_grid))
    if norm_factor > 0:
        psi = psi / norm_factor

    prob_density = np.abs(psi)**2

    # Promedios numéricos en x
    mean_x = trapz(x_grid * prob_density, x_grid)
    var_x = trapz(((x_grid - mean_x)**2) * prob_density, x_grid)
    delta_x = np.sqrt(max(var_x, 0))

    # Promedios numéricos en k
    if N > 1:
        weights_k = A_norm**2 / np.sum(A_norm**2)
        mean_k = np.sum(k_vals * weights_k)
        var_k = np.sum(((k_vals - mean_k)**2) * weights_k)
        delta_k = np.sqrt(max(var_k, 0))
    else:
        mean_k = k0
        delta_k = 0.0

    return {
        'x_grid': x_grid,
        'psi': psi,
        'prob_density': prob_density,
        'k_vals': k_vals,
        'A_norm': A_norm,
        'mean_x': mean_x,
        'delta_x': delta_x,
        'mean_k': mean_k,
        'delta_k': delta_k,
        'v_f': k0 / 2.0,
        'v_g': k0
    }


def launch_interactive_simulation():
    """Lanza la simulación interactiva con Matplotlib Viridis y soporte completo para ratón."""
    run_sympy_analysis()

    # Parámetros iniciales
    state = {
        'N': 40,
        'k0': 5.0,
        'sig_k': 0.8,
        'L': 25.0,
        't': 0.0,
        'running': True
    }

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 9.5), facecolor='#0b0f19')
    try:
        fig.canvas.manager.set_window_title("Laboratorio 2 Cuántica: Paquete de Broglie (Viridis Interactive)")
    except Exception:
        pass

    # Definir subplots con espacio libre abajo para controles
    plt.subplots_adjust(left=0.07, right=0.95, bottom=0.32, top=0.92, wspace=0.25, hspace=0.35)

    ax3d = fig.add_subplot(2, 2, 1, projection='3d', facecolor='#0e1424')
    ax2d = fig.add_subplot(2, 2, 2, facecolor='#0e1424')
    ax_k = fig.add_subplot(2, 2, 3, facecolor='#0e1424')
    ax_info = fig.add_subplot(2, 2, 4, facecolor='#0e1424')
    ax_info.axis('off')

    # Ejes para los controles Sliders y Botón
    ax_N    = fig.add_axes([0.12, 0.22, 0.32, 0.025], facecolor='#1a233a')
    ax_k0   = fig.add_axes([0.12, 0.17, 0.32, 0.025], facecolor='#1a233a')
    ax_sigk = fig.add_axes([0.12, 0.12, 0.32, 0.025], facecolor='#1a233a')
    ax_L    = fig.add_axes([0.12, 0.07, 0.32, 0.025], facecolor='#1a233a')
    ax_t    = fig.add_axes([0.12, 0.02, 0.32, 0.025], facecolor='#1a233a')

    ax_btn  = fig.add_axes([0.48, 0.14, 0.08, 0.06], facecolor='#21918c')

    # Instanciación de Sliders y Botones
    s_N    = Slider(ax_N,    'N (Num Ondas)', 1, 100, valinit=state['N'], valstep=1, color='#21918c')
    s_k0   = Slider(ax_k0,   'k₀ (Onda Central)', 1.0, 10.0, valinit=state['k0'], color='#35b779')
    s_sigk = Slider(ax_sigk, 'σₖ (Sigma k)', 0.1, 2.5, valinit=state['sig_k'], color='#90d743')
    s_L    = Slider(ax_L,    'Ancho L ([-L,L])', 10.0, 60.0, valinit=state['L'], color='#fde725')
    s_t    = Slider(ax_t,    'Tiempo t (s)', 0.0, 10.0, valinit=state['t'], color='#440154')

    btn_play = Button(ax_btn, '❚❚ Pausa', color='#21918c', hovercolor='#35b779')

    # ALMACENAR REFERENCIAS EN FIG PARA EVITAR EL GARBAGE COLLECTOR
    fig._widgets = [s_N, s_k0, s_sigk, s_L, s_t, btn_play]

    def render_all():
        """Función principal de renderizado de la escena."""
        N_val = int(s_N.val)
        k0_val = s_k0.val
        sig_k_val = s_sigk.val
        L_val = s_L.val
        t_val = s_t.val

        res = compute_wavepacket(N_val, k0_val, sig_k_val, t_val, L_val)
        x_grid = res['x_grid']
        psi = res['psi']
        prob = res['prob_density']
        re_psi = np.real(psi)
        im_psi = np.imag(psi)

        # 1. Gráfico 3D Complejo con mapa 'viridis'
        ax3d.clear()
        points = np.array([x_grid, re_psi, im_psi]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = Normalize(vmin=0, vmax=max(np.max(prob), 1e-5))
        cmap = plt.get_cmap('viridis')
        lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidth=2.0)
        lc.set_array(prob[:-1])
        ax3d.add_collection(lc)

        max_amp = max(np.max(np.abs(psi))*1.2, 0.25)
        ax3d.set_xlim([-L_val, L_val])
        ax3d.set_ylim([-max_amp, max_amp])
        ax3d.set_zlim([-max_amp, max_amp])
        ax3d.set_xlabel('x', color='white', fontsize=8)
        ax3d.set_ylabel('Re[Ψ]', color='#35b779', fontsize=8)
        ax3d.set_zlabel('Im[Ψ]', color='#fde725', fontsize=8)
        ax3d.set_title(f'Función de Onda 3D (x, Re, Im) - Viridis\nt = {t_val:.2f} s', color='#fde725', fontsize=10, fontweight='bold')

        # 2. Densidad de Probabilidad 2D Espacial
        ax2d.clear()
        ax2d.plot(x_grid, prob, color='#fde725', linewidth=2.2, label=r'Densidad $|\Psi(x,t)|^2$')
        ax2d.fill_between(x_grid, 0, prob, color='#21918c', alpha=0.4)
        ax2d.plot(x_grid, re_psi, color='#440154', linestyle='--', alpha=0.6, label=r'Parte Real Re$[\Psi]$')
        
        xc = res['v_g'] * t_val
        ax2d.axvline(xc, color='#90d743', linestyle=':', linewidth=2.0, label=f'Centro $x_c = v_g t$ ({xc:.2f})')
        
        ax2d.set_xlim([-L_val, L_val])
        ax2d.set_ylim([0, max(np.max(prob)*1.15, 0.1)])
        ax2d.set_xlabel('Posición x', color='white', fontsize=9)
        ax2d.set_ylabel('Densidad de Probabilidad', color='white', fontsize=9)
        ax2d.set_title(f'Densidad Espacial |Ψ(x,t)|² (N = {N_val})', color='#fde725', fontsize=10, fontweight='bold')
        ax2d.grid(True, color='#1f293d', linestyle='--', alpha=0.6)
        ax2d.legend(loc='upper right', facecolor='#141b2d', edgecolor='#35b779', fontsize=8)

        # 3. Espacio de Momentos k
        ax_k.clear()
        ax_k.stem(res['k_vals'], res['A_norm'], linefmt='#35b779', markerfmt='yo', basefmt='gray')
        ax_k.set_xlabel('Número de Onda k', color='white', fontsize=9)
        ax_k.set_ylabel('Amplitud A(k)', color='white', fontsize=9)
        ax_k.set_title('Espectro de Amplitudes en Espacio-k', color='#fde725', fontsize=10, fontweight='bold')
        ax_k.grid(True, color='#1f293d', linestyle='--', alpha=0.6)

        # 4. Panel de Métricas e Información
        ax_info.clear()
        ax_info.axis('off')
        info_str = (
            "=== PARÁMETROS Y MÉTRICAS EN TIEMPO REAL ===\n\n"
            f" • Número de Ondas (N): {N_val}\n"
            f" • Onda Central (k₀): {k0_val:.2f} rad/u\n"
            f" • Ancho Espectral (σₖ): {sig_k_val:.2f} rad/u\n"
            f" • Dominio Espacial x: [-{L_val:.1f}, {L_val:.1f}]\n"
            f" • Tiempo Actual (t): {t_val:.2f} s\n\n"
            "--- Cinemática de De Broglie ---\n"
            f" • Velocidad de Fase  (v_f = ω/k₀) : {res['v_f']:.3f} u/s\n"
            f" • Velocidad de Grupo (v_g = dw/dk): {res['v_g']:.3f} u/s\n"
            f" • Posición media actual <x>       : {res['mean_x']:.3f} u\n\n"
            "--- Relación de Indeterminación ---\n"
            f" • Ancho Espacial  Δx: {res['delta_x']:.4f}\n"
            f" • Ancho Espectral Δk: {res['delta_k']:.4f}\n"
            f" • Producto Δx · Δk  : {res['delta_x'] * res['delta_k']:.4f}  (Límite ≥ 0.500)\n"
        )
        ax_info.text(0.05, 0.95, info_str, color='#e2e8f0', fontsize=9, fontfamily='monospace',
                     verticalalignment='top', bbox=dict(boxstyle='round,pad=0.7', facecolor='#141b2d', edgecolor='#35b779'))

    def update_sliders(val):
        """Callback al mover cualquier deslizador."""
        render_all()
        fig.canvas.draw_idle()

    def toggle_play(event):
        """Callback al presionar el botón de Play/Pausa."""
        state['running'] = not state['running']
        btn_play.label.set_text('❚❚ Pausa' if state['running'] else '▶ Reanudar')
        fig.canvas.draw_idle()

    s_N.on_changed(update_sliders)
    s_k0.on_changed(update_sliders)
    s_sigk.on_changed(update_sliders)
    s_L.on_changed(update_sliders)
    s_t.on_changed(update_sliders)
    btn_play.on_clicked(toggle_play)

    def update_animation(frame):
        """Callback del bucle de animación."""
        if state['running']:
            t_next = (s_t.val + 0.04) % 10.0
            # Actualiza el slider sin disparar eventos recursivos
            s_t.eventson = False
            s_t.set_val(t_next)
            s_t.eventson = True
            render_all()

    anim = FuncAnimation(fig, update_animation, interval=40, cache_frame_data=False)
    fig._anim = anim  # Mantener referencia a la animación

    # Primer dibujado inicial
    render_all()

    plt.suptitle("Laboratorio Python 2: Simulación Animada e Interactiva de De Broglie (Paleta Viridis)",
                 color='white', fontsize=12, fontweight='bold', y=0.97)
    
    plt.show()


if __name__ == '__main__':
    launch_interactive_simulation()
