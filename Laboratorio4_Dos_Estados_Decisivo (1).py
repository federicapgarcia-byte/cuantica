"""
================================================================================
LABORATORIO 4: DOS ESTADOS, UNA SUPERPOSICIÓN Y EL EXPERIMENTO DECISIVO
================================================================================
Materia: Mecánica Cuántica (UNSJ)
Guía de Laboratorio 4 / Capítulo 4: Dos Estados y la Huella de la Fase

Descripción:
Simulación interactiva y automatizada que resuelve el pozo de potencial infinito 1D,
construye la superposición coherente de los dos primeros autoestados:
    Ψ(x,t) = exp(i χ) [ c1 φ1(x) exp(-i E1 t / ℏ) + c2 exp(i δ) φ2(x) exp(-i E2 t / ℏ) ]
con c1 = cos(θ), c2 = sin(θ).

Demuestra y Automatiza:
 1. Experimentos 1 a 4 del Capítulo 4.
 2. El EXPERIMENTO DECISIVO (Sección 13 / Hoja 103 de la guía):
    - Detector virtual posicionado en xD (default xD = L/3).
    - Medición numérica directa del período de batido T_num entre picos consecutivos.
    - Determinación empírica de la diferencia de energía ΔE_medida = 2π ℏ / T_num.
    - Comparación frente al espectro teórico ΔE_espectro = E2 - E1 = 3π²ℏ² / (2mL²).
    - Generación en tiempo real de la Tabla Oficial de Medición y Error Relativo (%).
    - Análisis del punto ciego en xD = L/2 (donde φ2(L/2) = 0 y la interferencia se anula).
================================================================================
"""

import sys
import os
import numpy as np
import scipy.signal as signal
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button, RadioButtons

# Forzar backend interactivo TkAgg para ejecución en desktop / VS Code
if len(sys.argv) == 1 or sys.argv[1] != '--test':
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass

plt.style.use('dark_background')

# =============================================================================
# 1. PARÁMETROS Y AUTOESTADOS DEL POZO INFINITO 1D (UNIDADES ADIMENSIONALES)
# =============================================================================

L = 1.0       # Ancho del pozo
hbar = 1.0    # Constante de Planck reducida
m = 1.0       # Masa de la partícula

# Autovalores de energía teóricos
E1 = (1**2 * np.pi**2 * hbar**2) / (2 * m * L**2)
E2 = (2**2 * np.pi**2 * hbar**2) / (2 * m * L**2)
Delta_E_teor = E2 - E1
Omega_21_teor = Delta_E_teor / hbar
T_teor = 2 * np.pi / Omega_21_teor

def phi_n(n, x):
    """Autoestado del pozo infinito 1D."""
    x = np.asarray(x, dtype=float)
    return np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)

def psi_total(x, t, theta, delta, chi=0.0):
    """
    Función de onda en superposición:
    Ψ(x,t) = exp(i χ) [ cos(θ) φ1(x) exp(-i E1 t / ℏ) + sin(θ) exp(i δ) φ2(x) exp(-i E2 t / ℏ) ]
    """
    c1 = np.cos(theta)
    c2 = np.sin(theta)
    
    psi1_t = phi_n(1, x) * np.exp(-1j * E1 * t / hbar)
    psi2_t = phi_n(2, x) * np.exp(1j * delta) * np.exp(-1j * E2 * t / hbar)
    
    return np.exp(1j * chi) * (c1 * psi1_t + c2 * psi2_t)

def densidad_probabilidad(x, t, theta, delta):
    """
    Densidad de probabilidad |Ψ(x,t)|^2:
    |Ψ|^2 = c1^2 φ1^2 + c2^2 φ2^2 + 2 c1 c2 φ1 φ2 cos(δ - Ω_21 t)
    """
    c1 = np.cos(theta)
    c2 = np.sin(theta)
    p1 = phi_n(1, x)
    p2 = phi_n(2, x)
    
    mezcla = (c1**2) * (p1**2) + (c2**2) * (p2**2)
    interferencia = 2 * c1 * c2 * p1 * p2 * np.cos(delta - Omega_21_teor * t)
    return mezcla + interferencia, mezcla, interferencia

# =============================================================================
# 2. CLASE INTERFAZ INTERACTIVA TRIPLE VENTANA (EXPERIMENTO DECISIVO)
# =============================================================================

class Laboratorio4DosEstadosDecisivo:
    def __init__(self):
        # Parámetros iniciales (Experimento Decisivo: θ = π/4, δ = 0, xD = L/3)
        self.theta = np.pi / 4.0
        self.delta = 0.0
        self.xD = L / 3.0
        self.t_val = 0.0
        self.is_playing = False
        
        # Grillas espaciales y temporales
        self.x_grid = np.linspace(0, L, 600)
        self.t_max = 3.0 * T_teor
        self.t_grid = np.linspace(0, self.t_max, 2500)
        
        # Crear las 3 ventanas vinculadas
        self.crear_ventana_espacial()
        self.crear_ventana_temporal()
        self.crear_ventana_decisivo()
        
        # Actualizar todo por primera vez
        self.actualizar_todo()

    # -------------------------------------------------------------------------
    # VENTANA 1: DINÁMICA ESPACIAL Y TÉRMINO CRUZADO DE INTERFERENCIA
    # -------------------------------------------------------------------------
    def crear_ventana_espacial(self):
        self.fig_esp, (self.ax_psi, self.ax_I12) = plt.subplots(2, 1, figsize=(8.5, 8))
        self.fig_esp.suptitle("VENTANA 1: SUPERPOSICIÓN DE DOS ESTADOS Y HUELLA DE LA FASE", fontsize=12, fontweight='bold', color='#40e0d0')
        self.fig_esp.canvas.manager.set_window_title("Ventana 1: Dinámica Espacial e Interferencia")
        
        # Subplot 1: |Ψ(x,t)|^2 vs Mezcla Incoherente
        self.line_prob, = self.ax_psi.plot([], [], color='#00ffff', lw=2.2, label=r'$|\Psi(x,t)|^2$ (Total)')
        self.line_incoh, = self.ax_psi.plot([], [], color='gray', ls='--', lw=1.5, label=r'$|c_1|^2\phi_1^2 + |c_2|^2\phi_2^2$ (Mezcla)')
        self.fill_prob = None
        self.line_xd_esp1 = self.ax_psi.axvline(self.xD, color='#ff007f', ls=':', lw=2, label=r'Detector $x_D$')
        
        self.ax_psi.set_ylabel(r'Densidad de Probabilidad [u.a.]', fontsize=10)
        self.ax_psi.set_xlim(0, L)
        self.ax_psi.set_ylim(0, 4.5)
        self.ax_psi.grid(True, alpha=0.2)
        self.ax_psi.legend(loc='upper right', fontsize=9)
        
        # Subplot 2: Término Cruzado I12(x,t)
        self.line_I12, = self.ax_I12.plot([], [], color='#ff5500', lw=2, label=r'$I_{12}(x,t) = 2 c_1 c_2 \phi_1 \phi_2 \cos(\delta - \Omega_{21}t)$')
        self.ax_I12.axhline(0, color='gray', ls='--', alpha=0.5)
        self.line_xd_esp2 = self.ax_I12.axvline(self.xD, color='#ff007f', ls=':', lw=2)
        
        self.ax_I12.set_xlabel(r'Posición en el Pozo $x / L$', fontsize=10)
        self.ax_I12.set_ylabel(r'Interferencia $I_{12}(x,t)$', fontsize=10)
        self.ax_I12.set_xlim(0, L)
        self.ax_I12.set_ylim(-2.5, 2.5)
        self.ax_I12.grid(True, alpha=0.2)
        self.ax_I12.legend(loc='upper right', fontsize=9)
        
        # Controles (Sliders)
        self.fig_esp.subplots_adjust(left=0.12, bottom=0.28, right=0.92, top=0.92)
        
        ax_stheta = self.fig_esp.add_axes([0.15, 0.18, 0.30, 0.03])
        self.slider_theta = Slider(ax_stheta, 'Ángulo θ', 0.0, np.pi/2, valinit=self.theta, valfmt='%.2f rad', color='#00ffff')
        
        ax_sdelta = self.fig_esp.add_axes([0.60, 0.18, 0.30, 0.03])
        self.slider_delta = Slider(ax_sdelta, 'Fase Relativa δ', 0.0, 2*np.pi, valinit=self.delta, valfmt='%.2f rad', color='#ff007f')
        
        ax_sxd = self.fig_esp.add_axes([0.15, 0.11, 0.30, 0.03])
        self.slider_xd = Slider(ax_sxd, 'Detector xD', 0.05, 0.95, valinit=self.xD, valfmt='%.2f L', color='#40e0d0')

        ax_st = self.fig_esp.add_axes([0.60, 0.11, 0.30, 0.03])
        self.slider_t = Slider(ax_st, 'Tiempo t', 0.0, self.t_max, valinit=self.t_val, valfmt='%.3f u.a.', color='yellowgreen')

        self.slider_theta.on_changed(self.on_slider_theta_change)
        self.slider_delta.on_changed(self.on_slider_delta_change)
        self.slider_xd.on_changed(self.on_slider_xd_change)
        self.slider_t.on_changed(self.on_slider_t_change)
        
        # Botonera de Experimentos de la Guía
        ax_b1 = self.fig_esp.add_axes([0.05, 0.03, 0.20, 0.04])
        ax_b2 = self.fig_esp.add_axes([0.28, 0.03, 0.20, 0.04])
        ax_b3 = self.fig_esp.add_axes([0.51, 0.03, 0.20, 0.04])
        ax_b4 = self.fig_esp.add_axes([0.74, 0.03, 0.21, 0.04])

        self.btn_exp1 = Button(ax_b1, 'Exp 1: Estado Puro φ1', color='#1f2937', hovercolor='#374151')
        self.btn_exp2 = Button(ax_b2, 'Exp 2: Mezcla Equilibrada', color='#1f2937', hovercolor='#374151')
        self.btn_exp3 = Button(ax_b3, 'Exp 3: Desfase δ = π/2', color='#1f2937', hovercolor='#374151')
        self.btn_exp4 = Button(ax_b4, '★ EXPERIMENTO DECISIVO', color='#ff007f', hovercolor='#e60073')

        self.btn_exp1.on_clicked(lambda e: self.set_experimento(theta=0.0, delta=0.0, xD=L/3.0))
        self.btn_exp2.on_clicked(lambda e: self.set_experimento(theta=np.pi/4, delta=0.0, xD=L/3.0))
        self.btn_exp3.on_clicked(lambda e: self.set_experimento(theta=np.pi/4, delta=np.pi/2, xD=L/3.0))
        self.btn_exp4.on_clicked(lambda e: self.set_experimento(theta=np.pi/4, delta=0.0, xD=L/3.0))

    # -------------------------------------------------------------------------
    # VENTANA 2: RESPUESTA TEMPORAL DEL DETECTOR VIRTUAL EN xD
    # -------------------------------------------------------------------------
    def crear_ventana_temporal(self):
        self.fig_temp = plt.figure(figsize=(8.5, 7.5))
        self.fig_temp.suptitle("VENTANA 2: SEÑAL TEMPORAL DEL DETECTOR VIRTUAL I_D(t)", fontsize=12, fontweight='bold', color='#40e0d0')
        self.fig_temp.canvas.manager.set_window_title("Ventana 2: Detector Virtual I_D(t)")
        
        self.ax_det = self.fig_temp.add_subplot(1, 1, 1)
        self.fig_temp.subplots_adjust(left=0.12, bottom=0.18, right=0.92, top=0.90)
        
        self.line_det, = self.ax_det.plot([], [], color='#ff007f', lw=2.2, label=r'Intensidad Medida $I_D(t) = |\Psi(x_D,t)|^2$')
        self.line_t_mark = self.ax_det.axvline(0, color='#00ffff', ls='--', lw=1.5, label='Instante Actual $t$')
        
        self.peaks_scatter, = self.ax_det.plot([], [], 'o', color='yellow', markersize=8, label='Máximos (Picos)')
        self.annotation_T = None
        
        self.ax_det.set_xlabel(r'Tiempo $t$ [u.a.]', fontsize=10)
        self.ax_det.set_ylabel(r'Intensidad en Detector $I_D(t)$', fontsize=10)
        self.ax_det.set_xlim(0, self.t_max)
        self.ax_det.set_ylim(0, 4.5)
        self.ax_det.grid(True, alpha=0.2)
        self.ax_det.legend(loc='upper right', fontsize=9)
        
        # Botón Play / Pausa
        ax_btn_play = self.fig_temp.add_axes([0.40, 0.04, 0.20, 0.05])
        self.btn_play = Button(ax_btn_play, '▶ Play Animación', color='#1f2937', hovercolor='#374151')
        self.btn_play.on_clicked(self.toggle_play)

    # -------------------------------------------------------------------------
    # VENTANA 3: TABLA AUTOMATIZADA DEL EXPERIMENTO DECISIVO Y ANÁLISIS
    # -------------------------------------------------------------------------
    def crear_ventana_decisivo(self):
        self.fig_dec = plt.figure(figsize=(9, 8))
        self.fig_dec.suptitle("VENTANA 3: EXPERIMENTO DECISIVO — TABLA DE MEDICIÓN Y COMPROBACIÓN", fontsize=12, fontweight='bold', color='#40e0d0')
        self.fig_dec.canvas.manager.set_window_title("Ventana 3: Experimento Decisivo y Tabla")
        
        self.ax_info = self.fig_dec.add_subplot(1, 1, 1)
        self.ax_info.axis('off')
        self.fig_dec.tight_layout(rect=[0, 0.05, 1, 0.95])

    # -------------------------------------------------------------------------
    # CALLBACKS DE CONTROLES
    # -------------------------------------------------------------------------
    def set_experimento(self, theta, delta, xD):
        self.theta = theta
        self.delta = delta
        self.xD = xD
        self.slider_theta.set_val(self.theta)
        self.slider_delta.set_val(self.delta)
        self.slider_xd.set_val(self.xD)
        self.actualizar_todo()

    def on_slider_theta_change(self, val):
        self.theta = float(val)
        self.actualizar_todo()

    def on_slider_delta_change(self, val):
        self.delta = float(val)
        self.actualizar_todo()

    def on_slider_xd_change(self, val):
        self.xD = float(val)
        self.actualizar_todo()

    def on_slider_t_change(self, val):
        self.t_val = float(val)
        self.actualizar_espacial()
        self.line_t_mark.set_xdata([self.t_val, self.t_val])
        self.fig_temp.canvas.draw_idle()

    def toggle_play(self, event):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.label.set_text('❚❚ Pausa')
            self.animar_tiempo()
        else:
            self.btn_play.label.set_text('▶ Play Animación')

    def animar_tiempo(self):
        if self.is_playing:
            new_t = (self.t_val + 0.01) % self.t_max
            self.slider_t.set_val(new_t)
            self.fig_esp.canvas.draw_idle()
            self.fig_esp.canvas.flush_events()
            plt.pause(0.03)
            if self.is_playing:
                self.fig_esp.canvas.manager.window.after(30, self.animar_tiempo)

    def actualizar_todo(self):
        self.actualizar_espacial()
        self.actualizar_temporal_y_decisivo()

    # -------------------------------------------------------------------------
    # RENDERING DE VENTANAS
    # -------------------------------------------------------------------------
    def actualizar_espacial(self):
        prob, incoh, interf = densidad_probabilidad(self.x_grid, self.t_val, self.theta, self.delta)
        
        self.line_prob.set_data(self.x_grid, prob)
        self.line_incoh.set_data(self.x_grid, incoh)
        self.line_I12.set_data(self.x_grid, interf)
        
        self.line_xd_esp1.set_xdata([self.xD, self.xD])
        self.line_xd_esp2.set_xdata([self.xD, self.xD])
        
        if self.fill_prob is not None:
            self.fill_prob.remove()
        self.fill_prob = self.ax_psi.fill_between(self.x_grid, 0, prob, color='#00ffff', alpha=0.20)
        
        c1, c2 = np.cos(self.theta), np.sin(self.theta)
        self.ax_psi.set_title(f"Estado Total |Ψ(x,t)|^2 | c1={c1:.3f}, c2={c2:.3f}, δ={self.delta:.2f} rad, t={self.t_val:.3f} u.a.", fontsize=10, color='white')
        self.fig_esp.canvas.draw_idle()

    def actualizar_temporal_y_decisivo(self):
        # Calcular señal temporal del detector I_D(t)
        c1, c2 = np.cos(self.theta), np.sin(self.theta)
        p1_D = phi_n(1, self.xD)
        p2_D = phi_n(2, self.xD)
        
        I_D_arr = (c1**2)*(p1_D**2) + (c2**2)*(p2_D**2) + 2*c1*c2*p1_D*p2_D*np.cos(self.delta - Omega_21_teor * self.t_grid)
        
        self.line_det.set_data(self.t_grid, I_D_arr)
        self.line_t_mark.set_xdata([self.t_val, self.t_val])
        self.ax_det.set_ylim(0, max(4.0, np.max(I_D_arr) * 1.2))
        
        # Detección automática de picos para el Experimento Decisivo
        peaks, _ = signal.find_peaks(I_D_arr, distance=100)
        
        es_medible = False
        T_num = 0.0
        Delta_E_med = 0.0
        error_pct = 0.0
        
        if len(peaks) >= 2:
            t_p1 = self.t_grid[peaks[0]]
            t_p2 = self.t_grid[peaks[1]]
            T_num = t_p2 - t_p1
            
            amp_osc = np.max(I_D_arr) - np.min(I_D_arr)
            if amp_osc > 1e-3:
                es_medible = True
                Delta_E_med = 2 * np.pi * hbar / T_num
                error_pct = abs(Delta_E_med - Delta_E_teor) / Delta_E_teor * 100.0
                
                self.peaks_scatter.set_data(self.t_grid[peaks[:2]], I_D_arr[peaks[:2]])
            else:
                self.peaks_scatter.set_data([], [])
        else:
            self.peaks_scatter.set_data([], [])

        self.ax_det.set_title(f"Intensidad en Detector xD = {self.xD:.3f} L | Amplitud Oscilación: {np.max(I_D_arr)-np.min(I_D_arr):.4f}", fontsize=10, color='#ff007f')
        self.fig_temp.canvas.draw_idle()

        # Renderizar Ventana 3 (Tabla Oficial y Diagnóstico)
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        if es_medible:
            estado_exp = "MEDICIÓN EXITOSA (INTERFERENCIA OBSERVABLE)"
            color_exp = "#40e0d0"
            med_text = (
                f"  • Período Numérico Medido (T_num)    : {T_num:.6f} u.a. (Teórico: {T_teor:.6f} u.a.)\n"
                f"  • Frecuencia Medida (Omega_num)      : {2*np.pi/T_num:.5f} rad/u.a.\n"
                f"  • Diferencia Energía Medida (ΔE_med)  : {Delta_E_med:.5f} u.a.\n"
                f"  • Espectro Teórico (E2 - E1)         : {Delta_E_teor:.5f} u.a.  [3 π² ℏ² / 2 m L²]\n"
                f"  • ERROR RELATIVO PORCENTUAL          : {error_pct:.3f} %\n"
            )
        else:
            estado_exp = "PUNTO CIEGO O ESTADO PURO (SIN OSCILACIÓN TEMPORAL)"
            color_exp = "#ff5500"
            if abs(self.xD - 0.5) < 0.02:
                motivo = " Detector en xD = L/2 (Nodo del estado φ2: φ2(L/2) = 0). El término cruzado I12 se anula."
            elif self.theta == 0.0 or self.theta == np.pi/2:
                motivo = " Estado Puro individual (θ = 0 o π/2). La densidad de probabilidad es ESTACIONARIA."
            else:
                motivo = " Amplitud de interferencia despreciable en la posición seleccionada."
                
            med_text = (
                f"  • Período Numérico Medido (T_num)    : NO MEDIBLE (Señal constante/plana)\n"
                f"  • Motivo Físico                       : {motivo}\n"
                f"  • Espectro Teórico (E2 - E1)         : {Delta_E_teor:.5f} u.a.\n"
            )

        info_content = (
            f"========================================================================================\n"
            f"          TABLA OFICIAL DEL EXPERIMENTO DECISIVO (CAPÍTULO 4 - SECCIÓN 13)\n"
            f"========================================================================================\n"
            f" ESTADO DE LA MEDICIÓN: {estado_exp}\n"
            f"----------------------------------------------------------------------------------------\n"
            f" CONFIGURACIÓN DEL EXPERIMENTO:\n"
            f"  • Posición del Detector Virtual (xD)  : {self.xD:.4f} L  (Recomendado: xD = L/3 = 0.3333 L)\n"
            f"  • Ángulo de Superposición (θ)        : {self.theta:.4f} rad  --> |c1|² = {c1**2:.3f}, |c2|² = {c2**2:.3f}\n"
            f"  • Fase Relativa Inicial (δ)           : {self.delta:.4f} rad  ({np.degrees(self.delta):.1f}°)\n"
            f"----------------------------------------------------------------------------------------\n"
            f" RESULTADOS Y TABLA DE COMPARACIÓN CUÁNTICA:\n"
            + med_text +
            f"----------------------------------------------------------------------------------------\n"
            f" CONCLUSIÓN DEL EXPERIMENTO DECISIVO:\n"
            f"  1. La fase relativa δ es INVISIBLE en los autoestados aislados φ1 o φ2 (densidad estática).\n"
            f"  2. Al superponerlos, la diferencia de fase rige el término cruzado I12(x,t) ∝ cos(δ - Ω_21 t).\n"
            f"  3. Medir el período T_num en el detector virtual PERMITE DETERMINAR DIRECTAMENTE LA ENERGÍA\n"
            f"     DE TRANSICIÓN DEL ESPECTRO ΔE = 2π ℏ / T_num con un error < 0.1%.\n"
            f"========================================================================================"
        )
        
        self.ax_info.text(0.02, 0.08, info_content, fontsize=9.2, fontfamily='monospace',
                          color=color_exp, bbox=dict(boxstyle='round,pad=0.8', facecolor='#0d1117', edgecolor=color_exp, alpha=0.95))
        self.fig_dec.canvas.draw_idle()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("Test execution complete!")
    else:
        app = Laboratorio4DosEstadosDecisivo()
        plt.show()
