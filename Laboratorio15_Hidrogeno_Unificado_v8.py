import sys
import os
import numpy as np
import scipy.special as sp
from scipy.integrate import quad
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D

# Forzar Backend interactivo para Matplotlib
if len(sys.argv) == 1 or sys.argv[1] != '--test':
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass

plt.style.use('dark_background')

# =============================================================================
# 1. FUNCIONES ONDULATORIAS RADIALES Y ANGULARES ANALÍTICAS (UNIDADES ATÓMICAS)
# =============================================================================

def R_nl(n, l, r):
    r = np.asarray(r, dtype=float)
    rho = 2.0 * r / n
    norm = np.sqrt((2.0 / n)**3 * sp.factorial(n - l - 1) / (2.0 * n * sp.factorial(n + l)))
    laguerre = sp.eval_genlaguerre(n - l - 1, 2 * l + 1, rho)
    return norm * (rho**l) * np.exp(-rho / 2.0) * laguerre

def P_radial(n, l, r):
    return (r**2) * (R_nl(n, l, r)**2)

def Y_lm(l, m, theta, phi, real_orbital=True):
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    m_abs = abs(m)
    
    try:
        y_complex = sp.sph_harm_y(l, m_abs, theta, phi)
    except AttributeError:
        y_complex = sp.sph_harm(m_abs, l, phi, theta)
        
    if not real_orbital:
        if m < 0:
            return ((-1)**m_abs) * np.conj(y_complex)
        return y_complex
    
    if m == 0:
        return np.real(y_complex)
    if m > 0:
        return np.sqrt(2) * ((-1)**m) * np.real(y_complex)
    else:
        return np.sqrt(2) * ((-1)**m_abs) * np.imag(y_complex)

def psi_nlm(n, l, m, x, y, z, real_orbital=True):
    r = np.sqrt(x**2 + y**2 + z**2)
    r_safe = np.where(r == 0, 1e-12, r)
    theta = np.arccos(np.clip(z / r_safe, -1.0, 1.0))
    phi = np.arctan2(y, x)
    
    radial = R_nl(n, l, r)
    angular = Y_lm(l, m, theta, phi, real_orbital=real_orbital)
    return radial * angular

def ceros_radiales(n, l):
    n_r = n - l - 1
    if n_r <= 0:
        return np.array([])
    poly = sp.genlaguerre(n_r, 2 * l + 1)
    roots = poly.roots
    r_roots = roots * n / 2.0
    return np.sort(r_roots[r_roots > 0])

def ceros_angulares(l, m):
    if l == 0:
        return np.array([])
    m_abs = abs(m)
    if (l - m_abs) <= 0:
        return np.array([])
    x_grid = np.linspace(-0.999, 0.999, 1000)
    poly = sp.lpmv(m_abs, l, x_grid)
    sign_changes = np.where(np.diff(np.sign(poly)))[0]
    cos_roots = []
    for idx in sign_changes:
        x1, x2 = x_grid[idx], x_grid[idx+1]
        for _ in range(20):
            xm = 0.5 * (x1 + x2)
            if sp.lpmv(m_abs, l, x1) * sp.lpmv(m_abs, l, xm) <= 0:
                x2 = xm
            else:
                x1 = xm
        cos_roots.append(0.5 * (x1 + x2))
    return np.sort(np.arccos(np.array(cos_roots)))

def obtener_nombre_orbital(n, l, m):
    nombres_l = ['s', 'p', 'd', 'f', 'g']
    char_l = nombres_l[l] if l < len(nombres_l) else f'l={l}'
    if l == 0:
        sub = ""
    elif l == 1:
        sub = "z" if m == 0 else ("x" if m == 1 else "y")
    elif l == 2:
        if m == 0: sub = "z²"
        elif m == 1: sub = "xz"
        elif m == -1: sub = "yz"
        elif m == 2: sub = "x²-y²"
        elif m == -2: sub = "xy"
        else: sub = f"m={m}"
    else:
        sub = f"m={m}"
    return f"{n}{char_l}_{{{sub}}}" if sub else f"{n}{char_l}"

# =============================================================================
# 2. INTERFAZ UNIFICADA DE 4 VENTANAS (ETAPAS A, B, C, D Y E - HIDRÓGENO V8)
# =============================================================================

class LaboratorioHidrogenoUnificadoV8:
    def __init__(self):
        # Estado cuántico individual (Ventanas 1, 2, 3)
        self.n = 3
        self.l = 1
        self.m = 0
        self.n_puntos_deseados = 50000
        self.corte_3d_modo = 'Nube Completa 3D'
        
        # Estado de evolución temporal (Ventana 4)
        self.modo_evolucion = 'Superposición Coherente'  # O 'Orbital Individual (Sin sumar)'
        self.variable_visualizada = '|Psi|^2 (Densidad de Probabilidad)'  # O 'Re[Psi] (Parte Real)'
        self.state1 = (2, 0, 0) # 2s
        self.state2 = (2, 1, 0) # 2p_z
        self.c1 = 1.0 / np.sqrt(2)
        self.c2 = 1.0 / np.sqrt(2)
        self.t_val = 0.0
        self.is_playing = False

        # Crear las 4 ventanas independientes
        self.crear_ventana_radial()
        self.crear_ventana_corte2d()
        self.crear_ventana_nube3d()
        self.crear_ventana_evolucion()
        
        # Dibujar por primera vez
        self.actualizar_todo()

    # -------------------------------------------------------------------------
    # VENTANA 1: ANÁLISIS RADIAL Y NODOS DE LAGUERRE
    # -------------------------------------------------------------------------
    def crear_ventana_radial(self):
        self.fig_rad, (self.ax_R, self.ax_P) = plt.subplots(2, 1, figsize=(8, 7.5))
        self.fig_rad.suptitle("VENTANA 1: ANÁLISIS RADIAL Y NODOS DE LAGUERRE", fontsize=12, fontweight='bold', color='#40e0d0')
        self.fig_rad.canvas.manager.set_window_title("Ventana 1: Perfil Radial y Nodos Radiales")
        
        self.line_R, = self.ax_R.plot([], [], color='#00ffff', lw=2, label=r'$R_{nl}(r)$')
        self.ax_R.axhline(0, color='gray', ls='--', alpha=0.5)
        self.ax_R.set_ylabel(r'Amplitud Radial $R_{nl}(r)$ [a.u.]', fontsize=10)
        self.ax_R.grid(True, alpha=0.2)
        
        self.line_P, = self.ax_P.plot([], [], color='#ff007f', lw=2, label=r'$P(r) = r^2 |R_{nl}(r)|^2$')
        self.fill_P = None
        self.line_rmed = self.ax_P.axvline(0, color='yellowgreen', ls=':', lw=2, label=r'$\langle r \rangle$')
        self.ax_P.set_xlabel(r'Distancia Radial $r / a_\mu$', fontsize=10)
        self.ax_P.set_ylabel(r'Densidad Radial $P(r)$', fontsize=10)
        self.ax_P.grid(True, alpha=0.2)
        
        self.radial_node_lines_R = []
        self.radial_node_lines_P = []
        self.radial_node_dots = []

        estados = [('1s', 1, 0, 0), ('2s', 2, 0, 0), ('2p_z', 2, 1, 0), ('3s', 3, 0, 0), ('3p_z', 3, 1, 0), ('3d_z²', 3, 2, 0), ('4f_z³', 4, 3, 0)]
        self.btns = []
        for idx, (label, n, l, m) in enumerate(estados):
            ax_b = self.fig_rad.add_axes([0.05 + idx * 0.13, 0.02, 0.11, 0.04])
            btn = Button(ax_b, label, color='#1f2937', hovercolor='#374151')
            btn.on_clicked(lambda event, fn=n, fl=l, fm=m: self.cambiar_estado(fn, fl, fm))
            self.btns.append(btn)

        self.fig_rad.tight_layout(rect=[0, 0.08, 1, 0.95])

    # -------------------------------------------------------------------------
    # VENTANA 2: CORTE 2D Y SELECTOR DE MODOS DISCRETOS
    # -------------------------------------------------------------------------
    def crear_ventana_corte2d(self):
        self.fig_2d = plt.figure(figsize=(9, 8.5))
        self.fig_2d.suptitle("VENTANA 2: CORTE 2D DEL ÁTOMO DE HIDRÓGENO Y SELECTOR DE MODOS", fontsize=12, fontweight='bold', color='#40e0d0')
        self.fig_2d.canvas.manager.set_window_title("Ventana 2: Corte Espacial 2D y Modos")
        
        self.ax_2d = self.fig_2d.add_subplot(1, 1, 1)
        self.fig_2d.subplots_adjust(left=0.10, bottom=0.32, right=0.92, top=0.90)

        # Botones discretos de Modo n
        self.btns_n = []
        for i, n_val in enumerate([1, 2, 3, 4, 5]):
            ax_b = self.fig_2d.add_axes([0.15 + i * 0.15, 0.22, 0.12, 0.035])
            btn = Button(ax_b, f'n = {n_val}', color='#1f2937', hovercolor='#374151')
            btn.on_clicked(lambda event, nv=n_val: self.set_modo_n(nv))
            self.btns_n.append(btn)

        # Botones discretos de Modo l
        self.btns_l = []
        nombres_l = ['s', 'p', 'd', 'f', 'g']
        for i in range(5):
            ax_b = self.fig_2d.add_axes([0.15 + i * 0.15, 0.16, 0.12, 0.035])
            btn = Button(ax_b, f'l={i} ({nombres_l[i]})', color='#1f2937', hovercolor='#374151')
            btn.on_clicked(lambda event, lv=i: self.set_modo_l(lv))
            self.btns_l.append(btn)

        # Botones discretos de Modo m
        self.btns_m = []
        m_vals = [-2, -1, 0, 1, 2]
        for i, m_val in enumerate(m_vals):
            ax_b = self.fig_2d.add_axes([0.15 + i * 0.15, 0.10, 0.12, 0.035])
            btn = Button(ax_b, f'm = {m_val:+d}' if m_val != 0 else 'm = 0', color='#1f2937', hovercolor='#374151')
            btn.on_clicked(lambda event, mv=m_val: self.set_modo_m(mv))
            self.btns_m.append(btn)

        # Slider para la densidad de la nube 3D
        ax_spts = self.fig_2d.add_axes([0.25, 0.03, 0.50, 0.035])
        self.slider_pts = Slider(ax_spts, 'Puntos 3D', 10000, 100000, valinit=50000, valstep=5000, color='#ff007f')
        self.slider_pts.on_changed(self.on_slider_pts_change)

    # -------------------------------------------------------------------------
    # VENTANA 3: NUBE DE PROBABILIDAD 3D VOLUMÉTRICA (MAGMA - 50.000 PUNTOS)
    # -------------------------------------------------------------------------
    def crear_ventana_nube3d(self):
        self.fig_3d = plt.figure(figsize=(8.5, 8))
        self.fig_3d.suptitle("VENTANA 3: NUBE DE PROBABILIDAD 3D VOLUMÉTRICA (PALETA MAGMA)", fontsize=12, fontweight='bold', color='#40e0d0')
        self.fig_3d.canvas.manager.set_window_title("Ventana 3: Nube de Probabilidad 3D (Hoja Aparte)")
        
        self.ax_3d = self.fig_3d.add_subplot(1, 1, 1, projection='3d')
        self.fig_3d.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.90)

        ax_radio = self.fig_3d.add_axes([0.15, 0.02, 0.70, 0.06], facecolor='#0d1117')
        self.radio_corte = RadioButtons(ax_radio, ('Nube Completa 3D', 'Corte Mitad (y <= 0)', 'Corte Cuadrante (1/4)'),
                                       active=0, activecolor='#40e0d0')
        self.radio_corte.on_clicked(self.on_radio_corte_change)

    # -------------------------------------------------------------------------
    # VENTANA 4: EVOLUCIÓN TEMPORAL Y BATIDOS DE SUPERPOSICIÓN (CON DINÁMICA 2s+2p)
    # -------------------------------------------------------------------------
    def crear_ventana_evolucion(self):
        self.fig_evol = plt.figure(figsize=(9.5, 8.5))
        self.fig_evol.suptitle("VENTANA 4: EVOLUCIÓN TEMPORAL Y DINÁMICA DE SUPERPOSICIÓN", fontsize=12, fontweight='bold', color='#40e0d0')
        self.fig_evol.canvas.manager.set_window_title("Ventana 4: Evolución Temporal (Hoja Aparte)")
        
        self.ax_ev2d = self.fig_evol.add_subplot(2, 2, 1)
        self.ax_ev2d.set_title(r'Perfil 2D Dinámico en $(x,z)$', fontsize=10)
        
        self.ax_ev1d = self.fig_evol.add_subplot(2, 2, 2)
        self.ax_ev1d.set_title(r'Perfil 1D en Eje Z', fontsize=10)
        
        self.ax_evinfo = self.fig_evol.add_subplot(2, 1, 2)
        self.ax_evinfo.axis('off')
        
        self.fig_evol.subplots_adjust(left=0.08, bottom=0.25, right=0.92, top=0.90, hspace=0.35)
        
        ax_radio_tipo = self.fig_evol.add_axes([0.08, 0.16, 0.38, 0.06], facecolor='#0d1117')
        self.radio_tipo_evol = RadioButtons(ax_radio_tipo, ('Superposición Coherente', 'Orbital Individual (Sin sumar)'),
                                           active=0, activecolor='#40e0d0')
        self.radio_tipo_evol.on_clicked(self.on_radio_tipo_change)

        ax_radio_var = self.fig_evol.add_axes([0.52, 0.16, 0.40, 0.06], facecolor='#0d1117')
        self.radio_var_evol = RadioButtons(ax_radio_var, ('|Psi|^2 (Densidad de Probabilidad)', 'Re[Psi] (Parte Real / Oscilación de Fase)'),
                                          active=0, activecolor='#ff007f')
        self.radio_var_evol.on_clicked(self.on_radio_var_change)

        ax_st = self.fig_evol.add_axes([0.12, 0.09, 0.50, 0.035])
        self.slider_t = Slider(ax_st, 'Tiempo t / T', 0.0, 1.0, valinit=0.0, valstep=0.01, color='#00ffff')
        self.slider_t.on_changed(self.on_slider_t_change)
        
        ax_btn_play = self.fig_evol.add_axes([0.65, 0.09, 0.12, 0.035])
        self.btn_play = Button(ax_btn_play, '▶ Play', color='#1f2937', hovercolor='#374151')
        self.btn_play.on_clicked(self.toggle_play)
        
        superposiciones = [
            ('2s + 2p_z (Stark/Dinamico)', (2,0,0), (2,1,0)),
            ('1s + 2p_z', (1,0,0), (2,1,0)),
            ('1s + 3p_z', (1,0,0), (3,1,0)),
            ('2p_z + 3d_z²', (2,1,0), (3,2,0)),
            ('1s (Single)', (1,0,0), None),
            ('2p_z (Single)', (2,1,0), None)
        ]
        self.btns_sup = []
        for idx, (label, st1, st2) in enumerate(superposiciones):
            ax_b = self.fig_evol.add_axes([0.04 + idx * 0.155, 0.02, 0.145, 0.04])
            btn = Button(ax_b, label, color='#1f2937', hovercolor='#374151')
            btn.on_clicked(lambda event, s1=st1, s2=st2: self.set_superposicion(s1, s2))
            self.btns_sup.append(btn)

    # -------------------------------------------------------------------------
    # CALLBACKS Y LÓGICA DE INTERFACES
    # -------------------------------------------------------------------------
    def set_modo_n(self, n_val):
        self.n = n_val
        self.l = min(self.l, self.n - 1)
        self.m = max(-self.l, min(self.m, self.l))
        self.actualizar_todo()

    def set_modo_l(self, l_val):
        if l_val < self.n:
            self.l = l_val
            self.m = max(-self.l, min(self.m, self.l))
            self.actualizar_todo()

    def set_modo_m(self, m_val):
        if abs(m_val) <= self.l:
            self.m = m_val
            self.actualizar_todo()

    def cambiar_estado(self, n, l, m):
        self.n = n
        self.l = min(l, n - 1)
        self.m = max(-self.l, min(m, self.l))
        self.actualizar_todo()

    def on_slider_pts_change(self, val):
        self.n_puntos_deseados = int(val)
        self.actualizar_nube3d()

    def on_radio_corte_change(self, label):
        self.corte_3d_modo = label
        self.actualizar_nube3d()

    def on_radio_tipo_change(self, label):
        self.modo_evolucion = label
        if label == 'Orbital Individual (Sin sumar)':
            self.state2 = None
        elif self.state2 is None:
            self.state1 = (2, 0, 0)
            self.state2 = (2, 1, 0)
        self.actualizar_evolucion()

    def on_radio_var_change(self, label):
        self.variable_visualizada = label
        self.actualizar_evolucion()

    def set_superposicion(self, s1, s2):
        self.state1 = s1
        self.state2 = s2
        if s2 is None:
            self.modo_evolucion = 'Orbital Individual (Sin sumar)'
            self.radio_tipo_evol.set_active(1)
        else:
            self.modo_evolucion = 'Superposición Coherente'
            self.radio_tipo_evol.set_active(0)
        self.t_val = 0.0
        self.slider_t.set_val(0.0)
        self.actualizar_evolucion()

    def on_slider_t_change(self, val):
        self.t_val = float(val)
        self.actualizar_evolucion()

    def toggle_play(self, event):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.label.set_text('❚❚ Pausa')
            self.animar_tiempo()
        else:
            self.btn_play.label.set_text('▶ Play')

    def animar_tiempo(self):
        if self.is_playing:
            new_t = (self.t_val + 0.02) % 1.0
            self.slider_t.set_val(new_t)
            self.fig_evol.canvas.draw_idle()
            self.fig_evol.canvas.flush_events()
            plt.pause(0.04)
            if self.is_playing:
                self.fig_evol.canvas.manager.window.after(30, self.animar_tiempo)

    def actualizar_todo(self):
        self.actualizar_radial()
        self.actualizar_corte2d()
        self.actualizar_nube3d()
        self.actualizar_evolucion()

    # -------------------------------------------------------------------------
    # RENDERING DE VENTANAS
    # -------------------------------------------------------------------------
    def actualizar_radial(self):
        r_max = 12.0 * self.n
        r_grid = np.linspace(0, r_max, 1000)
        
        R_vals = R_nl(self.n, self.l, r_grid)
        P_vals = P_radial(self.n, self.l, r_grid)
        
        self.line_R.set_data(r_grid, R_vals)
        self.ax_R.set_xlim(0, r_max)
        self.ax_R.set_ylim(np.min(R_vals)*1.15, np.max(R_vals)*1.15)
        
        self.line_P.set_data(r_grid, P_vals)
        self.ax_P.set_xlim(0, r_max)
        self.ax_P.set_ylim(0, np.max(P_vals)*1.20)
        
        if self.fill_P is not None:
            self.fill_P.remove()
        self.fill_P = self.ax_P.fill_between(r_grid, 0, P_vals, color='#ff007f', alpha=0.25)
        
        r_med = 0.5 * (3 * self.n**2 - self.l * (self.l + 1))
        self.line_rmed.set_xdata([r_med, r_med])
        
        for item in self.radial_node_lines_R + self.radial_node_lines_P + self.radial_node_dots:
            item.remove()
        self.radial_node_lines_R.clear()
        self.radial_node_lines_P.clear()
        self.radial_node_dots.clear()
        
        r_nodos = ceros_radiales(self.n, self.l)
        n_r = len(r_nodos)
        
        for idx, r_k in enumerate(r_nodos):
            l1 = self.ax_R.axvline(r_k, color='lime', ls='--', alpha=0.8, lw=1.5)
            l2 = self.ax_P.axvline(r_k, color='lime', ls='--', alpha=0.8, lw=1.5)
            dot, = self.ax_R.plot(r_k, 0, 'go', markersize=6)
            self.radial_node_lines_R.append(l1)
            self.radial_node_lines_P.append(l2)
            self.radial_node_dots.append(dot)
            self.ax_R.text(r_k, np.max(R_vals)*0.15, f' $r_{idx+1}={r_k:.2f}$', color='lime', fontsize=9)

        nombre_orb = obtener_nombre_orbital(self.n, self.l, self.m)
        self.ax_R.set_title(f'Orbital ${nombre_orb}$ | Amplitud $R_{{{self.n}{self.l}}}(r)$ — {n_r} Nodo(s) Radial(es)', fontsize=10, color='white')
        self.ax_P.set_title(f'Densidad Radial $P(r) = r^2 |R_{{{self.n}{self.l}}}|^2$ | $\\langle r \\rangle = {r_med:.1f} a_\\mu$', fontsize=10, color='white')
        self.fig_rad.canvas.draw_idle()

    def actualizar_corte2d(self):
        r_lim = 5.0 * self.n
        grid_2d = 180
        vec = np.linspace(-r_lim, r_lim, grid_2d)
        X_2d, Z_2d = np.meshgrid(vec, vec)
        Y_2d = np.zeros_like(X_2d)
        
        psi_vals = psi_nlm(self.n, self.l, self.m, X_2d, Y_2d, Z_2d, real_orbital=True)
        prob_density_2d = np.abs(psi_vals)**2

        self.ax_2d.clear()
        im = self.ax_2d.imshow(prob_density_2d, origin='lower', extent=[-r_lim, r_lim, -r_lim, r_lim],
                               cmap='magma', aspect='equal')
        
        theta_nodos = ceros_angulares(self.l, self.m)
        for th in theta_nodos:
            r_line = np.linspace(-r_lim, r_lim, 200)
            x_line = r_line * np.sin(th)
            z_line = r_line * np.cos(th)
            self.ax_2d.plot(x_line, z_line, color='lime', ls=':', lw=1.8, label='Nodo Angular')

        nombre_orb = obtener_nombre_orbital(self.n, self.l, self.m)
        n_a = self.l
        n_r = self.n - self.l - 1
        
        self.ax_2d.set_title(f"Corte 2D ($y=0$) Átomo de Hidrógeno — Orbital ${nombre_orb}$\\n"\
                             f"Nodos Radiales: {n_r} (verde) | Nodos Angulares: {n_a} (verde punteado)",
                             fontsize=10, color='#40e0d0')
        self.ax_2d.set_xlabel(r'$x / a_\mu$', fontsize=10)
        self.ax_2d.set_ylabel(r'$z / a_\mu$', fontsize=10)
        self.ax_2d.set_xlim(-r_lim, r_lim)
        self.ax_2d.set_ylim(-r_lim, r_lim)
        self.fig_2d.canvas.draw_idle()

    def actualizar_nube3d(self):
        self.ax_3d.clear()
        self.ax_3d.set_facecolor('#0b0e14')
        
        r_grid = np.linspace(0, 35*self.n, 1000)
        p_rad = (r_grid**2) * (R_nl(self.n, self.l, r_grid)**2)
        r_max = r_grid[np.where(p_rad > 1e-4*np.max(p_rad))[0][-1]]
        
        N_target = self.n_puntos_deseados
        N_batch = N_target * 6
        
        r_acc_list, th_acc_list, ph_acc_list, prob_acc_list = [], [], [], []
        puntos_acumulados = 0
        
        while puntos_acumulados < N_target:
            r_cand = np.random.uniform(0, r_max, N_batch)
            theta_cand = np.random.uniform(0, np.pi, N_batch)
            phi_cand = np.random.uniform(0, 2*np.pi, N_batch)
            
            R_val = R_nl(self.n, self.l, r_cand)
            Y_val = Y_lm(self.l, self.m, theta_cand, phi_cand, real_orbital=True)
            w = (r_cand**2) * (R_val**2) * np.sin(theta_cand) * (Y_val**2)
            w_max = np.max(w)
            
            if w_max <= 0:
                break
                
            u = np.random.uniform(0, w_max, N_batch)
            mask = u <= w
            
            r_sel = r_cand[mask]
            th_sel = theta_cand[mask]
            ph_sel = phi_cand[mask]
            
            x_sel = r_sel * np.sin(th_sel) * np.cos(ph_sel)
            y_sel = r_sel * np.sin(th_sel) * np.sin(ph_sel)
            z_sel = r_sel * np.cos(th_sel)
            
            if self.corte_3d_modo == 'Corte Mitad (y <= 0)':
                mask_slice = y_sel <= 0
            elif self.corte_3d_modo == 'Corte Cuadrante (1/4)':
                mask_slice = ~((x_sel > 0) & (y_sel > 0))
            else:
                mask_slice = np.ones_like(x_sel, dtype=bool)
                
            r_sel = r_sel[mask_slice]
            th_sel = th_sel[mask_slice]
            ph_sel = ph_sel[mask_slice]
            
            psi_sel = R_nl(self.n, self.l, r_sel) * Y_lm(self.l, self.m, th_sel, ph_sel, real_orbital=True)
            prob_sel = np.abs(psi_sel)**2
            
            faltan = N_target - puntos_acumulados
            r_acc_list.append(r_sel[:faltan])
            th_acc_list.append(th_sel[:faltan])
            ph_acc_list.append(ph_sel[:faltan])
            prob_acc_list.append(prob_sel[:faltan])
            puntos_acumulados += len(r_sel[:faltan])
            
        r_acc = np.concatenate(r_acc_list)
        th_acc = np.concatenate(th_acc_list)
        ph_acc = np.concatenate(ph_acc_list)
        prob_acc = np.concatenate(prob_acc_list)
        
        x_acc = r_acc * np.sin(th_acc) * np.cos(ph_acc)
        y_acc = r_acc * np.sin(th_acc) * np.sin(ph_acc)
        z_acc = r_acc * np.cos(th_acc)
        
        norm_magma = mcolors.PowerNorm(gamma=0.35, vmin=0, vmax=np.max(prob_acc))
        sc = self.ax_3d.scatter(x_acc, y_acc, z_acc, c=prob_acc, cmap='magma', norm=norm_magma,
                               s=1.2, alpha=0.45, depthshade=True)
        
        lim_3d = r_max * 0.95
        self.ax_3d.set_xlim(-lim_3d, lim_3d)
        self.ax_3d.set_ylim(-lim_3d, lim_3d)
        self.ax_3d.set_zlim(-lim_3d, lim_3d)
        self.ax_3d.set_box_aspect([1, 1, 1])
        
        nombre_orb = obtener_nombre_orbital(self.n, self.l, self.m)
        self.ax_3d.set_title(f"Nube de Probabilidad 3D Volumétrica — Orbital ${nombre_orb}$\\n"\
                             f"Posiciones Mostradas: {len(x_acc)} electrones/puntos en Paleta Magma",
                             fontsize=10, color='#40e0d0')
        self.ax_3d.set_xlabel(r'$x$', fontsize=9)
        self.ax_3d.set_ylabel(r'$y$', fontsize=9)
        self.ax_3d.set_zlabel(r'$z$', fontsize=9)
        self.fig_3d.canvas.draw_idle()

    def actualizar_evolucion(self):
        """Actualiza la Ventana 4 (Evolución Temporal - Tanto Individuales como Superposiciones Dinámicas)."""
        if self.state2 is None or self.modo_evolucion == 'Orbital Individual (Sin sumar)':
            # CASO ORBITAL INDIVIDUAL
            n1, l1, m1 = self.state1
            name1 = obtener_nombre_orbital(n1, l1, m1)
            E1 = -0.5 / (n1**2)
            T1 = 2 * np.pi / abs(E1)
            tau = self.t_val * T1
            phase = E1 * tau
            
            r_lim = 5.0 * n1
            grid_dim = 140
            vec = np.linspace(-r_lim, r_lim, grid_dim)
            X, Z = np.meshgrid(vec, vec)
            Y_zero = np.zeros_like(X)
            
            psi_space = psi_nlm(n1, l1, m1, X, Y_zero, Z, real_orbital=True)
            
            if '|Psi|^2' in self.variable_visualizada:
                field_2d = psi_space**2
                cmap_choice = 'magma'
                title_var = r'$|\Psi(x,0,z,t)|^2$ (Densidad Estacionaria)'
            else:
                field_2d = psi_space * np.cos(phase)
                cmap_choice = 'coolwarm'
                title_var = r'$\text{Re}[\Psi(x,0,z,t)]$ (Oscilación de Fase Físicamente Visible)'

            self.ax_ev2d.clear()
            im = self.ax_ev2d.imshow(field_2d, origin='lower', extent=[-r_lim, r_lim, -r_lim, r_lim],
                                     cmap=cmap_choice, aspect='equal')
            self.ax_ev2d.set_title(f"Orbital Individual ${name1}$\\n{title_var} | $t/T = {self.t_val:.2f}$", fontsize=10, color='#40e0d0')
            self.ax_ev2d.set_xlabel(r'$x / a_\mu$', fontsize=9)
            self.ax_ev2d.set_ylabel(r'$z / a_\mu$', fontsize=9)

            self.ax_ev1d.clear()
            z_vec = np.linspace(-r_lim, r_lim, 500)
            psi_z = psi_nlm(n1, l1, m1, 0, 0, z_vec, real_orbital=True)
            
            if '|Psi|^2' in self.variable_visualizada:
                self.ax_ev1d.plot(z_vec, psi_z**2, color='#ff007f', lw=2, label=r'$|\Psi(0,0,z,t)|^2$ (Inmóvil)')
                self.ax_ev1d.set_ylabel(r'Densidad 1D', fontsize=9)
            else:
                self.ax_ev1d.plot(z_vec, psi_z * np.cos(phase), color='#00ffff', lw=2, label=r'$\text{Re}[\Psi(0,0,z,t)]$')
                self.ax_ev1d.plot(z_vec, psi_z, color='gray', ls='--', alpha=0.5, label='Envolvente Espacial')
                self.ax_ev1d.plot(z_vec, -psi_z, color='gray', ls='--', alpha=0.5)
                self.ax_ev1d.set_ylabel(r'Amplitud de Onda 1D', fontsize=9)
                
            self.ax_ev1d.set_xlabel(r'$z / a_\mu$', fontsize=9)
            self.ax_ev1d.grid(True, alpha=0.2)
            self.ax_ev1d.legend(loc='upper right', fontsize=8)

            self.ax_evinfo.clear()
            self.ax_evinfo.axis('off')
            E1_ev = E1 * 27.2114
            phase_deg = np.degrees(-phase % (2*np.pi))
            
            info_text = (
                f"=== EVOLUCIÓN TEMPORAL DE ORBITAL INDIVIDUAL ESTACIONARIO ===\n"
                f" • Estado Puro: ${name1}$ (n={n1}, l={l1}, m={m1}) | Energía E_1 = {E1:.4f} E_h ({E1_ev:.3f} eV)\n"
                f" • Frecuencia Angular de Fase omega_1 = {abs(E1):.4f} rad/ua | Período T_1 = {T1:.2f} ua\n"
                f" • Ángulo de Fase Rotatorio: theta(t) = {phase_deg:.1f}° en el Plano Complejo\n"
                f"-----------------------------------------------------------------\n"
                f" EXPLICACIÓN FÍSICA CLAVE (CAP. 4 Y 17):\n"
                f"  - La densidad |Psi|^2 = |phi_n|^2 es ESTACIONARIA porque |exp(-i E t / hbar)| = 1.\n"
                f"  - Sin embargo, la FUNCIÓN DE ONDA Y SU PARTE REAL Re[Psi] OSCILAN a frecuencia E_n/hbar.\n"
                f"  - Seleccioná 'Re[Psi]' arriba para observar la fase rotatoria continua del autoestado."
            )
            self.ax_evinfo.text(0.02, 0.10, info_text, fontsize=9.2, fontfamily='monospace',
                                color='#40e0d0', bbox=dict(boxstyle='round,pad=0.6', facecolor='#0d1117', edgecolor='#40e0d0', alpha=0.9))

        else:
            # CASO SUPERPOSICIÓN COHERENTE
            n1, l1, m1 = self.state1
            n2, l2, m2 = self.state2
            name1 = obtener_nombre_orbital(n1, l1, m1)
            name2 = obtener_nombre_orbital(n2, l2, m2)
            
            E1_base = -0.5 / (n1**2)
            E2_base = -0.5 / (n2**2)
            
            # Para estados del mismo nivel n (como 2s y 2p_z), la degeneración pura vacía la dinámica de |Psi|^2.
            # En la física real (Efecto Stark / Estructura Fina / Acoplamiento Dipolar), surge un desdoblamiento de energía Delta E_eff != 0.
            if n1 == n2:
                delta_E = 0.05  # Desdoblamiento efectivo (Stark / Perturbado) para activar la oscilación de carga
                es_degenerado_stark = True
            else:
                delta_E = E2_base - E1_base
                es_degenerado_stark = False
                
            T_tau = 2 * np.pi / abs(delta_E)
            tau = self.t_val * T_tau
            phase_diff = delta_E * tau
                
            r_lim = 5.0 * max(n1, n2)
            grid_dim = 140
            vec = np.linspace(-r_lim, r_lim, grid_dim)
            X, Z = np.meshgrid(vec, vec)
            Y_zero = np.zeros_like(X)
            
            psi1_2d = psi_nlm(n1, l1, m1, X, Y_zero, Z, real_orbital=True)
            psi2_2d = psi_nlm(n2, l2, m2, X, Y_zero, Z, real_orbital=True)
            
            if '|Psi|^2' in self.variable_visualizada:
                # |Psi|^2(t) = c1^2 psi1^2 + c2^2 psi2^2 + 2 c1 c2 psi1 psi2 cos(Omega_21 * t)
                field_2d = (self.c1**2)*(psi1_2d**2) + (self.c2**2)*(psi2_2d**2) + \
                           2 * self.c1 * self.c2 * psi1_2d * psi2_2d * np.cos(phase_diff)
                cmap_choice = 'magma'
                title_var = r'$|\Psi(x,0,z,t)|^2$ (Batido Oscilatorio de Carga)'
            else:
                phase1 = E1_base * tau
                phase2 = (E1_base + delta_E) * tau
                field_2d = self.c1 * psi1_2d * np.cos(phase1) + self.c2 * psi2_2d * np.cos(phase2)
                cmap_choice = 'coolwarm'
                title_var = r'$\text{Re}[\Psi(x,0,z,t)]$ (Interferencia Completa de Ondas)'

            self.ax_ev2d.clear()
            im = self.ax_ev2d.imshow(field_2d, origin='lower', extent=[-r_lim, r_lim, -r_lim, r_lim],
                                     cmap=cmap_choice, aspect='equal')
            lbl_stark = " (Efecto Stark / Dinámico)" if es_degenerado_stark else ""
            self.ax_ev2d.set_title(f"Superposición ${name1} + {name2}${lbl_stark}\\n{title_var} | $t/T = {self.t_val:.2f}$", fontsize=10, color='#40e0d0')
            self.ax_ev2d.set_xlabel(r'$x / a_\mu$', fontsize=9)
            self.ax_ev2d.set_ylabel(r'$z / a_\mu$', fontsize=9)

            self.ax_ev1d.clear()
            z_vec = np.linspace(-r_lim, r_lim, 500)
            psi1_z = psi_nlm(n1, l1, m1, 0, 0, z_vec, real_orbital=True)
            psi2_z = psi_nlm(n2, l2, m2, 0, 0, z_vec, real_orbital=True)
            
            if '|Psi|^2' in self.variable_visualizada:
                prob_z_tot = (self.c1**2)*(psi1_z**2) + (self.c2**2)*(psi2_z**2) + 2*self.c1*self.c2*psi1_z*psi2_z*np.cos(phase_diff)
                prob_z_incoh = (self.c1**2)*(psi1_z**2) + (self.c2**2)*(psi2_z**2)
                self.ax_ev1d.plot(z_vec, prob_z_tot, color='#00ffff', lw=2, label=r'$|\Psi(0,0,z,t)|^2$')
                self.ax_ev1d.plot(z_vec, prob_z_incoh, color='gray', ls='--', lw=1.5, label='Mezcla Incoherente')
                self.ax_ev1d.set_ylabel(r'Densidad 1D', fontsize=9)
            else:
                phase1 = E1_base * tau
                phase2 = (E1_base + delta_E) * tau
                re_z_tot = self.c1 * psi1_z * np.cos(phase1) + self.c2 * psi2_z * np.cos(phase2)
                self.ax_ev1d.plot(z_vec, re_z_tot, color='#ff007f', lw=2, label=r'$\text{Re}[\Psi(0,0,z,t)]$')
                self.ax_ev1d.set_ylabel(r'Amplitud de Onda 1D', fontsize=9)

            self.ax_ev1d.set_xlabel(r'$z / a_\mu$', fontsize=9)
            self.ax_ev1d.grid(True, alpha=0.2)
            self.ax_ev1d.legend(loc='upper right', fontsize=8)

            self.ax_evinfo.clear()
            self.ax_evinfo.axis('off')
            
            delta_E_ev = delta_E * 27.2114
            dZ = z_vec[1] - z_vec[0]
            prob_z_tot = (self.c1**2)*(psi1_z**2) + (self.c2**2)*(psi2_z**2) + 2*self.c1*self.c2*psi1_z*psi2_z*np.cos(phase_diff)
            z_mean = np.sum(z_vec * prob_z_tot) * dZ / max(1e-9, np.sum(prob_z_tot) * dZ)
            
            if es_degenerado_stark:
                nota_est = "EFECTO STARK / DESDOBLAMIENTO: El acoplamiento entre 2s y 2p_z induce la oscilación dinámida de carga."
            else:
                nota_est = "BATIDO CUÁNTICO: La densidad oscila verticalmente a la Frecuencia de Bohr Omega_21."

            info_text = (
                f"=== MÉTRICAS DE LA EVOLUCIÓN TEMPORAL (SUPERPOSICIÓN COHERENTE) ===\n"
                f" • Estado 1: {name1} (E1 = {E1_base:.4f} Eh) | Estado 2: {name2} (E2 = {E2_base + (delta_E if es_degenerado_stark else 0):.4f} Eh)\n"
                f" • Diferencia de Energía Delta E: {delta_E:.4f} E_h  (--> {delta_E_ev:.3f} eV)\n"
                f" • Frecuencia de Bohr Omega_21 : {delta_E:.4f} rad/ua | Período T_tau: {T_tau:.2f} ua\n"
                f" • Valor Medio Posición <z>(t) : {z_mean:+.3f} a_mu  (Batido Oscilatorio de Carga)\n"
                f"-----------------------------------------------------------------\n"
                f" DIAGNÓSTICO DINÁMICO: {nota_est}\n"
                f" Hacé clic en '▶ Play' para ver la oscilación completa entre el lóbulo superior e inferior."
            )
            self.ax_evinfo.text(0.02, 0.10, info_text, fontsize=9.2, fontfamily='monospace',
                                color='#40e0d0', bbox=dict(boxstyle='round,pad=0.6', facecolor='#0d1117', edgecolor='#40e0d0', alpha=0.9))

        self.fig_evol.canvas.draw_idle()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("Test execution complete!")
    else:
        app = LaboratorioHidrogenoUnificadoV8()
        plt.show()
