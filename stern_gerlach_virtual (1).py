import sys
import os
import numpy as np
import sympy as sp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

# Forzar Backend interactivo para ejecuciones locales en Tkinter
if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] != '--test'):
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass

plt.style.use('dark_background')

# =============================================================================
# 1. ANÁLISIS SIMBÓLICO Y ÁLGEBRA DE SPIN 1/2 (SYMPY)
# =============================================================================

def run_sympy_analysis():
    """Análisis algebraico con SymPy para Spin 1/2."""
    print("=" * 80)
    print(" ANÁLISIS SIMBÓLICO DE SPIN 1/2 - STERN-GERLACH VIRTUAL")
    print("=" * 80)

    hbar = sp.Symbol('hbar', real=True, positive=True)
    theta, phi = sp.symbols('theta phi', real=True)

    sigma_x = sp.Matrix([[0, 1], [1, 0]])
    sigma_y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    sigma_z = sp.Matrix([[1, 0], [0, -1]])

    Sx = (hbar / 2) * sigma_x
    Sy = (hbar / 2) * sigma_y
    Sz = (hbar / 2) * sigma_z

    print("\n• Matrices de Pauli & Operadores de Spin:")
    print("Sx ="); sp.pprint(Sx)
    print("Sy ="); sp.pprint(Sy)
    print("Sz ="); sp.pprint(Sz)

    comm_xy = sp.simplify(Sx * Sy - Sy * Sx)
    print("\n• Conmutador [Sx, Sy] = i * hbar * Sz:")
    sp.pprint(comm_xy)

    ket_up_theta = sp.cos(theta / 2) * sp.Matrix([1, 0]) + sp.sin(theta / 2) * sp.Matrix([0, 1])
    overlap = (ket_up_theta.H * sp.Matrix([1, 0]))[0]
    prob_up = sp.simplify(sp.Abs(overlap)**2)

    print("\n• Probabilidad P(+theta | +z) = cos^2(theta/2):")
    sp.pprint(sp.Eq(sp.Symbol('P(+theta|+z)'), prob_up))
    print("=" * 80 + "\n")

# =============================================================================
# 2. MOTOR FÍSICO Y ÁLGEBRA DE SPIN 1/2 EN LA PALETA DEL ÁTOMO DE HIDRÓGENO
# =============================================================================

def vector_orientacion(theta, phi=0.0):
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

def probabilidad_born(vec_estado, vec_aparato):
    dot = np.clip(np.dot(vec_estado, vec_aparato), -1.0, 1.0)
    return 0.5 * (1.0 + dot)

# =============================================================================
# 3. INTERFAZ GRÁFICA INTERACTIVA - AULA UNIFICADA STERN-GERLACH
# =============================================================================

class SternGerlachVirtualApp:
    def __init__(self):
        # 1. Configuración de Cascada de Aparatos (1, 2 o 3)
        self.num_apparatus = 2
        
        # 2. Estado Inicial Preparado |psi0>
        self.is_unpolarized = False
        self.theta0 = 0.0  # Default: |+z>
        self.phi0 = 0.0
        
        # Orientación de Aparatos (ángulos respecto a Z)
        self.theta1 = 0.0              # SG1: Eje Z
        self.theta2 = np.pi / 2.0      # SG2: Eje X (90°)
        self.theta3 = 0.0              # SG3: Eje Z
        
        # Filtros de Canal por Aparato ('ambos', 'block_minus', 'block_plus')
        self.filter1 = 'block_minus'
        self.filter2 = 'block_minus'
        self.filter3 = 'ambos'
        
        # Conteos e Impactos 2D Acumulados
        self.total_emitted = 0
        self.hits_sg1 = []
        self.hits_sg2 = []
        self.hits_sg3 = []

        # Estética Átomo de Hidrógeno: Fondo #0b0e14, Contenedores #0d1117
        self.fig = plt.figure(figsize=(16.5, 9.8), facecolor='#0b0e14')
        self.fig.canvas.manager.set_window_title("Laboratorio Virtual 6: Experimento de Stern-Gerlach (Paleta Hidrógeno)")

        # Layout de 4 Subplots Espaciosos
        self.ax_schema = self.fig.add_subplot(2, 2, 1, facecolor='#0d1117')
        self.ax_screen = self.fig.add_subplot(2, 2, 2, facecolor='#0d1117')
        self.ax_bloch = self.fig.add_subplot(2, 2, 3, projection='3d', facecolor='#0d1117')
        self.ax_hist = self.fig.add_subplot(2, 2, 4, facecolor='#0d1117')

        self.fig.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.31, wspace=0.25, hspace=0.38)

        # Construir Controles e Interfaz
        self.crear_controles()

        # Render inicial
        self.actualizar_todo()

    def crear_controles(self):
        """Crea los controles interactivos distribuidos holgadamente en 3 filas."""
        
        # ---------------- FILA 1: Presets de Estado Inicial & Secuencias Históricas (Y = 0.23) ----------------
        presets = [
            ('|+z⟩', 0.0, 0.0, False),
            ('|-z⟩', np.pi, 0.0, False),
            ('|+x⟩', np.pi/2, 0.0, False),
            ('|-x⟩', np.pi/2, np.pi, False),
            ('|+y⟩', np.pi/2, np.pi/2, False),
            ('|-y⟩', np.pi/2, 3*np.pi/2, False),
            ('Mezcla 50/50', 0.0, 0.0, True)
        ]
        self.btns_preset = []
        for idx, (label, th, ph, unpoly) in enumerate(presets):
            ax_b = self.fig.add_axes([0.06 + idx * 0.075, 0.23, 0.07, 0.035], facecolor='#1f2937')
            btn = Button(ax_b, label, color='#1f2937', hovercolor='#374151')
            btn.on_clicked(lambda e, t=th, p=ph, u=unpoly: self.set_estado_inicial(t, p, u))
            self.btns_preset.append(btn)

        secuencias = [
            ('Sec 1: Z -> Z', 2, 0.0, 0.0, 0.0, 'block_minus', 'ambos'),
            ('Sec 2: Z -> X', 2, 0.0, np.pi/2, 0.0, 'block_minus', 'ambos'),
            ('Sec 3: Z -> X -> Z', 3, 0.0, np.pi/2, 0.0, 'block_minus', 'block_minus')
        ]
        self.btns_sec = []
        for idx, (label, n_app, th1, th2, th3, f1, f2) in enumerate(secuencias):
            ax_b = self.fig.add_axes([0.62 + idx * 0.11, 0.23, 0.105, 0.035], facecolor='#1f2937')
            btn = Button(ax_b, label, color='#1f2937', hovercolor='#ff007f')
            btn.on_clicked(lambda e, n=n_app, t1=th1, t2=th2, t3=th3, fl1=f1, fl2=f2: self.ejecutar_secuencia(n, t1, t2, t3, fl1, fl2))
            self.btns_sec.append(btn)

        # ---------------- FILA 2: Selector de Aparatos & Sliders (Y = 0.15) ----------------
        ax_b_app1 = self.fig.add_axes([0.06, 0.15, 0.07, 0.035], facecolor='#1f2937')
        ax_b_app2 = self.fig.add_axes([0.135, 0.15, 0.07, 0.035], facecolor='#1f2937')
        ax_b_app3 = self.fig.add_axes([0.21, 0.15, 0.07, 0.035], facecolor='#1f2937')

        self.btn_app1 = Button(ax_b_app1, '1 Aparato', color='#1f2937', hovercolor='#40e0d0')
        self.btn_app2 = Button(ax_b_app2, '2 Aparatos', color='#40e0d0', hovercolor='#40e0d0')
        self.btn_app3 = Button(ax_b_app3, '3 Aparatos', color='#1f2937', hovercolor='#40e0d0')

        self.btn_app1.on_clicked(lambda e: self.set_num_aparatos(1))
        self.btn_app2.on_clicked(lambda e: self.set_num_aparatos(2))
        self.btn_app3.on_clicked(lambda e: self.set_num_aparatos(3))

        # Sliders Orientaciones
        ax_s_th0 = self.fig.add_axes([0.31, 0.16, 0.16, 0.025], facecolor='#1f2937')
        ax_s_th2 = self.fig.add_axes([0.50, 0.16, 0.16, 0.025], facecolor='#1f2937')
        ax_s_th3 = self.fig.add_axes([0.69, 0.16, 0.16, 0.025], facecolor='#1f2937')

        self.slider_th0 = Slider(ax_s_th0, 'Ángulo θ₀', 0.0, np.pi, valinit=self.theta0, valfmt='%.2f rad', color='#40e0d0')
        self.slider_th2 = Slider(ax_s_th2, 'Eje SG2 θ₂', 0.0, np.pi, valinit=self.theta2, valfmt='%.2f rad', color='#ff007f')
        self.slider_th3 = Slider(ax_s_th3, 'Eje SG3 θ₃', 0.0, np.pi, valinit=self.theta3, valfmt='%.2f rad', color='#00ffff')

        self.slider_th0.on_changed(self.on_slider_th0_change)
        self.slider_th2.on_changed(self.on_slider_th2_change)
        self.slider_th3.on_changed(self.on_slider_th3_change)

        # Botón Reiniciar Conteo
        ax_b_reset = self.fig.add_axes([0.87, 0.14, 0.09, 0.045], facecolor='#ff007f')
        self.btn_reset = Button(ax_b_reset, 'Reiniciar\nConteos', color='#ff007f', hovercolor='#e60073')
        self.btn_reset.on_clicked(self.reset_conteos)

        # ---------------- FILA 3: Emisión de Átomos & Bloqueo de Brazos (Y = 0.05) ----------------
        emisiones = [('+1', 1), ('+100', 100), ('+1000', 1000), ('+5000', 5000)]
        self.btns_emit = []
        for idx, (label, count) in enumerate(emisiones):
            ax_b = self.fig.add_axes([0.06 + idx * 0.065, 0.05, 0.06, 0.04], facecolor='#1f2937')
            btn = Button(ax_b, label, color='#1f2937', hovercolor='#40e0d0')
            btn.on_clicked(lambda e, c=count: self.emitir_atomos(c))
            self.btns_emit.append(btn)

        # Bloqueo de Canales SG1
        ax_f1_ambos = self.fig.add_axes([0.34, 0.05, 0.11, 0.04], facecolor='#1f2937')
        ax_f1_block = self.fig.add_axes([0.46, 0.05, 0.11, 0.04], facecolor='#1f2937')
        self.btn_f1_ambos = Button(ax_f1_ambos, 'SG1: Pasan Ambos', color='#1f2937', hovercolor='#40e0d0')
        self.btn_f1_block = Button(ax_f1_block, 'SG1: Bloquear (-)', color='#40e0d0', hovercolor='#40e0d0')
        self.btn_f1_ambos.on_clicked(lambda e: self.set_filtro(1, 'ambos'))
        self.btn_f1_block.on_clicked(lambda e: self.set_filtro(1, 'block_minus'))

        # Bloqueo de Canales SG2
        ax_f2_ambos = self.fig.add_axes([0.60, 0.05, 0.11, 0.04], facecolor='#1f2937')
        ax_f2_block = self.fig.add_axes([0.72, 0.05, 0.11, 0.04], facecolor='#1f2937')
        self.btn_f2_ambos = Button(ax_f2_ambos, 'SG2: Pasan Ambos', color='#1f2937', hovercolor='#ff007f')
        self.btn_f2_block = Button(ax_f2_block, 'SG2: Bloquear (-)', color='#ff007f', hovercolor='#ff007f')
        self.btn_f2_ambos.on_clicked(lambda e: self.set_filtro(2, 'ambos'))
        self.btn_f2_block.on_clicked(lambda e: self.set_filtro(2, 'block_minus'))

    # -------------------------------------------------------------------------
    # CALLBACKS DE INTERFAZ
    # -------------------------------------------------------------------------
    def set_num_aparatos(self, num):
        self.num_apparatus = num
        self.btn_app1.ax.set_facecolor('#40e0d0' if num == 1 else '#1f2937')
        self.btn_app2.ax.set_facecolor('#40e0d0' if num == 2 else '#1f2937')
        self.btn_app3.ax.set_facecolor('#40e0d0' if num == 3 else '#1f2937')
        self.actualizar_todo()

    def set_estado_inicial(self, th, ph, unpoly):
        self.theta0 = th
        self.phi0 = ph
        self.is_unpolarized = unpoly
        self.slider_th0.set_val(th)
        self.reset_conteos()

    def set_filtro(self, app_num, filtro):
        if app_num == 1:
            self.filter1 = filtro
            self.btn_f1_ambos.ax.set_facecolor('#40e0d0' if filtro == 'ambos' else '#1f2937')
            self.btn_f1_block.ax.set_facecolor('#40e0d0' if filtro == 'block_minus' else '#1f2937')
        elif app_num == 2:
            self.filter2 = filtro
            self.btn_f2_ambos.ax.set_facecolor('#ff007f' if filtro == 'ambos' else '#1f2937')
            self.btn_f2_block.ax.set_facecolor('#ff007f' if filtro == 'block_minus' else '#1f2937')
        self.actualizar_todo()

    def on_slider_th0_change(self, val):
        self.theta0 = float(val)
        self.is_unpolarized = False
        self.actualizar_todo()

    def on_slider_th2_change(self, val):
        self.theta2 = float(val)
        self.actualizar_todo()

    def on_slider_th3_change(self, val):
        self.theta3 = float(val)
        self.actualizar_todo()

    def ejecutar_secuencia(self, num_app, th1, th2, th3, f1, f2):
        self.num_apparatus = num_app
        self.theta0 = 0.0 # |+z>
        self.theta1 = th1
        self.theta2 = th2
        self.theta3 = th3
        self.filter1 = f1
        self.filter2 = f2
        self.is_unpolarized = False
        
        self.btn_app1.ax.set_facecolor('#40e0d0' if num_app == 1 else '#1f2937')
        self.btn_app2.ax.set_facecolor('#40e0d0' if num_app == 2 else '#1f2937')
        self.btn_app3.ax.set_facecolor('#40e0d0' if num_app == 3 else '#1f2937')
        
        self.slider_th0.set_val(0.0)
        self.slider_th2.set_val(th2)
        self.slider_th3.set_val(th3)
        
        self.set_filtro(1, f1)
        self.set_filtro(2, f2)
        
        self.reset_conteos()
        self.emitir_atomos(1000)

    def reset_conteos(self, event=None):
        self.total_emitted = 0
        self.hits_sg1.clear()
        self.hits_sg2.clear()
        self.hits_sg3.clear()
        self.actualizar_todo()

    def emitir_atomos(self, n_atomos):
        """Simula la propagación Monte Carlo de átomos individualmente."""
        self.total_emitted += n_atomos
        
        n1 = vector_orientacion(self.theta1)
        n2 = vector_orientacion(self.theta2)
        n3 = vector_orientacion(self.theta3)

        for _ in range(n_atomos):
            if self.is_unpolarized:
                th_i = 0.0 if np.random.rand() < 0.5 else np.pi
                v_state = vector_orientacion(th_i)
            else:
                v_state = vector_orientacion(self.theta0, self.phi0)

            # SG1
            p1_plus = probabilidad_born(v_state, n1)
            outcome1 = '+' if np.random.rand() < p1_plus else '-'
            
            y_spot1 = 1.0 if outcome1 == '+' else -1.0
            x_spot1 = np.random.normal(0, 0.25)
            y_spot1 += np.random.normal(0, 0.15)
            self.hits_sg1.append((x_spot1, y_spot1, outcome1))

            if outcome1 == '-' and self.filter1 == 'block_minus':
                continue
            if outcome1 == '+' and self.filter1 == 'block_plus':
                continue

            v_state = n1 if outcome1 == '+' else -n1

            if self.num_apparatus == 1:
                continue

            # SG2
            p2_plus = probabilidad_born(v_state, n2)
            outcome2 = '+' if np.random.rand() < p2_plus else '-'
            
            y_spot2 = 1.0 if outcome2 == '+' else -1.0
            x_spot2 = np.random.normal(0, 0.25)
            y_spot2 += np.random.normal(0, 0.15)
            self.hits_sg2.append((x_spot2, y_spot2, outcome2))

            if outcome2 == '-' and self.filter2 == 'block_minus':
                continue
            if outcome2 == '+' and self.filter2 == 'block_plus':
                continue

            v_state = n2 if outcome2 == '+' else -n2

            if self.num_apparatus == 2:
                continue

            # SG3
            p3_plus = probabilidad_born(v_state, n3)
            outcome3 = '+' if np.random.rand() < p3_plus else '-'
            
            y_spot3 = 1.0 if outcome3 == '+' else -1.0
            x_spot3 = np.random.normal(0, 0.25)
            y_spot3 += np.random.normal(0, 0.15)
            self.hits_sg3.append((x_spot3, y_spot3, outcome3))

        self.actualizar_todo()

    def actualizar_todo(self):
        self.dibujar_esquema()
        self.dibujar_pantalla_2d()
        self.dibujar_esfera_bloch_y_bases()
        self.dibujar_histograma()

    # -------------------------------------------------------------------------
    # RENDERING DE SUBPLOTS
    # -------------------------------------------------------------------------
    def dibujar_esquema(self):
        self.ax_schema.clear()
        self.ax_schema.set_facecolor('#0d1117')
        
        self.ax_schema.plot([0, 1.5], [0, 0], color='#40e0d0', lw=2.5, label='Haz Incidente')
        
        # SG1 (Eje Z)
        self.ax_schema.plot([1.5, 2.5], [0.6, -0.6], color='#40e0d0', lw=4)
        self.ax_schema.text(2.0, 0.8, 'SG1 (Z)', color='#40e0d0', ha='center', fontsize=9, fontweight='bold')
        
        self.ax_schema.plot([2.5, 4.0], [0, 0.8], color='#00ffff', lw=2)
        self.ax_schema.plot([2.5, 4.0], [0, -0.8], color='#ff007f', lw=2)
        
        if self.filter1 == 'block_minus':
            self.ax_schema.plot([3.5, 3.5], [-0.4, -1.2], color='#ef4444', lw=6)
            self.ax_schema.text(3.5, -1.4, '[X] Bloqueo (-)', color='#ef4444', ha='center', fontsize=8.5, fontweight='bold')

        if self.num_apparatus >= 2:
            # SG2
            self.ax_schema.plot([4.0, 5.0], [1.4, 0.2], color='#ff007f', lw=4)
            th2_deg = int(np.degrees(self.theta2))
            self.ax_schema.text(4.5, 1.6, f'SG2 ({th2_deg}°)', color='#ff007f', ha='center', fontsize=9, fontweight='bold')
            
            self.ax_schema.plot([5.0, 6.5], [0.8, 1.4], color='#00ffff', lw=2)
            self.ax_schema.plot([5.0, 6.5], [0.8, 0.2], color='#ff007f', lw=2)
            
            if self.filter2 == 'block_minus':
                self.ax_schema.plot([6.0, 6.0], [0.0, 0.4], color='#ef4444', lw=6)
                self.ax_schema.text(6.0, -0.1, '[X] Bloqueo (-)', color='#ef4444', ha='center', fontsize=8.5, fontweight='bold')

        if self.num_apparatus >= 3:
            # SG3
            self.ax_schema.plot([6.5, 7.5], [2.0, 0.8], color='#00ffff', lw=4)
            th3_deg = int(np.degrees(self.theta3))
            self.ax_schema.text(7.0, 2.2, f'SG3 ({th3_deg}°)', color='#00ffff', ha='center', fontsize=9, fontweight='bold')
            
            self.ax_schema.plot([7.5, 9.0], [1.4, 2.0], color='#00ffff', lw=2)
            self.ax_schema.plot([7.5, 9.0], [1.4, 0.8], color='#ff007f', lw=2)

        self.ax_schema.set_xlim([-0.2, 9.5])
        self.ax_schema.set_ylim([-1.8, 2.5])
        self.ax_schema.axis('off')
        self.ax_schema.set_title('Esquema Físico de Cascada de Imanes Stern-Gerlach', color='#40e0d0', fontsize=11, fontweight='bold')
        self.fig.canvas.draw_idle()

    def dibujar_pantalla_2d(self):
        self.ax_screen.clear()
        self.ax_screen.set_facecolor('#0d1117')
        
        if self.num_apparatus == 1:
            hits = self.hits_sg1
            title_sg = "SG1 (Eje Z)"
        elif self.num_apparatus == 2:
            hits = self.hits_sg2
            title_sg = f"SG2 (Eje θ₂={int(np.degrees(self.theta2))}°)"
        else:
            hits = self.hits_sg3
            title_sg = f"SG3 (Eje θ₃={int(np.degrees(self.theta3))}°)"

        if len(hits) > 0:
            xs = [h[0] for h in hits]
            ys = [h[1] for h in hits]
            
            # Scatter con paleta Magma
            self.ax_screen.scatter(xs, ys, c=ys, cmap='magma', s=14, alpha=0.65, edgecolors='none')
            n_plus = sum(1 for h in hits if h[2] == '+')
            n_minus = sum(1 for h in hits if h[2] == '-')
        else:
            n_plus, n_minus = 0, 0

        self.ax_screen.axhline(1.0, color='#00ffff', ls='--', lw=1.2, label=r'Lóbulo $+\hbar/2$')
        self.ax_screen.axhline(-1.0, color='#ff007f', ls='--', lw=1.2, label=r'Lóbulo $-\hbar/2$')
        self.ax_screen.axhline(0, color='gray', ls=':', alpha=0.5)

        self.ax_screen.set_xlim([-1.5, 1.5])
        self.ax_screen.set_ylim([-2.2, 2.2])
        self.ax_screen.set_xlabel(r'Coordenada Horizontal $x$ [mm]', fontsize=9)
        self.ax_screen.set_ylabel(r'Deflexión Magnética $z$ [mm]', fontsize=9)
        self.ax_screen.set_title(f"Pantalla Sensora 2D de Impactos — {title_sg}\nTotal Medidos: {len(hits)} (|+⟩: {n_plus} | |-⟩: {n_minus})", color='#40e0d0', fontsize=10, fontweight='bold')
        self.ax_screen.grid(True, alpha=0.2)
        self.ax_screen.legend(loc='upper right', fontsize=8)
        self.fig.canvas.draw_idle()

    def dibujar_esfera_bloch_y_bases(self):
        self.ax_bloch.clear()
        self.ax_bloch.set_facecolor('#0d1117')

        # Esfera de Bloch
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x_s = np.outer(np.cos(u), np.sin(v))
        y_s = np.outer(np.sin(u), np.sin(v))
        z_s = np.outer(np.ones(np.size(u)), np.cos(v))
        self.ax_bloch.plot_wireframe(x_s, y_s, z_s, color='#1f2937', alpha=0.35, lw=0.5)

        # Ejes
        self.ax_bloch.plot([-1.2, 1.2], [0, 0], [0, 0], color='gray', ls='--', lw=1)
        self.ax_bloch.plot([0, 0], [-1.2, 1.2], [0, 0], color='gray', ls='--', lw=1)
        self.ax_bloch.plot([0, 0], [0, 0], [-1.2, 1.2], color='gray', ls='--', lw=1)
        self.ax_bloch.text(1.3, 0, 0, '+X', color='white', fontsize=8)
        self.ax_bloch.text(0, 0, 1.3, '+Z', color='white', fontsize=8)

        # Vector de Estado |psi0>
        if not self.is_unpolarized:
            v0 = vector_orientacion(self.theta0, self.phi0)
            self.ax_bloch.quiver(0, 0, 0, v0[0], v0[1], v0[2], color='#00ffff', lw=2.5, arrow_length_ratio=0.15)
            self.ax_bloch.text(v0[0]*1.1, v0[1]*1.1, v0[2]*1.1, r'$|\psi_0\rangle$', color='#00ffff', fontweight='bold', fontsize=10)

        # Vector del Aparato Activo (SG2 o SG3)
        if self.num_apparatus >= 2:
            v2 = vector_orientacion(self.theta2)
            self.ax_bloch.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='#ff007f', lw=2.5, arrow_length_ratio=0.15)
            self.ax_bloch.text(v2[0]*1.1, v2[1]*1.1, v2[2]*1.1, r'$\hat{n}_2$', color='#ff007f', fontweight='bold', fontsize=10)

        self.ax_bloch.set_xlim([-1.2, 1.2])
        self.ax_bloch.set_ylim([-1.2, 1.2])
        self.ax_bloch.set_zlim([-1.2, 1.2])
        self.ax_bloch.set_box_aspect([1, 1, 1])
        self.ax_bloch.set_title("Esfera de Bloch 3D & Cambio de Base", color='#40e0d0', fontsize=10, fontweight='bold')
        self.fig.canvas.draw_idle()

    def dibujar_histograma(self):
        self.ax_hist.clear()
        self.ax_hist.set_facecolor('#0d1117')

        if self.num_apparatus == 1:
            hits = self.hits_sg1
            vec_app = vector_orientacion(self.theta1)
            title_sg = "Frecuencias finales en SG1"
        elif self.num_apparatus == 2:
            hits = self.hits_sg2
            vec_app = vector_orientacion(self.theta2)
            title_sg = "Frecuencias finales en SG2"
        else:
            hits = self.hits_sg3
            vec_app = vector_orientacion(self.theta3)
            title_sg = "Frecuencias finales en SG3"

        total_hits = len(hits)
        n_plus = sum(1 for h in hits if h[2] == '+')
        n_minus = sum(1 for h in hits if h[2] == '-')

        freq_plus = n_plus / max(1, total_hits)
        freq_minus = n_minus / max(1, total_hits)

        if self.is_unpolarized:
            prob_p, prob_m = 0.5, 0.5
        else:
            v0 = vector_orientacion(self.theta0, self.phi0)
            prob_p = probabilidad_born(v0, vec_app)
            prob_m = 1.0 - prob_p

        categories = ['|+⟩ (Spin Up)', '|-⟩ (Spin Down)']
        x_pos = np.array([0, 1])
        width = 0.35

        rects1 = self.ax_hist.bar(x_pos - width/2, [freq_plus, freq_minus], width, label='Frecuencia Empírica', color='#40e0d0', alpha=0.85, edgecolor='white')
        rects2 = self.ax_hist.bar(x_pos + width/2, [prob_p, prob_m], width, label='Teoría Born', color='#ff007f', alpha=0.5, edgecolor='#ff007f', ls='--')

        self.ax_hist.set_xticks(x_pos)
        self.ax_hist.set_xticklabels(categories, color='white', fontsize=9)
        self.ax_hist.set_ylabel('Probabilidad / Frecuencia', color='white', fontsize=9)
        self.ax_hist.set_ylim([0, 1.15])
        self.ax_hist.set_title(f"{title_sg} (N_medidos = {total_hits})", color='#40e0d0', fontsize=10, fontweight='bold')
        self.ax_hist.grid(True, alpha=0.2, axis='y')
        self.ax_hist.legend(loc='upper right', fontsize=8)

        for rect in rects1:
            h = rect.get_height()
            self.ax_hist.text(rect.get_x() + rect.get_width()/2.0, h + 0.02, f'{h*100:.1f}%', ha='center', va='bottom', color='white', fontsize=8.5, fontweight='bold')

        self.fig.canvas.draw_idle()

if __name__ == '__main__':
    run_sympy_analysis()
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("Test execution complete!")
    else:
        app = SternGerlachVirtualApp()
        plt.show()
