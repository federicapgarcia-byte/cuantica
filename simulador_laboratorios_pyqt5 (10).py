import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QSlider, QLabel, QPushButton, 
                             QGroupBox, QRadioButton, QButtonGroup, QFrame, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette

"""
===============================================================================
BITÁCORA INTERACTIVA DE LABORATORIOS DE MECÁNICA CUÁNTICA (PyQt5)
Paleta de Diseño: Viridis Theme 🌿 (Violeta Oscuro, Turquesa, Esmeralda, Amarillo Vívido)
===============================================================================
"""

# QSS Stylesheet para la interfaz PyQt5 en Paleta Viridis
VIRIDIS_QSS = """
QMainWindow {
    background-color: #0d0826;
}
QWidget {
    background-color: #0d0826;
    color: #e0f3f8;
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #31688e;
    background: #130f30;
    border-radius: 8px;
}
QTabBar::tab {
    background: #18123c;
    color: #90b8a0;
    padding: 10px 20px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: bold;
    margin-right: 4px;
}
QTabBar::tab:selected {
    background: #21918c;
    color: #fde725;
    border-bottom: 3px solid #fde725;
}
QTabBar::tab:hover {
    background: #221a4c;
    color: #35b779;
}
QGroupBox {
    background-color: #130f30;
    border: 1px solid #31688e;
    border-radius: 8px;
    margin-top: 12px;
    font-weight: bold;
    color: #35b779;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #fde725;
}
QLabel {
    background-color: transparent;
    color: #e0f3f8;
}
QSlider::groove:horizontal {
    border: 1px solid #31688e;
    height: 6px;
    background: #18123c;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #21918c, stop:1 #fde725);
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #35b779;
    border: 2px solid #ffffff;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #fde725;
    border: 2px solid #35b779;
}
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #21918c, stop:1 #35b779);
    color: #0d0826;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    font-weight: bold;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #35b779, stop:1 #fde725);
}
QPushButton:pressed {
    background-color: #1f9e89;
}
QRadioButton {
    background-color: transparent;
    color: #e0f3f8;
    spacing: 6px;
}
QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border-radius: 7px;
    border: 2px solid #31688e;
    background-color: #18123c;
}
QRadioButton::indicator:checked {
    background-color: #fde725;
    border: 2px solid #35b779;
}
"""

class Lab1Canvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(7, 5), facecolor='#0d0826')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#130f30')
        super().__init__(self.fig)
        self.setParent(parent)
        
    def plot_sim(self, profile_type, alpha, n0_ref, num_rays):
        self.ax.clear()
        self.ax.set_facecolor('#130f30')
        
        x_min, x_max = 0.0, 10.0
        y_min, y_max = -5.0, 5.0
        x_src, y_src = 0.5, 0.0
        
        angles = np.linspace(-np.pi/3.4, np.pi/3.4, num_rays)
        X_grid, Y_grid = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        
        if profile_type == 'lineal':
            N_field = np.maximum(n0_ref + alpha * Y_grid, 0.05)
            title_str = rf"Medio Lineal: $n(y) = n_0 + \alpha y$ ($\alpha = {alpha:+.4f}$)"
        else:
            N_field = np.maximum(n0_ref - alpha * (Y_grid**2), 0.05)
            title_str = rf"Guía GRIN Cuadrática: $n(y) = n_0 - \alpha y^2$ ($\alpha = {alpha:+.4f}$)"
            
        im = self.ax.imshow(N_field, extent=[x_min, x_max, y_min, y_max], origin='lower',
                             cmap='viridis', alpha=0.6, aspect='auto')
        
        contours = self.ax.contour(X_grid, Y_grid, N_field, levels=7, colors='#35b779', linewidths=0.7, alpha=0.6)
        self.ax.clabel(contours, inline=True, fontsize=8, fmt='n=%.2f', colors='#e0f3f8')
        
        def ODE_system(s, state):
            x, y, px, py, L = state
            if profile_type == 'lineal':
                n = max(n0_ref + alpha * y, 0.05)
                dn_dy = alpha
            else:
                n = max(n0_ref - alpha * (y**2), 0.05)
                dn_dy = -2.0 * alpha * y
            return [px / n, py / n, 0.0, dn_dy, n]
            
        def boundary_event(s, state):
            x, y, px, py, L = state
            if x <= x_min + 0.001 or x >= x_max - 0.001 or y <= y_min + 0.001 or y >= y_max - 0.001:
                return -1.0
            return 1.0
        boundary_event.terminal = True
        
        s_span = (0, 30)
        n_src = max(n0_ref + alpha * y_src, 0.05) if profile_type == 'lineal' else max(n0_ref - alpha * (y_src**2), 0.05)
            
        ray_data = []
        for theta in angles:
            px0 = n_src * np.cos(theta)
            py0 = n_src * np.sin(theta)
            sol = solve_ivp(ODE_system, s_span, [x_src, y_src, px0, py0, 0.0],
                            events=boundary_event, dense_output=True, max_step=0.05)
            rx, ry = sol.y[0], sol.y[1]
            self.ax.plot(rx, ry, color='#35b779', linewidth=1.4, alpha=0.9)
            ray_data.append((sol, sol.y[4][-1]))
            
        max_lopt = min(r_max for _, r_max in ray_data)
        if max_lopt > 1.0:
            for L_target in np.linspace(1.2, max_lopt * 0.95, 12):
                wf_x, wf_y = [], []
                for sol, r_max in ray_data:
                    if r_max >= L_target:
                        s_target = np.interp(L_target, sol.y[4], sol.t)
                        pt = sol.sol(s_target)
                        wf_x.append(pt[0]); wf_y.append(pt[1])
                if len(wf_x) > 3:
                    self.ax.plot(wf_x, wf_y, color='#fde725', linestyle='--', linewidth=1.5, alpha=0.95)
                    
        self.ax.plot([], [], color='#fde725', linestyle='--', linewidth=1.5, label=r'Frentes de onda ($\Phi=\text{cte}$)')
        self.ax.plot([], [], color='#35b779', linewidth=1.5, label='Rayos ópticos')
        self.ax.plot(x_src, y_src, 'o', color='#fde725', markersize=8, markeredgecolor='#ffffff', label='Fuente')
        
        self.ax.set_xlim(x_min, x_max); self.ax.set_ylim(y_min, y_max)
        self.ax.set_title(title_str, color='#fde725', fontsize=12, fontweight='bold', pad=10)
        self.ax.set_xlabel('Posición $x$', color='#e0f3f8'); self.ax.set_ylabel('Posición $y$', color='#e0f3f8')
        self.ax.tick_params(colors='#b5de2b')
        for spine in self.ax.spines.values(): spine.set_color('#31688e')
        self.ax.grid(True, linestyle=':', color='#2c1e55', alpha=0.6)
        self.ax.legend(loc='upper right', facecolor='#18123c', edgecolor='#31688e', labelcolor='#e0f3f8', fontsize=9)
        self.draw()


class Lab2Canvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(7, 6.5), facecolor='#0d0826')
        self.ax_complex = self.fig.add_subplot(311)
        self.ax_prob = self.fig.add_subplot(312)
        self.ax_k = self.fig.add_subplot(313)
        self.fig.tight_layout(pad=2.5)
        super().__init__(self.fig)
        self.setParent(parent)
        
    def plot_wavepacket(self, N, k0, sigma_k, t):
        self.ax_complex.clear(); self.ax_prob.clear(); self.ax_k.clear()
        for ax in (self.ax_complex, self.ax_prob, self.ax_k):
            ax.set_facecolor('#130f30')
            ax.tick_params(colors='#b5de2b')
            for spine in ax.spines.values(): spine.set_color('#31688e')
            ax.grid(True, linestyle=':', color='#2c1e55', alpha=0.6)
            
        x = np.linspace(-30.0, 30.0, 800)
        dx = x[1] - x[0]
        
        if N == 1:
            k_vals = np.array([float(k0)]); A_vals = np.array([1.0])
        else:
            k_min = max(0.05, k0 - 3.5 * max(sigma_k, 0.1))
            k_max = k0 + 3.5 * max(sigma_k, 0.1)
            k_vals = np.linspace(k_min, k_max, int(N))
            A_vals = np.exp(-(k_vals - k0)**2 / (4.0 * max(sigma_k, 0.05)**2))
            
        omega_vals = (k_vals**2) / 2.0
        phases = k_vals[:, np.newaxis] * x[np.newaxis, :] - omega_vals[:, np.newaxis] * t
        components = A_vals[:, np.newaxis] * np.exp(1j * phases)
        psi = np.sum(components, axis=0)
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
        if norm > 1e-12: psi = psi / norm
            
        prob_x = np.abs(psi)**2
        mean_x = np.sum(x * prob_x) * dx
        std_x = np.sqrt(max(np.sum((x - mean_x)**2 * prob_x) * dx, 1e-12))
        
        prob_k = A_vals**2
        norm_k = np.sum(prob_k)
        if norm_k > 1e-12:
            prob_k /= norm_k
            mean_k = np.sum(k_vals * prob_k)
            std_k = np.sqrt(max(np.sum((k_vals - mean_k)**2 * prob_k), 1e-12))
        else:
            mean_k, std_k = k0, 0.0
            
        v_fase = k0 / 2.0; v_grupo = k0; x_c = v_grupo * t
        
        # Panel 1: Complejo
        self.ax_complex.plot(x, np.real(psi), color='#35b779', linewidth=1.3, label=r'$\mathrm{Re}(\psi)$')
        self.ax_complex.plot(x, np.imag(psi), color='#21918c', linestyle=':', linewidth=1.2, alpha=0.85, label=r'$\mathrm{Im}(\psi)$')
        self.ax_complex.plot(x, np.abs(psi), color='#fde725', linestyle='--', linewidth=1.7, label=r'Envolvente $|\psi|$')
        self.ax_complex.set_xlim(-30, 30); self.ax_complex.set_ylabel(r'Amplitud $\psi$', color='#e0f3f8', fontsize=9)
        self.ax_complex.set_title(f'Función de Onda Compleja | $t = {t:.2f}$', color='#fde725', fontsize=11, fontweight='bold')
        self.ax_complex.legend(loc='upper right', facecolor='#18123c', edgecolor='#31688e', labelcolor='#e0f3f8', fontsize=8)
        
        # Panel 2: Probabilidad
        self.ax_prob.fill_between(x, prob_x, color='#21918c', alpha=0.35)
        self.ax_prob.plot(x, prob_x, color='#35b779', linewidth=2.0, label=r'$|\psi(x,t)|^2$')
        self.ax_prob.axvline(x_c, color='#fde725', linestyle='--', linewidth=1.4, label=f'Centro $v_g t = {x_c:.1f}$')
        self.ax_prob.set_xlim(-30, 30); self.ax_prob.set_ylabel(r'Densidad $|\psi|^2$', color='#e0f3f8', fontsize=9)
        self.ax_prob.legend(loc='upper right', facecolor='#18123c', edgecolor='#31688e', labelcolor='#e0f3f8', fontsize=8)
        self.ax_prob.set_title(rf"$\Delta x = {std_x:.2f}$ | $\Delta k = {std_k:.2f}$ | $\mathbf{{\Delta x \cdot \Delta k = {std_x*std_k:.3f}}}$", color='#fde725', fontsize=9, pad=6)
        
        # Panel 3: Espacio k
        if N == 1:
            self.ax_k.stem(k_vals, A_vals, linefmt='#35b779', markerfmt='o', basefmt='k-')
        else:
            markerline, stemlines, _ = self.ax_k.stem(k_vals, A_vals, linefmt='#21918c', markerfmt='o', basefmt='k-', label='Amplitudes $A_j$')
            plt.setp(markerline, color='#fde725', markersize=4)
            plt.setp(stemlines, color='#35b779', linewidth=1.2)
            k_fine = np.linspace(k0 - 4*sigma_k, k0 + 4*sigma_k, 200)
            A_fine = np.exp(-(k_fine - k0)**2 / (4.0 * max(sigma_k, 0.05)**2))
            self.ax_k.plot(k_fine, A_fine, color='#fde725', linestyle='--', alpha=0.8, label='Perfil $A(k)$')
            
        self.ax_k.axvline(k0, color='#fde725', linestyle='--', linewidth=1.2, label=f'$k_0 = {k0:.2f}$')
        self.ax_k.set_xlabel(r'Número de Onda $k$', color='#e0f3f8', fontsize=9)
        self.ax_k.set_ylabel(r'Amplitud $A(k)$', color='#e0f3f8', fontsize=9)
        self.ax_k.set_xlim(0, 7)
        self.ax_k.legend(loc='upper right', facecolor='#18123c', edgecolor='#31688e', labelcolor='#e0f3f8', fontsize=8)
        self.draw()


class Lab3Canvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(7, 6.5), facecolor='#0d0826')
        self.ax_surface = self.fig.add_subplot(221, projection='3d')
        self.ax_prob = self.fig.add_subplot(222)
        self.ax_state = self.fig.add_subplot(223)
        self.ax_det = self.fig.add_subplot(224)
        self.fig.tight_layout(pad=2.2)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Precomputación
        self.L, self.N_grid, self.hbar, self.m = 1.0, 100, 1.0, 1.0
        self.dx = self.L / (self.N_grid + 1)
        self.x = np.linspace(self.dx, self.L - self.dx, self.N_grid)
        
        coeff = (self.hbar**2) / (2.0 * self.m * self.dx**2)
        H = np.diag(2.0*np.ones(self.N_grid)*coeff) + np.diag(-1.0*np.ones(self.N_grid-1)*coeff, 1) + np.diag(-1.0*np.ones(self.N_grid-1)*coeff, -1)
        evals, evecs = np.linalg.eigh(H)
        
        self.phi1 = evecs[:, 0] / np.sqrt(np.sum(evecs[:, 0]**2) * self.dx)
        self.phi2 = evecs[:, 1] / np.sqrt(np.sum(evecs[:, 1]**2) * self.dx)
        if self.phi1[20] < 0: self.phi1 = -self.phi1
        if self.phi2[20] < 0: self.phi2 = -self.phi2
            
        self.E1, self.E2 = evals[0], evals[1]
        self.omega_21 = (self.E2 - self.E1) / self.hbar
        self.T_per = 2.0 * np.pi / self.omega_21
        
        self.t_mesh = np.linspace(0, 1.2 * self.T_per, 50)
        self.X_grid, self.T_grid = np.meshgrid(self.x, self.t_mesh)
        
    def plot_lab3(self, theta, delta, chi, xD, t_curr):
        self.ax_surface.clear(); self.ax_prob.clear(); self.ax_state.clear(); self.ax_det.clear()
        for ax in (self.ax_prob, self.ax_state, self.ax_det):
            ax.set_facecolor('#130f30'); ax.tick_params(colors='#b5de2b')
            for spine in ax.spines.values(): spine.set_color('#31688e')
            ax.grid(True, linestyle=':', color='#2c1e55', alpha=0.6)
            
        self.ax_surface.set_facecolor('#0d0826')
        
        c1, c2 = np.cos(theta), np.sin(theta)
        
        # Malla 3D de probabilidad Z(x,t)
        P_mesh = (c1**2)*(self.phi1[np.newaxis, :]**2) + (c2**2)*(self.phi2[np.newaxis, :]**2) + \
                 2.0*c1*c2*np.outer(np.cos(delta - self.omega_21*self.t_mesh), self.phi1*self.phi2)
                 
        surf = self.ax_surface.plot_surface(self.X_grid, self.T_grid, P_mesh, cmap='viridis', alpha=0.85, edgecolor='none')
        
        # Estado actual
        psi1_t = c1 * self.phi1 * np.exp(-1j * self.E1 * t_curr)
        psi2_t = c2 * self.phi2 * np.exp(1j * delta - 1j * self.E2 * t_curr)
        psi_t = np.exp(1j * chi) * (psi1_t + psi2_t)
        prob_t = np.abs(psi_t)**2
        
        self.ax_surface.plot(self.x, np.full_like(self.x, t_curr), prob_t, color='#fde725', linewidth=2.5)
        self.ax_surface.set_title(r'Superficie 3D $|\psi(x,t)|^2$', color='#fde725', fontsize=10, fontweight='bold')
        self.ax_surface.set_xlabel('x', color='#e0f3f8'); self.ax_surface.set_ylabel('t', color='#e0f3f8')
        
        # Panel 2D Probabilidad
        self.ax_prob.fill_between(self.x, prob_t, color='#21918c', alpha=0.35)
        self.ax_prob.plot(self.x, prob_t, color='#fde725', linewidth=2.0, label=r'$|\psi(x,t)|^2$')
        self.ax_prob.axvline(xD, color='#35b779', linestyle='--', linewidth=1.5, label=f'Detector $x_D={xD:.2f}$')
        self.ax_prob.set_xlim(0, self.L); self.ax_prob.set_title(f'Densidad Probabilidad ($t={t_curr:.2f}$)', color='#fde725', fontsize=10, fontweight='bold')
        self.ax_prob.legend(loc='upper right', facecolor='#18123c', edgecolor='#31688e', labelcolor='#e0f3f8', fontsize=8)
        
        # Panel Autoestados
        self.ax_state.plot(self.x, self.phi1, color='#35b779', linewidth=1.6, label=r'$\phi_1(x)$')
        self.ax_state.plot(self.x, self.phi2, color='#21918c', linewidth=1.6, label=r'$\phi_2(x)$')
        self.ax_state.set_xlim(0, self.L); self.ax_state.set_title(rf'Autoestados ($\Omega_{{21}} = {self.omega_21:.2f}$ rad/s)', color='#e0f3f8', fontsize=10, fontweight='bold')
        self.ax_state.legend(loc='upper right', facecolor='#18123c', edgecolor='#31688e', labelcolor='#e0f3f8', fontsize=8)
        
        # Panel Detector Virtual
        t_arr = np.linspace(0, 1.5 * self.T_per, 200)
        idx_D = int(np.clip(xD / self.dx, 0, self.N_grid - 1))
        phi1_D, phi2_D = self.phi1[idx_D], self.phi2[idx_D]
        sig_arr = (c1**2)*(phi1_D**2) + (c2**2)*(phi2_D**2) + 2.0*c1*c2*phi1_D*phi2_D*np.cos(delta - self.omega_21*t_arr)
        sig_curr = (c1**2)*(phi1_D**2) + (c2**2)*(phi2_D**2) + 2.0*c1*c2*phi1_D*phi2_D*np.cos(delta - self.omega_21*t_curr)
        
        self.ax_det.plot(t_arr, sig_arr, color='#35b779', linewidth=1.8, label=r'$I_D(t)$')
        self.ax_det.plot(t_curr, sig_curr, 'o', color='#fde725', markersize=8, label=f't={t_curr:.2f}')
        self.ax_det.set_title(rf'Detector Virtual $I_D(t)$ en $x_D={xD:.2f}$', color='#fde725', fontsize=10, fontweight='bold')
        self.ax_det.legend(loc='upper right', facecolor='#18123c', edgecolor='#31688e', labelcolor='#e0f3f8', fontsize=8)
        self.draw()


class Lab4Canvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(7, 6.5), facecolor='#0d0826')
        self.ax_screen2d = self.fig.add_subplot(221)
        self.ax_theory = self.fig.add_subplot(222)
        self.ax_histo = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=2.2)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.y_max = 12.0
        self.num_bins = 120
        self.y_grid = np.linspace(-self.y_max, self.y_max, 1000)
        self.dy = self.y_grid[1] - self.y_grid[0]
        self.D_screen = 1000.0

    def compute_components(self, y, d, a, wavelength):
        sin_theta = y / self.D_screen
        k = 2.0 * np.pi / (wavelength * 1e-3)
        beta = (k * a / 2.0) * sin_theta
        sinc_factor = np.sinc(beta / np.pi)
        alpha = (k * d / 2.0) * sin_theta
        
        psi1_sq = 0.5 * (sinc_factor**2)
        psi2_sq = 0.5 * (sinc_factor**2)
        prob_total = (sinc_factor**2) * (np.cos(alpha)**2)
        return psi1_sq, psi2_sq, prob_total

    def compute_probability(self, y, d, a, wavelength, which_way):
        p1, p2, p_tot = self.compute_components(y, d, a, wavelength)
        prob = p_tot if not which_way else p1 + p2
        norm = np.sum(prob) * self.dy
        if norm > 1e-12: prob /= norm
        return prob

    def plot_lab4(self, d, a, wavelength, which_way_mode, detected_events):
        self.ax_screen2d.clear(); self.ax_theory.clear(); self.ax_histo.clear()
        for ax in (self.ax_screen2d, self.ax_theory, self.ax_histo):
            ax.set_facecolor('#130f30'); ax.tick_params(colors='#b5de2b')
            for spine in ax.spines.values(): spine.set_color('#31688e')
            ax.grid(True, linestyle=':', color='#2c1e55', alpha=0.6)
            
        p1_sq, p2_sq, p_tot = self.compute_components(self.y_grid, d, a, wavelength)
        prob_theory = self.compute_probability(self.y_grid, d, a, wavelength, which_way_mode)
        
        norm_factor = np.sum(p_tot) * self.dy
        p1_norm = p1_sq / norm_factor if norm_factor > 1e-12 else p1_sq
        p2_norm = p2_sq / norm_factor if norm_factor > 1e-12 else p2_sq
        
        N_total = len(detected_events)
        
        # 1. Pantalla 2D
        self.ax_screen2d.axhline(d/2.0, color='#21918c', linestyle=':', linewidth=1.2, alpha=0.7)
        self.ax_screen2d.axhline(-d/2.0, color='#35b779', linestyle=':', linewidth=1.2, alpha=0.7)
        if N_total > 0:
            events_arr = np.array(detected_events)
            display_events = events_arr[-1500:]
            x_screen = np.random.uniform(-0.8, 0.8, size=len(display_events))
            self.ax_screen2d.scatter(x_screen, display_events, c='#fde725', s=6, alpha=0.7, edgecolors='none')
            
        self.ax_screen2d.set_xlim(-1.2, 1.2); self.ax_screen2d.set_ylim(-self.y_max, self.y_max)
        self.ax_screen2d.set_xticks([]); self.ax_screen2d.set_ylabel('Posición $y$ (mm)', fontsize=9, color='#e0f3f8')
        self.ax_screen2d.set_title(rf'Pantalla 2D | $N = {N_total}$ Clics', fontsize=10, fontweight='bold', color='#fde725', pad=6)
        
        # 2. Distribución Teórica
        mode_str = "Coherente (Interferencia)" if not which_way_mode else "Which-Way (Incoherente)"
        self.ax_theory.plot(p1_norm, self.y_grid, color='#1f9e89', linestyle='--', linewidth=1.4, alpha=0.85, label=r'$|\psi_1|^2$')
        self.ax_theory.plot(p2_norm, self.y_grid, color='#35b779', linestyle=':', linewidth=1.4, alpha=0.85, label=r'$|\psi_2|^2$')
        
        main_color = '#fde725' if not which_way_mode else '#31688e'
        self.ax_theory.plot(prob_theory, self.y_grid, color=main_color, linewidth=2.2, label=r'Total Born $P(y)$')
        self.ax_theory.fill_betweenx(self.y_grid, 0, prob_theory, color=main_color, alpha=0.25)
        self.ax_theory.set_xlim(0, np.max(prob_theory) * 1.15); self.ax_theory.set_ylim(-self.y_max, self.y_max)
        self.ax_theory.set_xlabel('Densidad $P(y)$', fontsize=9, color='#e0f3f8')
        self.ax_theory.set_title(f'Contribuciones Teóricas - [{mode_str}]', fontsize=10, fontweight='bold', color=main_color, pad=6)
        self.ax_theory.legend(loc='upper right', fontsize=8, facecolor='#18123c', edgecolor='#31688e', labelcolor='#e0f3f8')
        
        # 3. Histograma
        if N_total > 0:
            counts, bin_edges = np.histogram(detected_events, bins=self.num_bins, range=(-self.y_max, self.y_max))
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_width = bin_edges[1] - bin_edges[0]
            density_hist = counts / (N_total * bin_width)
            self.ax_histo.bar(bin_centers, density_hist, width=bin_width*0.9, color='#35b779', alpha=0.6,
                              edgecolor='#fde725', label=rf'Histograma ($N = {N_total}$)')
        else:
            self.ax_histo.plot([], [], color='#35b779', label='Esperando disparos...')
            
        self.ax_histo.plot(self.y_grid, prob_theory, color='#fde725', linewidth=2.0, linestyle='--', label=r'Predicción $P(y)$')
        self.ax_histo.set_xlim(-self.y_max, self.y_max)
        self.ax_histo.set_xlabel('Posición en Pantalla $y$ (mm)', fontsize=10, color='#e0f3f8')
        self.ax_histo.set_ylabel('Frecuencia Relativa', fontsize=10, color='#e0f3f8')
        
        status_text = rf'$d = {d:.2f}$ mm  |  $a = {a:.2f}$ mm  |  $\lambda = {wavelength:.2f}$ µm'
        self.ax_histo.set_title(f'Emergencia del Patrón Evento a Evento (Paleta Viridis) — {status_text}', 
                                fontsize=10, fontweight='bold', color='#fde725',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='#18123c', edgecolor='#31688e', alpha=0.9))
        self.ax_histo.legend(loc='upper right', fontsize=8.5, facecolor='#18123c', edgecolor='#31688e', labelcolor='#e0f3f8')
        self.draw()


class QuantumLabMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bitácora de Laboratorios de Mecánica Cuántica — Paleta Viridis 🌿")
        self.resize(1180, 850)
        
        # Inicialización de estado para Lab 4
        self.lab4_events = []
        self.which_way4 = False
        self.is_running4 = False
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        title_label = QLabel("LABORATORIOS COMPUTACIONALES DE MECÁNICA CUÁNTICA")
        title_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title_label.setStyleSheet("color: #fde725; margin-bottom: 4px;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Configuración de Pestañas
        self.tab1 = QWidget(); self.setup_tab1(); self.tabs.addTab(self.tab1, "🌿 Lab 1: Óptica Geométrica n(y)")
        self.tab2 = QWidget(); self.setup_tab2(); self.tabs.addTab(self.tab2, "🌿 Lab 2: Ondas de de Broglie")
        self.tab3 = QWidget(); self.setup_tab3(); self.tabs.addTab(self.tab3, "🌿 Lab 3: Dos Estados y Fase")
        self.tab4 = QWidget(); self.setup_tab4(); self.tabs.addTab(self.tab4, "🌿 Lab 4: Doble Rendija 2D")
        
        # Timers de animación
        self.anim_timer2 = QTimer()
        self.anim_timer2.setInterval(40)
        self.anim_timer2.timeout.connect(self.advance_time2)
        self.is_playing2 = False
        
        self.anim_timer3 = QTimer()
        self.anim_timer3.setInterval(40)
        self.anim_timer3.timeout.connect(self.advance_time3)
        self.is_playing3 = False
        
        self.anim_timer4 = QTimer()
        self.anim_timer4.setInterval(40)
        self.anim_timer4.timeout.connect(self.advance_fire4)

    def setup_tab1(self):
        layout = QHBoxLayout(self.tab1)
        control_panel = QWidget(); control_panel.setMaximumWidth(320)
        control_layout = QVBoxLayout(control_panel)
        
        grp_profile = QGroupBox("Perfil de Índice n(y)")
        grp_layout = QVBoxLayout(grp_profile)
        self.radio_linear = QRadioButton("Lineal: n(y) = n₀ + α y")
        self.radio_quad = QRadioButton("Cuadrático: n(y) = n₀ - α y² (GRIN)")
        self.radio_linear.setChecked(True)
        grp_layout.addWidget(self.radio_linear); grp_layout.addWidget(self.radio_quad)
        control_layout.addWidget(grp_profile)
        
        grp_params = QGroupBox("Parámetros Físicos")
        params_layout = QGridLayout(grp_params)
        params_layout.addWidget(QLabel("Gradiente α:"), 0, 0, 1, 2)
        self.lbl_alpha = QLabel("+0.030"); self.lbl_alpha.setStyleSheet("color: #fde725; font-weight: bold;")
        params_layout.addWidget(self.lbl_alpha, 0, 2)
        self.slider_alpha = QSlider(Qt.Horizontal); self.slider_alpha.setRange(-80, 80); self.slider_alpha.setValue(30)
        params_layout.addWidget(self.slider_alpha, 1, 0, 1, 3)
        
        params_layout.addWidget(QLabel("Índice Base n₀:"), 2, 0, 1, 2)
        self.lbl_n0 = QLabel("1.50"); self.lbl_n0.setStyleSheet("color: #fde725; font-weight: bold;")
        params_layout.addWidget(self.lbl_n0, 2, 2)
        self.slider_n0 = QSlider(Qt.Horizontal); self.slider_n0.setRange(100, 250); self.slider_n0.setValue(150)
        params_layout.addWidget(self.slider_n0, 3, 0, 1, 3)
        
        params_layout.addWidget(QLabel("Número de Rayos:"), 4, 0, 1, 2)
        self.lbl_rays = QLabel("21"); self.lbl_rays.setStyleSheet("color: #fde725; font-weight: bold;")
        params_layout.addWidget(self.lbl_rays, 4, 2)
        self.slider_rays = QSlider(Qt.Horizontal); self.slider_rays.setRange(5, 41); self.slider_rays.setValue(21)
        params_layout.addWidget(self.slider_rays, 5, 0, 1, 3)
        control_layout.addWidget(grp_params)
        
        btn_reset1 = QPushButton("Medio Homogéneo (α = 0)")
        btn_reset1.clicked.connect(lambda: self.slider_alpha.setValue(0))
        control_layout.addWidget(btn_reset1)
        control_layout.addStretch()
        layout.addWidget(control_panel)
        
        self.canvas1 = Lab1Canvas(self.tab1)
        layout.addWidget(self.canvas1)
        
        self.radio_linear.toggled.connect(self.update_lab1)
        self.radio_quad.toggled.connect(self.update_lab1)
        self.slider_alpha.valueChanged.connect(self.update_lab1)
        self.slider_n0.valueChanged.connect(self.update_lab1)
        self.slider_rays.valueChanged.connect(self.update_lab1)
        self.update_lab1()

    def update_lab1(self):
        profile = 'lineal' if self.radio_linear.isChecked() else 'cuadratico'
        alpha = self.slider_alpha.value() / 1000.0
        n0 = self.slider_n0.value() / 100.0
        num_rays = self.slider_rays.value()
        self.lbl_alpha.setText(f"{alpha:+.3f}"); self.lbl_n0.setText(f"{n0:.2f}"); self.lbl_rays.setText(f"{num_rays}")
        self.canvas1.plot_sim(profile, alpha, n0, num_rays)

    def setup_tab2(self):
        layout = QHBoxLayout(self.tab2)
        control_panel = QWidget(); control_panel.setMaximumWidth(320)
        control_layout = QVBoxLayout(control_panel)
        
        grp_lab2 = QGroupBox("Parámetros del Paquete")
        layout_grid = QGridLayout(grp_lab2)
        
        layout_grid.addWidget(QLabel("Número de Ondas N:"), 0, 0, 1, 2)
        self.lbl_N = QLabel("31"); self.lbl_N.setStyleSheet("color: #fde725; font-weight: bold;")
        layout_grid.addWidget(self.lbl_N, 0, 2)
        self.slider_N = QSlider(Qt.Horizontal); self.slider_N.setRange(1, 100); self.slider_N.setValue(31)
        layout_grid.addWidget(self.slider_N, 1, 0, 1, 3)
        
        layout_grid.addWidget(QLabel("Momento Central k₀:"), 2, 0, 1, 2)
        self.lbl_k0 = QLabel("3.0"); self.lbl_k0.setStyleSheet("color: #fde725; font-weight: bold;")
        layout_grid.addWidget(self.lbl_k0, 2, 2)
        self.slider_k0 = QSlider(Qt.Horizontal); self.slider_k0.setRange(5, 60); self.slider_k0.setValue(30)
        layout_grid.addWidget(self.slider_k0, 3, 0, 1, 3)
        
        layout_grid.addWidget(QLabel("Ancho Espectral σₖ:"), 4, 0, 1, 2)
        self.lbl_sigmak = QLabel("0.80"); self.lbl_sigmak.setStyleSheet("color: #fde725; font-weight: bold;")
        layout_grid.addWidget(self.lbl_sigmak, 4, 2)
        self.slider_sigmak = QSlider(Qt.Horizontal); self.slider_sigmak.setRange(5, 200); self.slider_sigmak.setValue(80)
        layout_grid.addWidget(self.slider_sigmak, 5, 0, 1, 3)
        
        layout_grid.addWidget(QLabel("Tiempo t:"), 6, 0, 1, 2)
        self.lbl_t2 = QLabel("0.00"); self.lbl_t2.setStyleSheet("color: #fde725; font-weight: bold;")
        layout_grid.addWidget(self.lbl_t2, 6, 2)
        self.slider_t2 = QSlider(Qt.Horizontal); self.slider_t2.setRange(0, 1000); self.slider_t2.setValue(0)
        layout_grid.addWidget(self.slider_t2, 7, 0, 1, 3)
        control_layout.addWidget(grp_lab2)
        
        self.btn_play2 = QPushButton("▶ Reproducción (Play)"); self.btn_play2.clicked.connect(self.toggle_play2)
        control_layout.addWidget(self.btn_play2)
        btn_reset_t2 = QPushButton("↺ Reiniciar Tiempo (t = 0)"); btn_reset_t2.clicked.connect(lambda: self.slider_t2.setValue(0))
        control_layout.addWidget(btn_reset_t2)
        
        control_layout.addStretch()
        layout.addWidget(control_panel)
        
        self.canvas2 = Lab2Canvas(self.tab2)
        layout.addWidget(self.canvas2)
        
        self.slider_N.valueChanged.connect(self.update_lab2)
        self.slider_k0.valueChanged.connect(self.update_lab2)
        self.slider_sigmak.valueChanged.connect(self.update_lab2)
        self.slider_t2.valueChanged.connect(self.update_lab2)
        self.update_lab2()

    def update_lab2(self):
        N = self.slider_N.value(); k0 = self.slider_k0.value() / 10.0; sigma_k = self.slider_sigmak.value() / 100.0; t = self.slider_t2.value() / 100.0
        self.lbl_N.setText(f"{N}"); self.lbl_k0.setText(f"{k0:.1f}"); self.lbl_sigmak.setText(f"{sigma_k:.2f}"); self.lbl_t2.setText(f"{t:.2f}")
        self.canvas2.plot_wavepacket(N, k0, sigma_k, t)

    def toggle_play2(self):
        if self.is_playing2:
            self.anim_timer2.stop(); self.btn_play2.setText("▶ Reproducción (Play)"); self.is_playing2 = False
        else:
            self.anim_timer2.start(); self.btn_play2.setText("⏸ Pausar"); self.is_playing2 = True

    def advance_time2(self):
        self.slider_t2.setValue((self.slider_t2.value() + 5) % 1001)

    def setup_tab3(self):
        layout = QHBoxLayout(self.tab3)
        control_panel = QWidget(); control_panel.setMaximumWidth(320)
        control_layout = QVBoxLayout(control_panel)
        
        grp_lab3 = QGroupBox("Parámetros de Superposición")
        layout_grid = QGridLayout(grp_lab3)
        
        layout_grid.addWidget(QLabel("Mezcla θ:"), 0, 0, 1, 2)
        self.lbl_theta = QLabel("0.79 rad"); self.lbl_theta.setStyleSheet("color: #fde725; font-weight: bold;")
        layout_grid.addWidget(self.lbl_theta, 0, 2)
        self.slider_theta = QSlider(Qt.Horizontal); self.slider_theta.setRange(0, 157); self.slider_theta.setValue(79)
        layout_grid.addWidget(self.slider_theta, 1, 0, 1, 3)
        
        layout_grid.addWidget(QLabel("Fase Relativa δ:"), 2, 0, 1, 2)
        self.lbl_delta = QLabel("0.00 rad"); self.lbl_delta.setStyleSheet("color: #fde725; font-weight: bold;")
        layout_grid.addWidget(self.lbl_delta, 2, 2)
        self.slider_delta = QSlider(Qt.Horizontal); self.slider_delta.setRange(0, 628); self.slider_delta.setValue(0)
        layout_grid.addWidget(self.slider_delta, 3, 0, 1, 3)
        
        layout_grid.addWidget(QLabel("Fase Global χ:"), 4, 0, 1, 2)
        self.lbl_chi = QLabel("0.00 rad"); self.lbl_chi.setStyleSheet("color: #fde725; font-weight: bold;")
        layout_grid.addWidget(self.lbl_chi, 4, 2)
        self.slider_chi = QSlider(Qt.Horizontal); self.slider_chi.setRange(0, 628); self.slider_chi.setValue(0)
        layout_grid.addWidget(self.slider_chi, 5, 0, 1, 3)
        
        layout_grid.addWidget(QLabel("Posición Detector xD:"), 6, 0, 1, 2)
        self.lbl_xD = QLabel("0.33 L"); self.lbl_xD.setStyleSheet("color: #fde725; font-weight: bold;")
        layout_grid.addWidget(self.lbl_xD, 6, 2)
        self.slider_xD = QSlider(Qt.Horizontal); self.slider_xD.setRange(5, 95); self.slider_xD.setValue(33)
        layout_grid.addWidget(self.slider_xD, 7, 0, 1, 3)
        
        layout_grid.addWidget(QLabel("Tiempo t:"), 8, 0, 1, 2)
        self.lbl_t3 = QLabel("0.00 s"); self.lbl_t3.setStyleSheet("color: #fde725; font-weight: bold;")
        layout_grid.addWidget(self.lbl_t3, 8, 2)
        self.slider_t3 = QSlider(Qt.Horizontal); self.slider_t3.setRange(0, 500); self.slider_t3.setValue(0)
        layout_grid.addWidget(self.slider_t3, 9, 0, 1, 3)
        control_layout.addWidget(grp_lab3)
        
        self.btn_play3 = QPushButton("▶ Reproducción (Play)"); self.btn_play3.clicked.connect(self.toggle_play3)
        control_layout.addWidget(self.btn_play3)
        btn_center = QPushButton("Detector en Nodo (xD = L/2)"); btn_center.clicked.connect(lambda: self.slider_xD.setValue(50))
        control_layout.addWidget(btn_center)
        
        control_layout.addStretch()
        layout.addWidget(control_panel)
        
        self.canvas3 = Lab3Canvas(self.tab3)
        layout.addWidget(self.canvas3)
        
        self.slider_theta.valueChanged.connect(self.update_lab3)
        self.slider_delta.valueChanged.connect(self.update_lab3)
        self.slider_chi.valueChanged.connect(self.update_lab3)
        self.slider_xD.valueChanged.connect(self.update_lab3)
        self.slider_t3.valueChanged.connect(self.update_lab3)
        self.update_lab3()

    def update_lab3(self):
        theta = self.slider_theta.value() / 100.0; delta = self.slider_delta.value() / 100.0; chi = self.slider_chi.value() / 100.0
        xD = self.slider_xD.value() / 100.0; t_curr = self.slider_t3.value() / 100.0
        self.lbl_theta.setText(f"{theta:.2f} rad"); self.lbl_delta.setText(f"{delta:.2f} rad"); self.lbl_chi.setText(f"{chi:.2f} rad")
        self.lbl_xD.setText(f"{xD:.2f} L"); self.lbl_t3.setText(f"{t_curr:.2f} s")
        self.canvas3.plot_lab3(theta, delta, chi, xD, t_curr)

    def toggle_play3(self):
        if self.is_playing3:
            self.anim_timer3.stop(); self.btn_play3.setText("▶ Reproducción (Play)"); self.is_playing3 = False
        else:
            self.anim_timer3.start(); self.btn_play3.setText("⏸ Pausar"); self.is_playing3 = True

    def advance_time3(self):
        self.slider_t3.setValue((self.slider_t3.value() + 2) % 501)

    def setup_tab4(self):
        layout = QHBoxLayout(self.tab4)
        control_panel = QWidget(); control_panel.setMaximumWidth(320)
        control_layout = QVBoxLayout(control_panel)
        
        grp_lab4 = QGroupBox("Parámetros Doble Rendija")
        layout_grid = QGridLayout(grp_lab4)
        
        layout_grid.addWidget(QLabel("Separación d (mm):"), 0, 0, 1, 2)
        self.lbl_d4 = QLabel("1.80 mm"); self.lbl_d4.setStyleSheet("color: #fde725; font-weight: bold;")
        layout_grid.addWidget(self.lbl_d4, 0, 2)
        self.slider_d4 = QSlider(Qt.Horizontal); self.slider_d4.setRange(50, 400); self.slider_d4.setValue(180)
        layout_grid.addWidget(self.slider_d4, 1, 0, 1, 3)
        
        layout_grid.addWidget(QLabel("Ancho a (mm):"), 2, 0, 1, 2)
        self.lbl_a4 = QLabel("0.35 mm"); self.lbl_a4.setStyleSheet("color: #fde725; font-weight: bold;")
        layout_grid.addWidget(self.lbl_a4, 2, 2)
        self.slider_a4 = QSlider(Qt.Horizontal); self.slider_a4.setRange(10, 100); self.slider_a4.setValue(35)
        layout_grid.addWidget(self.slider_a4, 3, 0, 1, 3)
        
        layout_grid.addWidget(QLabel("Longitud λ (µm):"), 4, 0, 1, 2)
        self.lbl_w4 = QLabel("0.50 µm"); self.lbl_w4.setStyleSheet("color: #fde725; font-weight: bold;")
        layout_grid.addWidget(self.lbl_w4, 4, 2)
        self.slider_w4 = QSlider(Qt.Horizontal); self.slider_w4.setRange(20, 120); self.slider_w4.setValue(50)
        layout_grid.addWidget(self.slider_w4, 5, 0, 1, 3)
        control_layout.addWidget(grp_lab4)
        
        # Botones de disparo
        btn_grid = QGridLayout()
        btn_10 = QPushButton("+10"); btn_10.clicked.connect(lambda: self.add_events4(10))
        btn_100 = QPushButton("+100"); btn_100.clicked.connect(lambda: self.add_events4(100))
        btn_1000 = QPushButton("+1000"); btn_1000.clicked.connect(lambda: self.add_events4(1000))
        btn_grid.addWidget(btn_10, 0, 0); btn_grid.addWidget(btn_100, 0, 1); btn_grid.addWidget(btn_1000, 0, 2)
        control_layout.addLayout(btn_grid)
        
        self.btn_stream4 = QPushButton("Disparo Continuo"); self.btn_stream4.clicked.connect(self.toggle_stream4)
        control_layout.addWidget(self.btn_stream4)
        
        self.btn_mode4 = QPushButton("Modo: Coherente"); self.btn_mode4.clicked.connect(self.toggle_which_way4)
        control_layout.addWidget(self.btn_mode4)
        
        btn_clear4 = QPushButton("Limpiar Pantalla"); btn_clear4.clicked.connect(self.clear_events4)
        control_layout.addWidget(btn_clear4)
        
        control_layout.addStretch()
        layout.addWidget(control_panel)
        
        self.canvas4 = Lab4Canvas(self.tab4)
        layout.addWidget(self.canvas4)
        
        self.slider_d4.valueChanged.connect(self.update_lab4)
        self.slider_a4.valueChanged.connect(self.update_lab4)
        self.slider_w4.valueChanged.connect(self.update_lab4)
        self.update_lab4()

    def update_lab4(self):
        d = self.slider_d4.value() / 100.0
        a = self.slider_a4.value() / 100.0
        w = self.slider_w4.value() / 100.0
        self.lbl_d4.setText(f"{d:.2f} mm"); self.lbl_a4.setText(f"{a:.2f} mm"); self.lbl_w4.setText(f"{w:.2f} µm")
        self.canvas4.plot_lab4(d, a, w, self.which_way4, self.lab4_events)

    def add_events4(self, count):
        d = self.slider_d4.value() / 100.0
        a = self.slider_a4.value() / 100.0
        w = self.slider_w4.value() / 100.0
        p_density = self.canvas4.compute_probability(self.canvas4.y_grid, d, a, w, self.which_way4)
        p_norm = p_density / np.sum(p_density)
        sampled = np.random.choice(self.canvas4.y_grid, size=count, p=p_norm).tolist()
        self.lab4_events.extend(sampled)
        self.update_lab4()

    def toggle_stream4(self):
        if self.is_running4:
            self.anim_timer4.stop(); self.btn_stream4.setText("Disparo Continuo"); self.is_running4 = False
        else:
            self.anim_timer4.start(); self.btn_stream4.setText("Pausar Disparo"); self.is_running4 = True

    def advance_fire4(self):
        self.add_events4(80)

    def toggle_which_way4(self):
        self.which_way4 = not self.which_way4
        self.lab4_events = []
        if self.which_way4:
            self.btn_mode4.setText("Modo: Which-Way (Incoherente)")
        else:
            self.btn_mode4.setText("Modo: Coherente")
        self.update_lab4()

    def clear_events4(self):
        self.is_running4 = False
        self.anim_timer4.stop()
        self.btn_stream4.setText("Disparo Continuo")
        self.lab4_events = []
        self.update_lab4()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(VIRIDIS_QSS)
    window = QuantumLabMainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
