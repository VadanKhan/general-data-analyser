import numpy as np
import matplotlib.pyplot as plt


def plot_differential_cross_section(delta0_deg=40.0, delta1_deg=-10.0, E_cm_MeV=25.0):
    """
    Plots the S-wave and P-wave nucleon-nucleon differential cross section.
    Allows for variable phase shifts and center-of-mass energies.
    """
    # --- Physical Constants ---
    hbar_c = 197.327     # MeV fm
    m_nucleon = 938.27   # Average nucleon mass in MeV/c^2

    # --- Kinematics & Scaling ---
    # Reduced mass for identical nucleons (m*m)/(m+m) = m/2
    mu = m_nucleon / 2.0

    # Calculate k^2 in fm^-2
    k_sq = (2 * mu * E_cm_MeV) / (hbar_c**2)

    # Prefactor 1/k^2 gives area in fm^2. We multiply by 10 to convert to millibarns (mb).
    prefactor_mb = (1.0 / k_sq) * 10.0

    # --- Phase Shifts ---
    # Convert degrees to radians for numpy trig functions
    d0 = np.radians(delta0_deg)
    d1 = np.radians(delta1_deg)

    # --- Angular Arrays ---
    theta_deg = np.linspace(0, 180, 500)
    theta_rad = np.radians(theta_deg)
    cos_theta = np.cos(theta_rad)

    # --- Cross Section Calculation ---
    # Breaking down the equation into pure S, interference, and pure P terms
    term_S_pure = np.sin(d0)**2
    term_interference = 6 * np.sin(d0) * np.sin(d1) * np.cos(d0 - d1) * cos_theta
    term_P_pure = 9 * (np.sin(d1)**2) * (cos_theta**2)

    # Total differential cross section (mb/sr)
    dsig_dOmega = prefactor_mb * (term_S_pure + term_interference + term_P_pure)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: dSigma/dOmega vs Theta
    ax1.plot(theta_deg, dsig_dOmega, 'b-', linewidth=2)
    ax1.set_xlabel(r'Center-of-Mass Angle $\theta$ (degrees)', fontsize=12)
    ax1.set_ylabel(r'$\frac{d\sigma}{d\Omega}$ (mb / sr)', fontsize=14)
    ax1.set_title(
        f'Differential Cross Section vs. $\\theta$\n($\\delta_0={delta0_deg}^\\circ$, $\\delta_1={delta1_deg}^\\circ$, $E_{{cm}}={E_cm_MeV}$ MeV)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlim(0, 180)
    ax1.set_xticks(np.arange(0, 181, 30))
    ax1.set_ylim(0, max(dsig_dOmega) * 1.1)  # Scale y-axis 10% above max value

    # Plot 2: dSigma/dOmega vs Cos(Theta)
    ax2.plot(cos_theta, dsig_dOmega, 'r-', linewidth=2)
    ax2.set_xlabel(r'$\cos(\theta)$', fontsize=12)
    ax2.set_ylabel(r'$\frac{d\sigma}{d\Omega}$ (mb / sr)', fontsize=14)
    ax2.set_title('Differential Cross Section vs. $\\cos(\\theta)$')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(0, max(dsig_dOmega) * 1.1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # You can easily change the variables here to see how the shape adapts!
    plot_differential_cross_section(delta0_deg=40, delta1_deg=-10, E_cm_MeV=25)
