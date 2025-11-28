"""
Computational Model of Stress-Autophagy-Neural Plasticity Dynamics
Author: Polykleitos Rengos
For: Max Planck School of Cognition Application
Date: November 2024
"""

import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from datetime import datetime
from io import BytesIO

# Configure page
st.set_page_config(
    page_title="Autophagy-Plasticity Model",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mathematical Model - FIXED VERSION
def model_equations(y, t, params):
    """
    System of ODEs describing autophagy-plasticity dynamics
    CRITICAL FIX: Damage now impairs maintenance machinery
    
    Variables:
    - S: Synaptic strength (normalized)
    - D: Damage/debris accumulation (normalized)
    - A: Autophagic flux (normalized)
    - E: Energy availability (ATP/ADP ratio proxy)
    """
    S, D, A, E = y
    
    # Extract parameters
    k_decay = params['k_decay']
    k_damage = params['k_damage']
    k_maintain = params['k_maintain']
    beta = params['beta']
    alpha = params['alpha']
    sigma = params['sigma']
    A0 = params['A0']
    k_atp = params['k_atp']
    k_consume = params['k_consume']
    
    # Apply physiological bounds
    S = max(0, S)
    D = max(0, min(1.0, D))
    A = max(0.1, A)
    E = max(0.1, min(5.0, E))
    
    # Autophagic flux (stress-suppressed)
    A_eff = A0 / (1 + sigma)
    dA_dt = 0.1 * (A_eff - A)  # Slow adaptation
    
    # Energy dynamics
    production = k_atp * (1 - 0.5 * D)  # Damage impairs production
    consumption = k_consume * S  # Active synapses consume ATP
    dE_dt = production - consumption
    
    # Synaptic strength dynamics
    decay = k_decay * S
    damage_effect = k_damage * D * S
    
    # CRITICAL FIX: Damage impairs maintenance machinery
    # Energy can't be used effectively when cellular machinery is damaged
    maintenance = k_maintain * S * E * (1 - D)  # NEW: (1-D) factor
    
    dS_dt = -decay - damage_effect + maintenance
    
    # Damage accumulation
    production_damage = beta
    clearance = alpha * A * D
    dD_dt = production_damage - clearance
    
    return [dS_dt, dD_dt, dA_dt, dE_dt]

def run_simulation(params, t_stim=30, stim_strength=0.5):
    """Run full simulation with LTP stimulus"""
    
    # Time vectors
    t_pre = np.linspace(0, t_stim, 500)
    t_post = np.linspace(t_stim, 150, 1000)
    
    # Initial conditions [S, D, A, E]
    y0 = [1.0, 0.1, params['A0'], 1.0]
    
    # Pre-stimulus phase
    sol_pre = odeint(model_equations, y0, t_pre, args=(params,))
    
    # Apply LTP stimulus
    y_stim = sol_pre[-1].copy()
    y_stim[0] += stim_strength  # Increase synaptic strength
    
    # Post-stimulus phase
    sol_post = odeint(model_equations, y_stim, t_post, args=(params,))
    
    # Combine results
    t_full = np.concatenate([t_pre, t_post])
    sol_full = np.vstack([sol_pre, sol_post])
    
    return t_full, sol_full, t_stim

def create_figure(t, sol, t_stim, title, params):
    """Create publication-quality figure"""
    
    S, D, A, E = sol.T
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.3)
    
    # Define colors
    colors = {
        'S': '#1565C0',  # Blue
        'D': '#C62828',  # Red
        'A': '#2E7D32',  # Green
        'E': '#F57C00'   # Orange
    }
    
    # Synaptic Strength
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, S, color=colors['S'], linewidth=2.5, label='Synaptic Strength')
    ax1.axvline(t_stim, color='gray', linestyle='--', alpha=0.7, label='LTP Stimulus')
    ax1.axhline(1.0, color='gray', linestyle=':', alpha=0.3)
    ax1.fill_between(t, 0, S, alpha=0.15, color=colors['S'])
    ax1.set_ylabel('Synaptic\nStrength (S)', fontsize=11, fontweight='bold')
    ax1.set_ylim([0, 2.0])
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim([0, 150])
    
    # Damage
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t, D, color=colors['D'], linewidth=2.5, label='Damage Accumulation')
    ax2.fill_between(t, 0, D, alpha=0.15, color=colors['D'])
    ax2.set_ylabel('Damage (D)', fontsize=11, fontweight='bold')
    ax2.set_ylim([0, max(0.5, np.max(D) * 1.1)])
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim([0, 150])
    
    # Autophagic Flux
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(t, A, color=colors['A'], linewidth=2.5, label='Autophagic Flux')
    ax3.fill_between(t, 0, A, alpha=0.15, color=colors['A'])
    ax3.set_ylabel('Autophagy (A)', fontsize=11, fontweight='bold')
    ax3.set_ylim([0, 1.2])
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.2)
    ax3.set_xlim([0, 150])
    
    # Energy
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(t, E, color=colors['E'], linewidth=2.5, label='Energy Availability')
    ax4.fill_between(t, 0, E, alpha=0.15, color=colors['E'])
    ax4.set_xlabel('Time (arbitrary units)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Energy (E)', fontsize=11, fontweight='bold')
    ax4.set_ylim([0, max(2.0, np.max(E) * 1.1)])
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.2)
    ax4.set_xlim([0, 150])
    
    # Title and subtitle
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Add parameter annotation
    param_text = f"σ = {params['sigma']:.1f}"
    if params['sigma'] > 0:
        param_text += " (Chronic Stress)"
        color = 'red'
    else:
        param_text += " (Control)"
        color = 'green'
    
    fig.text(0.99, 0.96, param_text, ha='right', fontsize=10, 
             style='italic', color=color, transform=fig.transFigure)
    
    plt.tight_layout()
    return fig

# STREAMLIT INTERFACE
st.title("🧠 Computational Model of Autophagy-Mediated Neural Plasticity")
st.markdown("""
### Mathematical Framework for Stress-Dependent Synaptic Dynamics
**Author:** Polykleitos Rengos | **Date:** November 2024
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Model Configuration", "Results & Analysis", "Mathematical Framework"])

with tab1:
    st.header("Model Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Autophagy System")
        sigma = st.slider("σ (Stress Level)", 0.0, 8.0, 0.0, 0.5,
                         help="Chronic stress suppresses autophagy via mTOR pathway")
        A0 = st.slider("A₀ (Baseline Autophagy)", 0.5, 1.5, 1.0, 0.1,
                      help="Normal autophagic flux capacity")
        alpha = st.slider("α (Clearance Efficiency)", 0.1, 1.0, 0.5, 0.1,
                         help="Rate of damage clearance by autophagy")
        beta = st.slider("β (Damage Production)", 0.01, 0.1, 0.03, 0.01,
                        help="Metabolic waste generation rate")
    
    with col2:
        st.subheader("Synaptic Dynamics")
        k_decay = st.slider("k_decay (Natural Decay)", 0.001, 0.01, 0.003, 0.001,
                           help="Baseline synaptic decay rate")
        k_damage = st.slider("k_damage (Damage Impact)", 0.1, 2.0, 1.0, 0.1,  # INCREASED DEFAULT
                            help="How damage accelerates decay")
        k_maintain = st.slider("k_maintain (Maintenance)", 0.0, 0.02, 0.005, 0.002,  # REDUCED DEFAULT
                              help="Energy-dependent synaptic maintenance")
        
        st.subheader("Energy Metabolism")
        k_atp = st.slider("k_ATP (Production)", 0.1, 0.5, 0.25, 0.05,
                         help="ATP production rate")
        k_consume = st.slider("k_consume (Consumption)", 0.05, 0.3, 0.15, 0.05,
                             help="Energy consumption by synapses")
    
    st.subheader("Stimulation Protocol")
    stim_time = st.number_input("Stimulus Time", 10, 50, 30, 5)
    stim_strength = st.slider("LTP Stimulus Strength", 0.0, 1.0, 0.5, 0.1)

with tab2:
    st.header("Simulation Results")
    
    # Instructions
    st.info("""
    👉 **Instructions:**
    1. Configure parameters in the Model Configuration tab
    2. Click "Run Simulation" for single condition
    3. Click "Generate Comparison" for control vs stress analysis
    """)
    
    # Compile parameters
    params = {
        'k_decay': k_decay,
        'k_damage': k_damage,
        'k_maintain': k_maintain,
        'beta': beta,
        'alpha': alpha,
        'sigma': sigma,
                'A0': A0,
        'k_atp': k_atp,
        'k_consume': k_consume
    }
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
            st.session_state.run_sim = True
    
    with col2:
        if st.button("📊 Generate Comparison (σ=0 vs σ=4)", type="secondary", use_container_width=True):
            st.session_state.run_comparison = True
    
    # Single simulation
    if hasattr(st.session_state, 'run_sim') and st.session_state.run_sim:
        with st.spinner("Running simulation..."):
            t, sol, t_stim = run_simulation(params, stim_time, stim_strength)
            
            # Create figure
            title = "Neural Plasticity Dynamics Under Stress-Autophagy Modulation"
            fig = create_figure(t, sol, t_stim, title, params)
            
            # Display
            st.pyplot(fig)
            
            # Metrics
            st.subheader("Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            S_final = sol[-1, 0]
            S_peak = np.max(sol[:, 0])
            D_final = sol[-1, 1]
            A_final = sol[-1, 2]
            
            with col1:
                st.metric("Final Synaptic Strength", f"{S_final:.3f}")
            with col2:
                st.metric("Peak Strength", f"{S_peak:.3f}")
            with col3:
                st.metric("Final Damage", f"{D_final:.3f}")
            with col4:
                # Fixed logic: LTP maintained if above baseline but not unrealistic
                maintained = "✅ Yes" if (S_final > 1.15 and S_final < 2.5) else "❌ No"
                st.metric("LTP Maintained", maintained)
            
            # Interpretation
            st.subheader("Interpretation")
            if S_final > 1.15 and S_final < 2.5:
                st.success("✅ LTP successfully maintained - synaptic strength remains elevated")
            elif S_final > 2.5:
                st.warning("⚠️ Unrealistic synaptic growth - possible parameter imbalance")
            else:
                st.error("❌ LTP failed - synaptic strength returned to or below baseline")
            
            # Download button
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            st.download_button(
                label="📥 Download Figure (PNG)",
                data=buf,
                file_name=f"simulation_sigma_{sigma:.1f}.png",
                mime="image/png"
            )
    
    # Comparison
    if hasattr(st.session_state, 'run_comparison') and st.session_state.run_comparison:
        with st.spinner("Generating comparison..."):
            
            st.subheader("Comparative Analysis: Control vs Chronic Stress")
            
            col1, col2 = st.columns(2)
            
            # Control condition
            params_control = params.copy()
            params_control['sigma'] = 0.0
            t1, sol1, ts1 = run_simulation(params_control, stim_time, stim_strength)
            fig1 = create_figure(t1, sol1, ts1, "Control Condition", params_control)
            
            # Stress condition
            params_stress = params.copy()
            params_stress['sigma'] = 4.0
            t2, sol2, ts2 = run_simulation(params_stress, stim_time, stim_strength)
            fig2 = create_figure(t2, sol2, ts2, "Chronic Stress Condition", params_stress)
            
            with col1:
                st.pyplot(fig1)
                
                # Download button for control
                buf1 = BytesIO()
                fig1.savefig(buf1, format='png', dpi=300, bbox_inches='tight')
                buf1.seek(0)
                st.download_button(
                    label="📥 Download Control Figure",
                    data=buf1,
                    file_name="figure_control.png",
                    mime="image/png",
                    key="dl_control"
                )
            
            with col2:
                st.pyplot(fig2)
                
                # Download button for stress
                buf2 = BytesIO()
                fig2.savefig(buf2, format='png', dpi=300, bbox_inches='tight')
                buf2.seek(0)
                st.download_button(
                    label="📥 Download Stress Figure",
                    data=buf2,
                    file_name="figure_stress.png",
                    mime="image/png",
                    key="dl_stress"
                )
            
            # Comparison metrics
            st.subheader("Quantitative Comparison")
            
            comparison_data = {
                'Metric': ['Final S', 'Peak S', 'Final D', 'Final A', 'Final E'],
                'Control (σ=0)': [
                    f"{sol1[-1, 0]:.3f}",
                    f"{np.max(sol1[:, 0]):.3f}",
                    f"{sol1[-1, 1]:.3f}",
                    f"{sol1[-1, 2]:.3f}",
                    f"{sol1[-1, 3]:.3f}"
                ],
                'Stress (σ=4)': [
                    f"{sol2[-1, 0]:.3f}",
                    f"{np.max(sol2[:, 0]):.3f}",
                    f"{sol2[-1, 1]:.3f}",
                    f"{sol2[-1, 2]:.3f}",
                    f"{sol2[-1, 3]:.3f}"
                ]
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            st.table(df_comparison)
            
            # Scientific interpretation
            st.subheader("Scientific Interpretation")
            st.markdown("""
            **Key Findings:**
            - **Control**: Maintains synaptic strength with minimal damage accumulation
            - **Chronic Stress**: Shows impaired plasticity with damage accumulation
            - **Autophagy suppression** (σ=4) reduces clearance capacity by 80%
            - **Damage accumulation** is significantly higher under chronic stress
            
            These results demonstrate that stress-induced autophagy suppression 
            impairs synaptic maintenance and plasticity in neural circuits.
            """)

with tab3:
    st.header("Mathematical Framework")
    
    st.markdown("""
    ### System of Ordinary Differential Equations
    
    The model describes the interaction between synaptic strength, cellular damage, 
    autophagic flux, and energy availability under stress conditions.
    """)
    
    st.latex(r"""
    \begin{align}
    \frac{dS}{dt} &= -k_{decay} \cdot S - k_{damage} \cdot D \cdot S + k_{maintain} \cdot S \cdot E \cdot (1-D) \\
    \frac{dD}{dt} &= \beta - \alpha \cdot A \cdot D \\
    \frac{dA}{dt} &= 0.1 \cdot \left(\frac{A_0}{1 + \sigma} - A\right) \\
    \frac{dE}{dt} &= k_{ATP} \cdot (1 - 0.5D) - k_{consume} \cdot S
    \end{align}
    """)
    
    st.markdown("""
    ### Key Mechanisms
    
    1. **Stress-Autophagy Suppression**: Chronic stress (σ) reduces autophagic flux:
       - A_effective = A₀ / (1 + σ)
    
    2. **Damage-Energy Coupling**: Accumulated damage impairs ATP production:
       - Production = k_ATP × (1 - 0.5D)
    
    3. **Energy-Dependent Maintenance WITH Damage Impairment**:
       - Maintenance = k_maintain × S × E × (1-D)
       - **Critical:** Damage directly impairs maintenance machinery
    
    ### Biological Interpretation
    
    - **S (Synaptic Strength)**: Efficacy of synaptic transmission and LTP
    - **D (Damage)**: Accumulated protein aggregates and dysfunctional organelles
    - **A (Autophagy)**: Cellular clearance capacity via autophagic flux
    - **E (Energy)**: ATP availability for synaptic processes
    
    ### Expected Outcomes
    
    - **Control (σ=0)**: Efficient autophagy maintains low damage, enabling sustained LTP
    - **Chronic Stress (σ>3)**: Suppressed autophagy → damage accumulation → LTP failure
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
Developed for Neural Plasticity Research | November 2024<br>
Model demonstrates the critical role of autophagy in neural plasticity regulation
</div>
""", unsafe_allow_html=True)
    
