import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

# -----------------------------------------------------------------------------

def modelo(t,y,p):

    V   = y[0]
    Ap  = y[1]
    ApM = y[2]
    I   = y[3]
    ThN = y[4]
    ThE = y[5]
    TkN = y[6]
    TkE = y[7]
    B   = y[8]
    Ps  = y[9]
    Pl  = y[10]
    Bm  = y[11]
    IgM = y[12]
    IgG = y[13]
    C   = y[14]

    Ap0  = p['Ap0']
    ThN0 = p['ThN0']
    TkN0 = p['TkN0']
    B0   = p['B0']

    dVdt   = p['pi_v']*V - p['kv1']*V*IgG - p['kv1']*V*IgM - p['kv2']*V*TkE - p['kv3']*V*ApM
    dApdt  = p['alpha_ap']*(C + 1.0)*(Ap0 - Ap) - p['beta_ap']*Ap*((p['cap1']*V)/(p['cap2'] + V))
    dApMdt = p['beta_ap']*Ap*((p['cap1']*V)/(p['cap2'] + V)) - p['beta_apm']*ApM*V - p['gama_apm']*ApM 
    dIdt   = p['beta_apm']*ApM*V + p['beta_tke']*TkE*V - p['gama_apm']*I 
    dThNdt = p['alpha_th']*(ThN0 - ThN) - p['beta_th']*ApM*ThN
    dThEdt = p['beta_th']*ApM*ThN + p['pi_th']*ApM*ThE - p['delta_th']*ThE 
    dTkNdt = p['alpha_tk']*(C + 1)*(TkN0 - TkN) - p['beta_tk']*(C + 1)*ApM*TkN
    dTkEdt = p['beta_tk']*(C + 1)*ApM*TkN + p['pi_tk']*ApM*TkE - p['beta_tke']*TkE*V - p['delta_tk']*TkE
    dBdt   = p['alpha_b']*(B0 - B) + p['pi_b1']*V*B + p['pi_b2']*ThE*B - p['beta_ps']*ApM*B - p['beta_pl']*ThE*B - p['beta_bm']*ThE*B
    dPsdt  = p['beta_ps']*ApM*B - p['delta_ps']*Ps
    dPldt  = p['beta_pl']*ThE*B - p['delta_pl']*Pl + p['delta_bm']*Bm
    dBmdt  = p['beta_bm']*ThE*B + p['pi_bm1']*Bm*(1.0 - (Bm/p['pi_bm2'])) - p['delta_bm']*Bm
    dIgMdt = p['pi_ps']*Ps - p['delta_am']*IgM
    dIgGdt = p['pi_pl']*Pl - p['delta_ag']*IgG
    dCdt   = p['pi_capm']*ApM + p['pi_ci']*I + p['pi_ctke']*TkE - p['gama_c']*C

    dydt = [dVdt, dApdt, dApMdt, dIdt, dThNdt, 
            dThEdt, dTkNdt, dTkEdt, dBdt, dPsdt,
            dPldt, dBmdt,  dIgMdt,  dIgGdt, dCdt]

    return dydt

# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # parametros
    pars = {}
    pars['pi_v']     = 1.47
    pars['kv1']      = 9.82e-3
    pars['kv2']      = 6.10e-5
    pars['kv3']      = 6.45e-2
    pars['alpha_ap'] = 1.0
    pars['beta_ap']  = 1.79e-1
    pars['cap1']     = 8.0
    pars['cap2']     = 8.08e6
    pars['gama_apm'] = 4.0e-2
    pars['beta_apm'] = 1.33e-2
    pars['beta_tke'] = 3.5e-6
    pars['alpha_th'] = 2.17e-4
    pars['beta_th']  = 1.8e-5
    pars['pi_th']    = 1.0e-8
    pars['delta_th']  = 3.0e-1
    pars['alpha_tk'] = 1.0
    pars['beta_tk']  = 1.43e-5
    pars['pi_tk']    = 1.0e-8
    #pars['delta_tk']  = 3.0e-1
    pars['delta_tk']  = 3.0e-2
    #pars['alpha_b']  = 3.58e2
    pars['alpha_b']  = 3.578236584
    pars['pi_b1']    = 8.98e-5
    pars['pi_b2']    = 1.27e-8
    pars['beta_ps']  = 6.0e-6
    pars['beta_pl']  = 5.0e-6
    pars['beta_bm']  = 1.0e-6
    pars['delta_ps']  = 2.5
    pars['delta_pl']  = 3.5e-1
    pars['delta_bm']  = 9.75e-4
    pars['pi_bm1']   = 1.0e-5
    pars['pi_bm2']   = 2.5e3
    pars['pi_ps']    = 8.7e-2
    pars['pi_pl']    = 1.0e-3
    pars['delta_am']  = 7.0e-2
    pars['delta_ag']  = 7.0e-2
    pars['pi_capm']  = 3.28e2
    pars['pi_ci']    = 6.44e-3
    pars['pi_ctke']  = 1.78e-2
    pars['gama_c']   = 7.04e2

    # condicoes Iniciais 
    V0   = 61.0  # copies/mL
    Ap0  = 1.0e6 # cells/mL
    ApM0 = 0.0   # cells/mL
    I0   = 0.0   # cells/mL
    ThN0 = 1.0e6 # cells/mL
    ThE0 = 0.0   # cells/mL
    TkN0 = 5.0e5 # cells/mL
    TkE0 = 0.0   # cells/mL
    B0   = 2.5e5 # cells/mL
    Ps0  = 0.0   # cells/mL
    Pl0  = 0.0   # cells/mL
    Bm0  = 0.0   # cells/mL
    IgM0 = 0.0   # S/CO
    IgG0 = 0.0   # S/CO
    C0   = 0.0   # pg/mL

    # adiciona algumas condicoes iniciais nos parametros
    pars['Ap0'] = Ap0
    pars['ThN0'] = ThN0
    pars['TkN0'] = TkN0
    pars['B0'] = B0

    y0 = [V0, Ap0, ApM0, I0, ThN0, 
        ThE0, TkN0, TkE0, B0, Ps0, 
        Pl0, Bm0, IgM0, IgG0, C0]

    # tempo de simulacao
    tf = 35.0 #dias
    dt = 0.01
    N = int(tf/dt)
    t = np.linspace(0,tf,N)

    #sol = odeint(modelo, y0, t, args=(pars,))
    sol = solve_ivp(modelo, [0, tf], y0, args=(pars,), method='Radau')

    t = sol.t

    V   = sol.y[0,:]
    Ap  = sol.y[1,:]
    ApM = sol.y[2,:]
    I   = sol.y[3,:]
    ThN = sol.y[4,:]
    ThE = sol.y[5,:]
    TkN = sol.y[6,:]
    TkE = sol.y[7,:]
    B   = sol.y[8,:]
    Ps  = sol.y[9,:]
    Pl  = sol.y[10,:]
    Bm  = sol.y[11,:]
    IgM = sol.y[12,:]
    IgG = sol.y[13,:]
    C   = sol.y[14,:]

    plt.subplot(2,2,1)
    plt.plot(t,V, color='red', label='V')
    plt.xlabel('Time (days)')
    plt.ylabel('Viremia(copies/mL)')
    plt.legend()
    plt.ylim(1,1e8)
    plt.yscale('log')
    plt.grid()

    plt.subplot(2,2,2)
    plt.plot(t,C, color='blue', label='C')
    plt.xlabel('Time (days)')
    plt.ylabel('Cytokines(pg/mL)')
    plt.legend()
    plt.ylim(0,35)
    plt.grid()

    plt.subplot(2,2,3)
    plt.plot(t,IgG, color='orange', label='IgG')
    plt.xlabel('Time (days)')
    plt.ylabel('IgG (S/CO)')
    plt.legend()
    plt.yscale('log', base=2)
    plt.ylim(2e-1,2e6)
    plt.grid()

    plt.subplot(2,2,4)
    plt.plot(t,IgM, color='black', label='IgM')
    plt.xlabel('Time (days)')
    plt.ylabel('IgM (S/CO)')
    plt.legend()
    plt.yscale('log', base=2)
    plt.ylim(2e-1,2e6)
    plt.grid()

    plt.tight_layout()
    plt.show()
