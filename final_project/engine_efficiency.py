# https://ntrs.nasa.gov/api/citations/19750022978/downloads/19750022978.pdf 
# https://leehamnews.com/2014/11/13/fundamentals-of-airliner-performance-part-2 
# https://www.boeing.com/content/dam/boeing/boeingdotcom/commercial/airports/acaps/737.pdf
# https://en.wikipedia.org/wiki/CFM_International_LEAP

import math
import csv
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "AirTable.csv")

def std_atm_SI(geometric_altitude_m):

    Rez = 6356766.0
    R = 287.0528
    gssl = 9.80665
    gamma = 1.4
    Ts = 273.15
    mu_s = 1.716e-5
    Ks = 110.4
    tol = 1e-12

    ti = [288.150,216.650,216.650,228.650,270.650,270.650,252.650,180.650]
    ti_prime = [-0.0065,0.0,0.001,0.0028,0.0,-0.0020,-0.004,0.0]
    p_i = [
        1.01325e5,2.263203182e4,5.474873528e3,8.680147691e2,
        1.109055889e2,5.900052428e1,1.8209924905e1,1.037700455
    ]
    zi = [0,11000,20000,32000,47000,52000,61000,79000,90000]

    geopotential_altitude_m = Rez * geometric_altitude_m / (Rez + geometric_altitude_m)

    i = 0
    while i+1 < len(zi) and zi[i+1] <= geopotential_altitude_m:
        i += 1

    temp_k = ti[i] + ti_prime[i] * (geopotential_altitude_m - zi[i])

    if abs(ti_prime[i]) <= tol:
        pressure = p_i[i] * math.exp(-(gssl*(geopotential_altitude_m-zi[i]))/(R*ti[i]))
    else:
        pressure = p_i[i] * ((temp_k)/ti[i]) ** (-gssl/(R*ti_prime[i]))

    density = pressure / (R * temp_k)
    dyn_visc = mu_s * ((Ts+Ks)/(temp_k+Ks)) * (temp_k/Ts)**1.5
    sos = math.sqrt(gamma * R * temp_k)

    return {
        "temp_k": temp_k,
        "pressure_N_per_m2": pressure,
        "density_kg_per_m3": density,
        "dyn_viscosity_pa_sec": dyn_visc,
        "sos_m_per_sec": sos,
    }

def load_airtable():
    table = {"T": [], "h": [], "PR": []}

    cleaned = []
    with open(CSV_PATH, "r") as f:
        for line in f:
            if line.strip(",\n ") != "": 
                cleaned.append(line)

    import csv
    reader = csv.DictReader(cleaned)

    for row in reader:
        if row["T"] == "":
            continue

        table["T"].append(float(row["T"]))
        table["h"].append(float(row["h"]))
        table["PR"].append(float(row["PR"]))

    return table

_AIRTABLE = load_airtable()

T  = _AIRTABLE["T"]
h  = _AIRTABLE["h"]
PR = _AIRTABLE["PR"]

def interp1(x_table, y_table, x_query):
    if x_query <= x_table[0]:
        return y_table[0]
    if x_query >= x_table[-1]:
        return y_table[-1]

    for i in range(len(x_table)-1):
        x0 = x_table[i]
        x1 = x_table[i+1]
        if x0 <= x_query <= x1:
            y0 = y_table[i]
            y1 = y_table[i+1]
            t = (x_query - x0) / (x1 - x0)
            return y0 + t*(y1 - y0)

    raise RuntimeError("Interpolation error")

class Turbofan:
    def __init__(self, altitude, mach_num, bypass_ratio, combustion_temp, combustion_pressure):
        self.mach_num = mach_num
        self.bypass_ratio = bypass_ratio
        self.gamma = 1.4

        atm = std_atm_SI(altitude)
        self.temp_atm = atm["temp_k"]
        self.p_atm = atm["pressure_N_per_m2"]
        self.density_atm = atm["density_kg_per_m3"]
        self.sos_atm = atm["sos_m_per_sec"]

    def thrust(self, CD, mach_num):
        self.V0 = mach_num * self.sos_atm
        thrust = 0.5*self.density_atm*self.V0**2*127.0*CD*0.5
        return thrust/1000  # kN

    def efficiency(self, P_electric):
        pc_ratio = 41
        BPR = 9

        pd_ratio = 1.5
        eta_comp = 0.92
        eta_fan  = 0.90
        eta_turb = 0.89

        FAR   = 0.027
        T03   = 1900
        gamma_air = 1.4

        # STAGE 0: Inlet
        V1 = 850*1000/3600
        T1 = 228.7
        P1 = 30.1 * 1000

        h1   = interp1(T, h, T1)
        h01  = h1 + 0.5*V1**2/1000
        T01  = interp1(h, T, h01)
        Pr01 = interp1(h, PR, h01)
        P01  = P1 * (T01/T1)**(gamma_air/(gamma_air-1))

        # COMPRESSOR
        Pr02i = pc_ratio * Pr01
        h02i  = interp1(PR, h, Pr02i)
        wc    = (h02i - h01)/eta_comp    # compressor work
        h02   = h01 + wc
        Pr02  = interp1(h, PR, h02)
        P02   = Pr02 * P01

        # FAN (original)
        Pr02di = pd_ratio * Pr01
        h02di  = interp1(PR, h, Pr02di)
        wf     = (h02di - h01)/eta_fan   # fan work (kJ/kg)
        h02d   = h01 + wf

        # ELECTRIC ASSIST ON FAN
        m_dot = 300.0   # initial guess for iteration

        for _ in range(5):
            bypass_frac = BPR / (1 + BPR)
            m_dot_bypass = m_dot * bypass_frac

            # electric power -> fan specific power reduction [kJ/kg]
            if m_dot_bypass > 0:
                delta_wf = P_electric / m_dot_bypass / 1000.0
            else:
                delta_wf = 0.0

            # new fan work
            wf_new = max(0.0, wf - delta_wf)
            h02d_new = h01 + wf_new

            # BYPASS NOZZLE
            Pr02d_new = interp1(h, PR, h02d_new)
            Pr5d      = Pr02d_new / pd_ratio
            h5d       = interp1(PR, h, Pr5d)
            V5d       = math.sqrt(max(0, (h02d_new - h5d)*2*1000))

            # COMBUSTOR
            h03  = interp1(T, h, T03)
            Pr03 = interp1(T, PR, T03)
            qc   = h03 - h02

            # turbine work requirement
            wt_new = wc + BPR*wf_new

            # TURBINE
            h04   = h03 - wt_new
            Pr04  = interp1(h, PR, h04)
            h04i  = h03 - wt_new/eta_turb
            Pr04i = interp1(h, PR, h04i)
            P04   = (Pr04i/Pr03)*P02

            # CORE NOZZLE
            P5  = P1
            Pr5 = P5 * Pr04 / P04
            h5  = interp1(PR, h, Pr5)
            V5  = math.sqrt(max(0, (h04 - h5)*2*1000))

            m_dot = thrust / max(1e-6, (V5 - self.V0))

        thermal_eff = (V5**2/2 + 2*V5d**2/2) / (qc*1000)
        eta_total   = thermal_eff * eta_comp * eta_fan * eta_turb

        return eta_total

# Run engine model
leap_1b = Turbofan(10668, 0.78, 9, 3000, 20)
thrust = leap_1b.thrust(0.0397, 0.8)
# power in watts
efficiency = leap_1b.efficiency(P_electric=1000.0)
print(f"Thrust required (kN): {thrust:.3f}")

print(f'efficiency (%): {efficiency * 100:.6}')