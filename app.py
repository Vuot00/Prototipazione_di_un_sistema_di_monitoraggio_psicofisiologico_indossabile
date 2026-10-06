import streamlit as st
import time
import csv
import os
import joblib
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch

from manager import get_manager
from workers import start_threads_if_needed

COM_IMU = "COM4"
COM_ECG = "COM5"
NOME_FILE_CSV = "dati_sessione.csv"
NOME_MODELLO = "modello_wesad.pkl"
FEATURE_COLS = ['BPM', 'RMSSD', 'SDNN', 'MeanRR', 'LF_HF', 'Activity_Mean', 'Activity_Std']


@st.cache_resource
def carica_modello_ia():
    try:
        return joblib.load(NOME_MODELLO)
    except Exception as e:
        st.error(f"Modello non trovato o errore di caricamento: {e}")
        return None


def calcola_feature_live(rr_list, imu_data, bpm, fs_rr=4.0):
    if len(rr_list) < 8:
        return None

    rr = np.array(rr_list)
    rr_ms = rr * 1000

    rmssd = np.sqrt(np.mean(np.diff(rr_ms) ** 2))
    sdnn = np.std(rr_ms)
    mean_rr = np.mean(rr_ms)

    try:
        t = np.cumsum(rr_ms) / 1000.0
        t -= t[0]
        f_interp = interp1d(t, rr_ms, kind='linear', fill_value='extrapolate')
        t_uniform = np.arange(0, t[-1], 1.0 / fs_rr)
        rr_uniform = f_interp(t_uniform)
        freqs, psd = welch(rr_uniform, fs=fs_rr, nperseg=min(len(rr_uniform), 256))
        lf = np.trapezoid(psd[(freqs >= 0.04) & (freqs < 0.15)],
                          freqs[(freqs >= 0.04) & (freqs < 0.15)])
        hf = np.trapezoid(psd[(freqs >= 0.15) & (freqs < 0.40)],
                          freqs[(freqs >= 0.15) & (freqs < 0.40)])
        lf_hf = lf / hf if hf > 0 else 0.0
    except Exception:
        lf_hf = 0.0

    imu_arr = np.array(imu_data) if len(imu_data) > 0 else np.array([0])
    activity_mean = np.mean(imu_arr)
    activity_std = np.std(imu_arr)

    return [bpm, rmssd, sdnn, mean_rr, lf_hf, activity_mean, activity_std]


def main():
    st.set_page_config(page_title="Dashboard Shimmer Live", layout="wide")
    st.title("🧠 Intelligenza Artificiale & Analisi dello Stress")

    manager = get_manager()
    modello_ia = carica_modello_ia()

    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        start_threads_if_needed(manager, COM_IMU, COM_ECG)

    # --- ZONA CONTROLLI ---
    c_header1, c_header2 = st.columns([3, 1])
    with c_header2:
        registra = st.toggle("🔴 Registra Dati CSV")

    # --- GESTIONE CSV ---
    if registra:
        manager.is_recording = True
        if not os.path.isfile(NOME_FILE_CSV):
            with open(NOME_FILE_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Sensore", "Valore_Primario", "Valore_Secondario", "Stato_IA"])
        with manager.data_lock:
            dati_copia = list(manager.dati_da_salvare)
            manager.dati_da_salvare.clear()
        if dati_copia:
            with open(NOME_FILE_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(dati_copia)
    else:
        manager.is_recording = False
        with manager.data_lock:
            manager.dati_da_salvare.clear()

    # --- ESTRAZIONE DATI ---
    with manager.data_lock:
        ecg_data = list(manager.ecg_history)
        imu_data = list(manager.imu_history)
        bpm_current = manager.bpm_display
        activity_current = manager.activity_level
        ecg_status = manager.ecg_status
        imu_status = manager.imu_status
        rr_list = list(manager.rr_intervals)

    # --- CALCOLO FEATURE E INFERENZA ---
    feat = calcola_feature_live(rr_list, imu_data, bpm_current)
    rmssd_display = int(feat[1]) if feat else 0

    if modello_ia is not None and ecg_status == "✅ ECG Connesso" and imu_status == "✅ IMU Connesso":
        if activity_current > 150:
            manager.stato_fisiologico = "🏃 IN MOVIMENTO (Analisi Emozioni Sospesa)"
        elif feat is None:
            manager.stato_fisiologico = "⏳ Raccolta dati HRV in corso..."
        else:
            feature_attuali = pd.DataFrame([feat], columns=FEATURE_COLS)
            predizione = modello_ia.predict(feature_attuali)[0]
            if predizione == 0:
                manager.stato_fisiologico = "🟢 RIPOSO (Baseline)"
            elif predizione == 1:
                manager.stato_fisiologico = "🔴 STRESS COGNITIVO RILEVATO"
            elif predizione == 2:
                manager.stato_fisiologico = "🎉 DIVERTIMENTO / AMUSEMENT"
    else:
        manager.stato_fisiologico = "⏳ Attesa Dati..."

    # --- RENDERIZZAZIONE ---
    st.subheader(f"Stato Mentale/Fisico (AI): {manager.stato_fisiologico}")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Frequenza Cardiaca", f"{bpm_current} BPM", ecg_status)
    with c2:
        st.metric("Variabilità Cardiaca (HRV)", f"{rmssd_display} ms", "RMSSD")
    with c3:
        st.metric("Attività Motoria", activity_current, imu_status)

    cg1, cg2 = st.columns(2)
    ADC_TO_MV = (2.42 / 4) / (2**23) * 1000  # ≈ 0.0000721 mV/LSB

    ecg_data_mv = [v * ADC_TO_MV for v in ecg_data]
    with cg1:
        st.line_chart(ecg_data_mv, height=300)
    with cg2:
        st.area_chart(imu_data, height=300)

    time.sleep(0.5)
    st.rerun()


if __name__ == "__main__":
    main()