import serial
import time
import threading
import sys
import io
import contextlib

from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket
from pyshimmer.bluetooth.bt_api import BluetoothRequestHandler
from pyshimmer.dev.channels import ESensorGroup
from pyshimmer.util import fmt_hex
from collections import deque

# ============================================================
# PATCH RUNTIME
# ============================================================
def _patched_process_resp_from_queue(self):
    try:
        cmd, return_obj = self._resp_queue.get_nowait()
    except Exception:
        return
    resp_code = cmd.get_response_code()
    peek = self._serial.peek(len(resp_code))
    if peek != resp_code:
        raise ValueError(f"Expecting {fmt_hex(resp_code)} but found {fmt_hex(peek)}")
    result = cmd.receive(self._serial)
    return_obj.set_result(result)


BluetoothRequestHandler._process_resp_from_queue = _patched_process_resp_from_queue


def _patched_process_ack(self):
    try:
        self._serial.read_ack()
    except Exception:
        pass  

    try:
        compl_obj, cmd_resp_pair = self._ack_queue.get_nowait()
        if None not in cmd_resp_pair:
            self._resp_queue.put_nowait(cmd_resp_pair)
        compl_obj.set_completed()
    except Exception:
        pass


BluetoothRequestHandler._process_ack = _patched_process_ack


def safe_print(messaggio):
    try:
        if not sys.stdout.closed:
            print(messaggio)
            sys.stdout.flush()
    except Exception:
        pass


# ============================================================
# WORKER IMU
# ============================================================
def imu_worker(port, manager):
    finestra_imu = deque(maxlen=50) # Memoria per gli ultimi 50 campioni
    while manager.running:
        ser = None
        shim_dev = None
        try:
            safe_print(f"[IMU] Apertura {port}...")

            ser = serial.Serial(port, DEFAULT_BAUDRATE, timeout=None)

            time.sleep(0.5)
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            time.sleep(0.3)
            ser.reset_input_buffer()

            with contextlib.redirect_stderr(io.StringIO()):
                shim_dev = ShimmerBluetooth(ser)
                shim_dev.initialize()

            shim_dev.set_sensors([ESensorGroup.ACCEL_LN])
            manager.imu_status = "✅ IMU Connesso"
            safe_print("[IMU] Inizializzato correttamente.")

            def imu_handler(pkt: DataPacket) -> None:
                with manager.data_lock:
                    manager.ultimo_pacchetto_imu = time.time()

                x, y, z = 0, 0, 0
                found = False
                for ch_key, val in pkt._values.items():
                    ch_str = str(ch_key).upper()
                    if "ACCEL" in ch_str:
                        if "X" in ch_str: x = val
                        elif "Y" in ch_str: y = val
                        elif "Z" in ch_str: z = val
                        found = True

                if found:
                    mov = int((x**2 + y**2 + z**2) ** 0.5)
                    
                    # Aggiungiamo il valore alla nostra finestra mobile
                    finestra_imu.append(mov)
                    
                    # Calcoliamo l'attività solo quando la finestra è piena
                    activity = 0
                    if len(finestra_imu) == 50:
                        # Escursione = differenza tra picco massimo e minimo nella finestra
                        escursione = max(finestra_imu) - min(finestra_imu)
                        # Soglia tarata per camminata (puoi aggiustarla tra 100 e 200)
                        activity = int(escursione) if escursione > 150 else 0

                    with manager.data_lock:
                        manager.activity_level = activity
                        manager.imu_history.append(activity)
                        
                        if getattr(manager, 'is_recording', False):
                            manager.dati_da_salvare.append([
                                time.time(), "IMU", activity, mov, manager.stato_fisiologico
                            ])

            shim_dev.add_stream_callback(imu_handler)
            shim_dev.start_streaming()

            while manager.running:
                with manager.data_lock:
                    last = manager.ultimo_pacchetto_imu
                if time.time() - last > 10.0:
                    raise ConnectionError("Timeout dati IMU")
                time.sleep(0.5)

        except (serial.SerialException, OSError, ConnectionError) as e:
            safe_print(f"[IMU] Errore di Connessione: {e}")
            manager.imu_status = "🔄 Riconnessione IMU..."
        except Exception as e:
            safe_print(f"[IMU] Errore inatteso: {type(e).__name__}: {e}")
        finally:
            if shim_dev:
                try:
                    shim_dev.stop_streaming()
                except Exception:
                    pass
                try:
                    shim_dev.shutdown()
                except Exception:
                    pass
            if ser:
                try:
                    ser.close()
                except Exception:
                    pass
            time.sleep(10)


# ============================================================
# WORKER ECG
# ============================================================
def ecg_worker(port, manager):
    time.sleep(4)

    HEADER = 0x00
    SIZE_PKT = 16
    OFFSET_STATUS = 6
    OFFSET_CH1 = 7  
    OFFSET_CH2 = 10  

    while manager.running:
        ser = None
        try:
            manager.ecg_status = "⏳ Stabilizzazione..."
            safe_print(f"[ECG] Apertura {port}...")

            ser = serial.Serial(port, 115200, timeout=0.1)
            time.sleep(0.5)
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            ser.write(b"\x01")
            time.sleep(0.3)
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            time.sleep(0.3)
            ser.reset_input_buffer()

            ser.write(b"\x07")
            safe_print("[ECG] Streaming avviato.")
            manager.ecg_status = "✅ ECG Connesso"

            with manager.data_lock:
                manager.ultimo_pacchetto_ecg = time.time()

            data_buffer = bytearray()
            val_prec = 0
            ultimo_battito = time.time()
            base_line = None

            while manager.running:
                try:
                    if ser.in_waiting > 0:
                        data_buffer.extend(ser.read(ser.in_waiting))
                        with manager.data_lock:
                            manager.ultimo_pacchetto_ecg = time.time()
                except OSError as e:
                    safe_print(f"[ECG] Errore porta: {e}")
                    break

                while len(data_buffer) >= SIZE_PKT:
                    if data_buffer[0] == HEADER:
                        raw = data_buffer[:SIZE_PKT]
                        status = raw[OFFSET_STATUS]

                        if status == 0x80:
                            val_raw = int.from_bytes(
                                raw[OFFSET_CH1 : OFFSET_CH1 + 3],
                                byteorder="big",
                                signed=True,
                            )

                            if base_line is None:
                                base_line = val_raw
                            else:
                                base_line = 0.99 * base_line + 0.01 * val_raw

                            val_f = val_raw - base_line
                            ora = time.time()
                            delta = abs(val_f - val_prec)

                            if val_f > 8000 and val_prec <= val_f and (ora - ultimo_battito) > 0.5:
                                intervallo_sec = ora - ultimo_battito
                                bpm_instant = 60 / max(0.01, ora - ultimo_battito)
                                bpm_constrained = max(40, min(200, int(bpm_instant)))
                                if 0.3 < intervallo_sec < 2.0:
                                    with manager.data_lock:
                                        manager.bpm_buffer.append(bpm_constrained)
                                        manager.bpm_display = int(
                                            sum(manager.bpm_buffer) / len(manager.bpm_buffer)
                                        )
                                        manager.rr_intervals.append(intervallo_sec)
                                ultimo_battito = ora

                            with manager.data_lock:
                                manager.ecg_history.append(val_f)
                                
                                # --- NUOVO: Salvataggio dati ECG ---
                                if getattr(manager, 'is_recording', False):
                                    manager.dati_da_salvare.append([ora, "ECG", val_f, manager.bpm_display, manager.stato_fisiologico])

                            val_prec = val_f

                        data_buffer = data_buffer[SIZE_PKT:]
                    else:
                        try:
                            next_h = data_buffer.index(HEADER, 1)
                            data_buffer = data_buffer[next_h:]
                        except ValueError:
                            data_buffer.clear()

                time.sleep(0.01)

                with manager.data_lock:
                    last = manager.ultimo_pacchetto_ecg
                if time.time() - last > 5.0:
                    safe_print("[ECG] Timeout dati")
                    break

        except Exception as e:
            safe_print(f"[ECG] Errore: {type(e).__name__}: {e}")
            manager.ecg_status = "🔄 Riconnessione ECG..."
        finally:
            if ser is not None:
                try:
                    if ser.is_open:
                        ser.write(b"\x01")
                        time.sleep(0.1)
                except Exception:
                    pass
                try:
                    ser.close()
                except Exception:
                    pass
            time.sleep(10)


# ============================================================
# AVVIO THREAD
# ============================================================
def start_threads_if_needed(manager, port_imu, port_ecg):
    threads_to_run = [
        ("shimmer-IMU", imu_worker, port_imu),
        ("shimmer-ECG", ecg_worker, port_ecg),
    ]

    active_thread_names = [t.name for t in threading.enumerate() if t.is_alive()]

    for name, worker_func, port in threads_to_run:
        if name not in active_thread_names:
            t = threading.Thread(
                target=worker_func, args=(port, manager), daemon=True, name=name
            )
            t.start()
            safe_print(f"[Sistema] Avviato thread {name}")