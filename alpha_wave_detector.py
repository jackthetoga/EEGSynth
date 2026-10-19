"""Live alpha indicator for OpenBCI Cyton with motion rejection."""
import time
import tkinter as tk
from tkinter import messagebox

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

CHANNEL = 7
MOTION_THRESHOLD = 0.12  # Acceleration deviation in g; tune experimentally
MIN_ACCEL_SAMPLES = 8


class AlphaGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cyton Alpha Monitor")
        self.root.geometry("390x290")

        self.board = None
        self.board_id = BoardIds.CYTON_BOARD.value
        self.fs = BoardShim.get_sampling_rate(self.board_id)
        self.n = int(2 * self.fs)

        self.eeg_row = BoardShim.get_eeg_channels(self.board_id)[CHANNEL - 1]
        self.accel_rows = BoardShim.get_accel_channels(self.board_id)

        self.freqs = np.fft.rfftfreq(self.n, d=1 / self.fs)
        self.window = np.hanning(self.n)

        self.baseline = None
        self.cal_start = None
        self.cal_end = None
        self.cal_values = []
        self.closed = False

        top = tk.Frame(root)
        top.pack(pady=15)

        tk.Label(top, text="Cyton port:").pack(side="left")
        self.port = tk.Entry(top, width=9)
        self.port.insert(0, "COM3")
        self.port.pack(side="left", padx=8)

        self.connect_btn = tk.Button(top, text="Connect", command=self.connect)
        self.connect_btn.pack(side="left")

        self.indicator = tk.Label(
            root, text="Disconnected", font=("Segoe UI", 23, "bold"),
            fg="white", bg="#3b3e43", width=20, height=3
        )
        self.indicator.pack(padx=20, pady=8, fill="x")

        self.cal_btn = tk.Button(
            root, text="Calibrate (eyes open)",
            command=self.calibrate, state="disabled"
        )
        self.cal_btn.pack(pady=4)

        self.status = tk.Label(
            root, text="Connect, then calibrate with your eyes open.",
            wraplength=360
        )
        self.status.pack(pady=9)

        root.protocol("WM_DELETE_WINDOW", self.close)

    def connect(self):
        params = BrainFlowInputParams()
        params.serial_port = self.port.get().strip()

        try:
            self.board = BoardShim(self.board_id, params)
            self.board.prepare_session()
            self.board.config_board("1234568")
            self.board.config_board("&")
            self.board.config_board("/0")
            self.board.start_stream()
        except Exception as exc:
            self.disconnect()
            messagebox.showerror("Connection failed", str(exc))
            return

        self.connect_btn.config(state="disabled")
        self.cal_btn.config(state="normal")
        self.status.config(text="Collecting EEG and accelerometer data.")
        self.root.after(250, self.update)

    def calibrate(self):
        self.baseline = None
        self.cal_values = []
        self.cal_start = time.monotonic() + 2
        self.cal_end = self.cal_start + 5
        self.cal_btn.config(state="disabled")

    def motion_score(self, data):
        # Cyton accelerometer samples are sparse (about 25 Hz).
        # BrainFlow fills unused accelerometer slots with zeros.
        accel = data[self.accel_rows].T
        accel = accel[np.all(np.isfinite(accel), axis=1)]
        accel = accel[np.any(accel != 0, axis=1)]

        if len(accel) < MIN_ACCEL_SAMPLES:
            return None

        # Measure deviations from the typical acceleration vector.
        center = np.median(accel, axis=0)
        deviation = np.linalg.norm(accel - center, axis=1)

        return float(np.percentile(deviation, 95))

    def update(self):
        if self.closed or self.board is None:
            return

        try:
            data = self.board.get_current_board_data(self.n)

            if data.shape[1] == self.n:
                motion = self.motion_score(data)

                if motion is None:
                    self.indicator.config(
                        text="NO ACCEL DATA", bg="#87612e"
                    )
                    self.status.config(
                        text="Cannot verify motion. Alpha reading paused."
                    )

                elif motion > MOTION_THRESHOLD:
                    self.indicator.config(
                        text="MOVEMENT", bg="#a34b32"
                    )
                    self.status.config(
                        text=f"Motion: {motion:.3f} g. "
                             "Alpha reading rejected."
                    )

                else:
                    x = data[self.eeg_row]
                    spectrum = np.abs(
                        np.fft.rfft((x - x.mean()) * self.window)
                    ) ** 2

                    broadband = spectrum[
                        (self.freqs >= 4) & (self.freqs <= 30)
                    ].sum()

                    if broadband > 0:
                        alpha = spectrum[
                            (self.freqs >= 8) & (self.freqs <= 13)
                        ].sum()
                        self.display(float(alpha / broadband))

        except Exception as exc:
            self.status.config(text=f"Read error: {exc}")
            return

        self.root.after(250, self.update)

    def display(self, fraction):
        now = time.monotonic()

        if self.cal_end is not None:
            if self.cal_start <= now < self.cal_end:
                self.cal_values.append(fraction)

            if now >= self.cal_end:
                if len(self.cal_values) >= 8:
                    self.baseline = float(np.median(self.cal_values))
                    self.status.config(
                        text="Now close your eyes. Green = increased alpha."
                    )
                else:
                    self.baseline = None
                    self.status.config(
                        text="Calibration failed: insufficient clean data. Retry."
                    )

                self.cal_end = None
                self.cal_btn.config(state="normal")

            else:
                seconds = max(0, int(self.cal_end - now) + 1)

                if now < self.cal_start:
                    self.indicator.config(
                        text="Preparing...", bg="#555d6a"
                    )
                else:
                    self.indicator.config(
                        text=f"Eyes OPEN\n{seconds}s", bg="#555d6a"
                    )

                self.status.config(
                    text="Calibrating. Keep your head still."
                )
                return

        if self.baseline is None or self.baseline <= 0:
            self.indicator.config(
                text=f"Alpha: {fraction:.0%}", bg="#3b3e43"
            )
            if self.cal_end is None:
                self.status.config(
                    text="Calibrate with your eyes open."
                )
            return

        gain = fraction / self.baseline
        strength = np.clip((gain - 1) / 1.5, 0, 1)

        color = (
            f"#{int(59 - 25 * strength):02x}"
            f"{int(62 + 140 * strength):02x}"
            f"{int(67 + 15 * strength):02x}"
        )

        self.indicator.config(
            text=f"Alpha: {gain:.1f}x", bg=color
        )
        self.status.config(
            text="Motion rejection active. Green = increased alpha."
        )

    def disconnect(self):
        if self.board is not None:
            try:
                self.board.stop_stream()
            except Exception:
                pass
            try:
                self.board.release_session()
            except Exception:
                pass
            self.board = None

    def close(self):
        self.closed = True
        self.disconnect()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    AlphaGUI(root)
    root.mainloop()


