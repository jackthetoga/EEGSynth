# EEGSynth

Turns OpenBCI Cyton EEG into MIDI to control a synth in Logic Pro, with a live
browser dashboard showing every signal and detection.

```
Cyton ))) USB dongle -> Mac: eegsynth -> "EEGSynth" virtual MIDI port -> Logic Pro
                                     \-> http://localhost:8765 (live GUI)
```

## Detectors

| Gesture | Signal | Electrodes | MIDI (default) |
|---------|--------|------------|----------------|
| Jaw clench | 30–55 Hz muscle (EMG) burst, graded by strength | T7, T8 | CC23 = strength, note 36 held (velocity = strength) |
| Eyebrow raise | Frontal EMG, distinguished from jaw by location | Fp1, Fp2 | CC24 = strength, note 38 held |
| Blink | Large spike on the forehead (EOG) | Fp1, Fp2 | single: note 40, double: note 41 |
| Look left / right | Slow step in temple voltage, polarity = direction | F7, F8 | left: note 43, right: note 45 |
| Alpha | 8–13 Hz share of 4–30 Hz power, smoothed (3 s), vs. your calibrated eyes-open baseline; rejected during head movement | O1 | one long note (C3/60) whose pitch bends up with alpha, plus CC1 |

Everything (channels, thresholds, CC and note numbers) is in `config.toml`.

## Electrode placement

| Cyton pin | Site | Where |
|-----------|------|-------|
| N1P | Fp1 | forehead, above left eye |
| N2P | Fp2 | forehead, above right eye |
| N3P | F7 | left temple, level with the outer eye corner |
| N4P | F8 | right temple, level with the outer eye corner |
| N5P | T7 | above left ear (temporalis muscle) |
| N6P | T8 | above right ear |
| N7P | O1 | back of head, left of centre |
| N8P | O2 | back of head, right of centre |
| SRB2 (bottom) | reference | left earlobe |
| BIAS | ground | right earlobe |

## Setup (Mac)

```bash
cd EEGSynth
uv venv --python 3.12 .venv && uv pip install -e '.[midi]'
```

1. Plug the Cyton USB dongle into the Mac, set its switch to **GPIO_6**, and switch the Cyton board to **PC**.
2. Start it:

   ```bash
   .venv/bin/eegsynth run
   ```

   The dongle is auto-detected (`/dev/cu.usbserial-*`; pass `--port` to override), the
   `EEGSynth` MIDI port is created and the dashboard opens in your browser.
3. Sit still and relaxed for the first few seconds while the detectors learn your resting levels.
4. In the dashboard, check every channel's label shows a green µV value (not `railed`/`noisy`).
   Then try each gesture and watch its card light up.

No headset handy? `--source synthetic` plays a scripted 40-second loop that performs every
gesture (the script is printed at startup).

## Logic Pro

1. If Logic is already open, it picks up the new port automatically; check
   **Logic Pro > Settings > MIDI > Inputs** shows `EEGSynth` enabled.
2. Select or record-arm a Software Instrument track with a sustaining sound (pad, organ).
   In the GUI press **Calibrate (eyes open)**: 2 s to settle, then 5 s with eyes open and
   head still. When it finishes, a long note starts; closing your eyes bends it up
   (full bend at 2.5x your baseline). The synth's **pitch-bend range** sets the interval,
   e.g. 12 semitones for an octave. **Restart note** retriggers it, **Stop note** ends it.
   Recalibrate whenever you reposition the headset.
3. Notes from the detectors **play the instrument** by default (a jaw clench plays C1 with
   velocity = how hard you clenched). To use a gesture as a switch instead, assign it with
   **Logic Pro > Control Surfaces > Learn Assignment** (Cmd+L): click a control in Logic,
   then do the gesture. Learned messages are consumed and no longer play notes.
4. While learning, restrict output to one detector so Logic captures the right message:

   ```bash
   .venv/bin/eegsynth run --only blink
   ```

   (`--only` accepts `jaw`, `eyebrow`, `blink`, `eyes`, `alpha`, `bands`, comma-separated.)
   This sets the starting state of the **Mute** buttons on each GUI card, which you can
   toggle live. Muting releases any notes that detector is holding.

## Noise, artifacts and the accelerometer

Every detector input passes through three layers of protection (`[motion]` and `[artifacts]` in `config.toml`):

1. **Motion cancellation.** An adaptive (NLMS) filter learns how head acceleration
   leaks into each channel and subtracts it. It only helps if the Cyton board is
   mounted on the headset, and it removes motion-correlated drift, not electrode pops.
2. **Motion gate.** While the accelerometer shows head movement (and 0.8 s after),
   all detectors pause. Blink and eye events also wait 150 ms and are dropped if
   movement or an artifact shows up right after them.
3. **Artifact monitor.** Each channel is blocked while it shows an electrode pop
   (sudden jump), an excursion too large to be physiological, saturation, or a
   flatline, plus 1 s afterwards. Blocked channels are shaded red in the dashboard
   and each detector card says why it's blocked.

Jaw clench and eyebrow raise additionally require activity on **both** of their
electrodes, a muscle-shaped spectrum (more 30–55 Hz than 1–10 Hz), a physiological
amplitude, at least 0.2 s duration, and 0.3 s between clenches.

None of this can create signal that isn't there. If the dashboard or the startup
contact report says `railed` or `no signal`:

- Check the **SRB2 (reference)** and **BIAS (ground)** ear clips first; if either is
  off, every channel saturates.
- Dry comb electrodes need firm pressure through the hair; part the hair under each one.
  Gel or saline electrodes give far cleaner signals.
- Good contact looks like a fuzzy trace of ±10–50 µV with mains hum under ~20 µV.
  Flat lines with occasional huge jumps mean no contact.
- Keep the dongle away from the laptop charger, run on battery, and let cables hang
  without swinging.

## Tips per detector

- **Jaw clench** works first try. Light vs. hard clench gives a clear velocity/CC difference.
- **Eyebrow raise** must be stronger on the forehead than at the temples; if it registers as
  jaw, lower `dominance` in `[eyebrow]`.
- **Blinks**: use the double blink as a deliberate trigger; single blinks happen involuntarily
  and are reported 0.6 s late (the detector waits to see if a second blink follows).
- **Eye movements**: look firmly to one side and back to centre; only the move away from centre
  triggers. If left and right are swapped, set `invert = true` in `[eyes]`. Too few
  detections: lower `threshold_uv`. False ones: raise it.
- **Alpha**: calibrate first, then close your eyes and relax; the gain should climb toward 2× over
  a few seconds (3 s smoothing). "MOVEMENT" means the accelerometer score exceeded
  `motion_threshold_g`; tune it experimentally.
  The baseline is the 20th percentile of the last 90 s, so start with your eyes open.

## Other commands

- `eegsynth check`: per-channel signal quality in the terminal.
- `eegsynth ports`: list MIDI outputs.
- `eegsynth run --udp HOST:9000` + `eegsynth bridge` on the Mac: run the reader on another
  machine (e.g. a Raspberry Pi, see `pi/`) and send MIDI over the network.

## Notes

- Only the 8-channel Cyton is supported (not the Daisy 16-channel add-on), at the default gain (24×).
- Set `mains_hz = 50` in `config.toml` outside the Americas.
# EEGSynth
# EEGSynth
# EEGSynth
