import io
import math
import os
import struct
import tempfile
import threading
import time
import wave
from datetime import datetime

import winsound

_SND_SYNC = getattr(winsound, "SND_SYNC", 0)


def _build_tone_frames(
    primary_freq,
    secondary_freq,
    duration_sec,
    sample_rate=44100,
    volume=1.5,
):
    """Generate PCM frames for a dense dual-tone alarm segment."""
    frame_count = max(1, int(sample_rate * duration_sec))
    attack = max(1, int(sample_rate * 0.01))
    release = max(1, int(sample_rate * 0.03))
    frames = bytearray()

    for i in range(frame_count):
        t = i / sample_rate

        envelope = 1.0
        if i < attack:
            envelope = i / attack
        elif i > frame_count - release:
            envelope = max(0.0, (frame_count - i) / release)

        pulse = 0.72 + 0.28 * math.sin(2 * math.pi * 2.4 * t)
        tone = (
            0.60 * math.sin(2 * math.pi * primary_freq * t)
            + 0.25 * math.sin(2 * math.pi * secondary_freq * t)
            + 0.10 * math.sin(2 * math.pi * primary_freq * 2 * t)
            + 0.05 * math.sin(2 * math.pi * secondary_freq * 2 * t)
        )

        sample = max(-1.0, min(1.0, tone * envelope * pulse * volume))
        pcm = int(sample * 32767)
        frames.extend(struct.pack("<h", pcm))

    return bytes(frames)


def _build_silence_frames(duration_sec, sample_rate=44100):
    frame_count = max(1, int(sample_rate * duration_sec))
    return b"\x00\x00" * frame_count


def _build_alarm_wave(sequence, sample_rate=44100):
    """Build an in-memory WAV buffer for playback with winsound."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        for segment in sequence:
            if segment["type"] == "tone":
                wav_file.writeframes(
                    _build_tone_frames(
                        primary_freq=segment["primary"],
                        secondary_freq=segment["secondary"],
                        duration_sec=segment["duration"],
                        sample_rate=sample_rate,
                        volume=segment.get("volume", 0.9),
                    )
                )
            else:
                wav_file.writeframes(
                    _build_silence_frames(
                        duration_sec=segment["duration"],
                        sample_rate=sample_rate,
                    )
                )

    return buffer.getvalue()


def _play_beep_sequence(sequence):
    for segment in sequence:
        if segment["type"] == "tone":
            frequency = max(37, min(32767, int(segment["primary"])))
            duration_ms = max(80, int(segment["duration"] * 1000))
            winsound.Beep(frequency, duration_ms)
        else:
            time.sleep(max(0.02, float(segment["duration"])))


def _play_wave(wave_bytes, fallback_sequence=None):
    try:
        winsound.PlaySound(wave_bytes, winsound.SND_MEMORY | _SND_SYNC)
        return
    except RuntimeError:
        pass

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(wave_bytes)
            temp_path = temp_file.name
        winsound.PlaySound(temp_path, winsound.SND_FILENAME | _SND_SYNC)
        return
    except RuntimeError:
        pass
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    if fallback_sequence:
        _play_beep_sequence(fallback_sequence)


def _default_alarm_sequence():
    """Phone-like alarm pattern with short pauses and layered tones."""
    return [
        {"type": "tone", "primary": 880, "secondary": 1320, "duration": 0.32, "volume": 0.95},
        {"type": "silence", "duration": 0.08},
        {"type": "tone", "primary": 988, "secondary": 1480, "duration": 0.32, "volume": 0.95},
        {"type": "silence", "duration": 0.08},
        {"type": "tone", "primary": 784, "secondary": 1175, "duration": 0.40, "volume": 0.98},
        {"type": "silence", "duration": 0.18},
    ]


def play_alarm(duration_ms=500, frequency_hz=1000, repeat_count=3):
    """
    Putar alarm suara dengan karakter lebih tebal daripada beep biasa.

    Args:
        duration_ms: Durasi setiap bunyi dalam milidetik.
        frequency_hz: Frekuensi dasar suara.
        repeat_count: Jumlah pengulangan alarm.
    """
    print("\n" + "=" * 80)
    print("TRAINING SELESAI!")
    print("=" * 80)

    duration_sec = max(0.15, duration_ms / 1000.0)
    sequence = []
    total_repeat = max(1, repeat_count)

    for i in range(total_repeat):
        sequence.append(
            {
                "type": "tone",
                "primary": max(300, frequency_hz),
                "secondary": max(450, int(frequency_hz * 1.5)),
                "duration": duration_sec,
                "volume": 0.95,
            }
        )
        if i < total_repeat - 1:
            sequence.append({"type": "silence", "duration": 0.12})

    _play_wave(_build_alarm_wave(sequence), fallback_sequence=sequence)

    print(f"Waktu Selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


def play_alert_sequence():
    """Putar alarm yang lebih keras dan lebih mirip pola alarm HP."""
    print("\n" + "=" * 80)
    print("TRAINING SELESAI!")
    print("=" * 80)

    sequence = _default_alarm_sequence()
    _play_wave(_build_alarm_wave(sequence), fallback_sequence=sequence)

    print(f"Waktu Selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


def play_error_alert_sequence():
    """Putar alarm khusus saat training gagal atau error."""
    print("\n" + "=" * 80)
    print("TRAINING GAGAL / ERROR!")
    print("=" * 80)

    error_sequence = [
        {"type": "tone", "primary": 620, "secondary": 930, "duration": 0.35, "volume": 0.98},
        {"type": "silence", "duration": 0.08},
        {"type": "tone", "primary": 620, "secondary": 930, "duration": 0.35, "volume": 0.98},
        {"type": "silence", "duration": 0.10},
        {"type": "tone", "primary": 520, "secondary": 780, "duration": 0.55, "volume": 1.0},
    ]
    _play_wave(_build_alarm_wave(error_sequence), fallback_sequence=error_sequence)

    print(f"Waktu Error: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


def play_alert_until_action():
    """Putar alarm terus menerus sampai user menekan Enter."""
    print("\n" + "=" * 80)
    print("TRAINING SELESAI!")
    print("Alarm akan terus bunyi. Tekan Enter untuk hentikan...")
    print("=" * 80)

    stop_beep = threading.Event()
    alarm_wave = _build_alarm_wave(_default_alarm_sequence())

    def alarm_forever():
        while not stop_beep.is_set():
            _play_wave(alarm_wave, fallback_sequence=_default_alarm_sequence())
            time.sleep(0.15)

    beep_thread = threading.Thread(target=alarm_forever)
    beep_thread.start()

    input("Tekan Enter untuk hentikan alarm: ")
    stop_beep.set()
    beep_thread.join()

    print("Alarm dihentikan.")
    print(f"Waktu Selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    play_alert_until_action()
