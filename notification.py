import threading
import time
from datetime import datetime
import winsound


def play_alarm(duration_ms=500, frequency_hz=1000, repeat_count=3):
    """
    Putar alarm suara menggunakan winsound (Windows built-in).
    
    Args:
        duration_ms: Durasi setiap beep dalam milidetik (default 500ms)
        frequency_hz: Frekuensi suara dalam Hz (default 1000 Hz)
        repeat_count: Jumlah pengulangan alarm (default 3x)
    """
    print("\n" + "="*80)
    print("🎉 TRAINING SELESAI!")
    print("="*80)
    
    for i in range(repeat_count):
        winsound.Beep(frequency_hz, duration_ms)
        if i < repeat_count - 1:
            import time
            time.sleep(0.3)  # Delay antar beep
    
    print(f"⏰ Waktu Selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


def play_alert_sequence():
    """
    Putar sequence alarm dengan variasi frekuensi (lebih menarik).
    """
    print("\n" + "="*80)
    print("🎉 TRAINING SELESAI!")
    print("="*80)
    
    sequence = [1000, 1200, 1400]  # Frekuensi naik
    for freq in sequence:
        winsound.Beep(freq, 400)
        import time
        time.sleep(0.2)
    
    print(f"⏰ Waktu Selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


def play_alert_until_action():
    """
    Putar alarm terus menerus sampai user tekan Enter.
    Menggunakan threading untuk beep tanpa henti sampai input.
    """
    print("\n" + "="*80)
    print("🎉 TRAINING SELESAI!")
    print("Alarm akan terus bunyi. Tekan Enter untuk hentikan...")
    print("="*80)
    
    # Flag untuk hentikan beep
    stop_beep = threading.Event()
    
    def beep_forever():
        while not stop_beep.is_set():
            winsound.Beep(1000, 500)
            time.sleep(1)
    
    # Start thread beep
    beep_thread = threading.Thread(target=beep_forever)
    beep_thread.start()
    
    # Tunggu input untuk hentikan
    input("Tekan Enter untuk hentikan alarm: ")
    stop_beep.set()
    beep_thread.join()
    
    print("Alarm dihentikan.")
    print(f"⏰ Waktu Selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Untuk tes: ubah ke play_alert_until_action() untuk tes alarm terus
    play_alert_until_action()
