import pyaudio
import numpy as np
from PyQt5.QtGui import QPixmap
import acoustid
import wave

class ApplicationManger:
    def __init__(self, ui):
        self.ui = ui
        self.fingerprint_mode = False
        self.audio_input = pyaudio.PyAudio()
        self.recorded_data = None
        self.sample_rate = 44100
        self.fingerprint = None
        
        self.pass_sentences = ["grant me access", "open middle door", "unlock the gate"]
        self.pass_sentences_progress_bars = [ui.grant_progressBar, ui.open_progressBar, ui.unlock_progressBar]
        
        self.right_mark_icon = QPixmap("Assets/Correct.png")
        self.right_mark_icon = self.right_mark_icon.scaledToWidth(50)
        self.wrong_mark_icon = QPixmap("Assets/Wrong.png")
        self.wrong_mark_icon = self.wrong_mark_icon.scaledToWidth(50)

    def switch_modes(self):
        self.fingerprint_mode = not self.fingerprint_mode 
    
    def record_voice(self):
        chunk = 1024
        duration = 3

        stream = self.audio_input.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024
        )

        print("Recording...")

        self.recorded_data = []
        for i in range(0, int(self.sample_rate / chunk * duration)):
            data = np.frombuffer(stream.read(chunk), dtype=np.int16)
            self.recorded_data.append(data)

        print("Finished recording.")
        
        self.ui.SpectrogramWidget.canvas.plot_spectrogram(self.recorded_data, 44100)
        self.save_audio()
        self.fingerprint_audio_file()
        

    def save_audio(self):
        wf = wave.open("audio.wav", 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(self.recorded_data))
        wf.close()
        
    def fingerprint_audio_file(self):
        duration, self.fingerprint = acoustid.fingerprint_file("audio.wav")
        print(duration, self.fingerprint) 

    def display_text(self):
        self.ui.VoiceRecognizedLabel.setText(self.recorded_voice_text)

    def check_pass_sentence(self):
        pixmap_added = False
        for i in range(3):
            if self.recorded_voice_text == self.pass_sentences[i]:
                self.pass_sentences_progress_bars[i].setValue(100)
                self.ui.AccessLabel.setPixmap(self.right_mark_icon)
                self.ui.label_6.setText("Access Authorized")
                pixmap_added = True
            else:
                self.pass_sentences_progress_bars[i].setValue(0)
            if not pixmap_added:
                self.ui.AccessLabel.setPixmap(self.wrong_mark_icon)
                self.ui.label_6.setText("Access Denied")
