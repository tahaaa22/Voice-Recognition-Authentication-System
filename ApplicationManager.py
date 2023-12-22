import sounddevice as sd
import soundfile as sf
import librosa as lb
from PyQt5.QtGui import QPixmap


class ApplicationManger:
    def __init__(self, ui):
        self.ui = ui
        self.fingerprint_mode = False
        self.recorded_voice_text = ""
        self.recorded_voice = None
        self.pass_sentences = ["grant me access", "open middle door", "unlock the gate"]
        self.pass_sentences_progress_bars = [ui.grant_progressBar, ui.open_progressBar, ui.unlock_progressBar]
        self.right_mark_icon = QPixmap("Assets/Correct.png")
        self.right_mark_icon = self.right_mark_icon.scaledToWidth(50)
        self.wrong_mark_icon = QPixmap("Assets/Wrong.png")
        self.wrong_mark_icon = self.wrong_mark_icon.scaledToWidth(50)

    def switch_modes(self):
        self.fingerprint_mode = not self.fingerprint_mode 
    
    def record_voice(self):
        if self.ui.Public_RadioButton.isChecked:
            duration = 3  # seconds
            sampling_frequency = 44100
            self.recorded_voice = sd.rec(frames=int(sampling_frequency*duration), samplerate=sampling_frequency,
                                         channels=1, blocking=True, dtype='int16')
            sf.write("output.wav", self.recorded_voice, sampling_frequency)
            self.recorded_voice, sampling_frequency = lb.load("output.wav")
            self.ui.SpectrogramWidget.canvas.plot_spectrogram(self.recorded_voice, sampling_frequency)
            print(self.recorded_voice)

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
