import speech_recognition as sr
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QPixmap


class ApplicationManger:
    def __init__(self, ui):
        self.ui = ui
        self.fingerprint_mode = False
        self.voice_recorder = sr.Recognizer()
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
            while True:
                with sr.Microphone() as microphone:
                    print("Say something...")
                    self.recorded_voice = self.voice_recorder.listen(microphone)
                try:
                    print("Recognizing...")
                    self.recorded_voice_text = self.voice_recorder.recognize_google(self.recorded_voice)
                    self.recorded_voice_text = self.recorded_voice_text.lower()
                    print(f"You said: {self.recorded_voice_text}")
                    self.display_text()
                    self.check_pass_sentence()
                    break
                except sr.UnknownValueError:
                    print("Could not understand audio, Please try again")
                except sr.RequestError as e:
                    print(f"Error with the request to Google Web Speech API; {0} {e}")

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
