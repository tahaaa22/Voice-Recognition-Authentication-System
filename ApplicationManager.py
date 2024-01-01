from PyQt5.QtGui import QPixmap
import sounddevice as sd
import soundfile as sf
import librosa as lb
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

class ApplicationManger:
    def __init__(self, ui):
        self.ui = ui
        self.fingerprint_mode = False
        self.recorded_voice_text = ""
        self.recorded_voice = None
        self.sampling_frequency = 44100
        self.pass_sentences = ["grant me access", "open middle door", "unlock the gate"]
        self.pass_sentences_progress_bars = [ui.grant_progressBar, ui.open_progressBar, ui.unlock_progressBar]
        self.right_mark_icon = QPixmap("Assets/Correct.png")
        self.right_mark_icon = self.right_mark_icon.scaledToWidth(50)
        self.wrong_mark_icon = QPixmap("Assets/Wrong.png")
        self.wrong_mark_icon = self.wrong_mark_icon.scaledToWidth(50)
        
        self.Dataset = []
        self.mfccs = []
        self.chroma = []
        self.spectral_contrast = []
        self.zero_crossings = []
        self.file_names = []

    def switch_modes(self):
        self.fingerprint_mode = not self.fingerprint_mode 
    
    def record_voice(self):
        if self.ui.Public_RadioButton.isChecked:
            duration = 3  # seconds
            
            self.recorded_voice = sd.rec(frames=int(self.sampling_frequency*duration), samplerate=self.sampling_frequency,
                                         channels=1, blocking=True, dtype='int16')
            sf.write("output.wav", self.recorded_voice, self.sampling_frequency)
            self.recorded_voice, sampling_frequency = lb.load("output.wav")
            self.ui.SpectrogramWidget.canvas.plot_spectrogram(self.recorded_voice, sampling_frequency)

            self.extract_features()

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

    def extract_features(self):
        mfccs = lb.feature.mfcc(y = self.recorded_voice, sr = self.sampling_frequency)
        chroma = lb.feature.chroma_stft(y = self.recorded_voice, sr = self.sampling_frequency)
        spectral_contrast = lb.feature.spectral_contrast(y = self.recorded_voice, sr = self.sampling_frequency)
        zero_crossings = lb.feature.zero_crossing_rate(self.recorded_voice)
        test_data = []
        test_data.append(mfccs)
        test_data.append(chroma)
        test_data.append(spectral_contrast)
        test_data.append(zero_crossings)
        
        model = self.train_model()
        prediction = model.predict(test_data)
        print(prediction)
    
    def train_model(self):
        k = KNeighborsClassifier(n_neighbors = 1)
        return k.fit(self.Dataset,self.file_names)

    def calculate_sound_features(self,file_path):
        voice_data, sampling_frequency = lb.load(file_path)
        mfccs = lb.feature.mfcc(y = voice_data, sr = sampling_frequency)
        chroma = lb.feature.chroma_stft(y = voice_data, sr = sampling_frequency)
        spectral_contrast = lb.feature.spectral_contrast(y = voice_data, sr = sampling_frequency)
        zero_crossings = lb.feature.zero_crossing_rate(voice_data)
        filename = file_path[14:23]
        
        data_row = []
        data_row.append(mfccs)
        data_row.append(chroma)
        data_row.append(spectral_contrast)
        data_row.append(zero_crossings)
        self.Dataset.append(data_row)
        
        self.mfccs.append(mfccs)
        self.chroma.append(chroma)
        self.spectral_contrast.append(spectral_contrast)
        self.zero_crossings.append(zero_crossings)
        self.file_names.append(filename)
        
    def calculate_all(self):
        for word in ("Door","Gate","Access"):
            for i in range(1,11):
                self.calculate_sound_features(f"Voice Dataset/Omar_{word} ({i}).ogg")  
        self.save_csv()

    def save_csv(self):
        df = pd.DataFrame({
            'mfccs': self.mfccs,
            'chroma': self.chroma,
            'spectral_contrast': self.spectral_contrast,
            'zero_crossings': self.zero_crossings,
            'result' : self.file_names
        })
        df.to_csv('Dataset.csv', index=False)