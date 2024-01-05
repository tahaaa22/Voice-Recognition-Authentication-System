from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
import sounddevice as sd
import soundfile as sf
import librosa as lb
import numpy as np
from numpy import mean, var
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings

class ApplicationManger:
    def __init__(self, ui):
        self.ui = ui
        self.fingerprint_mode = False
        self.recorded_voice_text = ""
        self.recorded_voice = None
        self.sampling_frequency = 44100
        
        self.pass_sentences = ["grant me access", "open middle door", "unlock the gate"]
        self.pass_sentences_progress_bars = [ui.Access_progressBar, ui.Door_progressBar, ui.Key_progressBar]
        self.people_progress_bars = [ui.Hazem_Bar,ui.Omar_Bar,ui.Taha_Bar,ui.Youssef_Bar]
        
        self.features_array = None
        self.database_features_array = []
        self.file_names = []
        self.c = 1

        self.right_mark_icon = QPixmap("Assets/Correct.png").scaledToWidth(50)
        self.wrong_mark_icon = QPixmap("Assets/Wrong.png").scaledToWidth(50)
        self.icons = [[self.wrong_mark_icon,"Denied"],[self.right_mark_icon,"Authorized"]]

    def switch_modes(self):
        self.fingerprint_mode = not self.fingerprint_mode 
    
    def create_database(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            for name in ("Hazem","Omar","Taha","Youssef"):
                for word in ("Access","Door","key"):
                    for i in range(1, 31):
                        self.calculate_sound_features(f"Voice Dataset/{name}_{word} ({i}).ogg")
                            
            df = pd.DataFrame({
                'Features': self.database_features_array,
                'result': self.file_names
            })
            df.to_csv('Dataset.csv', index=False)

    def calculate_sound_features(self, file_path, database_flag=True):
        log_mel_spectrogram_mean = []
        log_mel_spectrogram_var = []
        mfccs_mean = []
        mfccs_var = []
        cqt_mean = []
        cqt_var = []
        chroma_mean = []
        chroma_var = []
        tone_mean = []
        tone_var = []

        voice_data, sampling_frequency = lb.load(file_path)
        mfccs = lb.feature.mfcc(y=voice_data, sr=sampling_frequency, n_fft=256, hop_length=64, n_mels=13)
        chroma = lb.feature.chroma_stft(y=voice_data, sr=sampling_frequency, n_fft=256, hop_length=64)
        log_mel_spectrogram = lb.power_to_db(
            lb.feature.melspectrogram(y=voice_data, sr=sampling_frequency, n_fft=256, hop_length=64, n_mels=13))
        constant_q_transform = np.abs(lb.cqt(y=voice_data, sr=sampling_frequency))
        tone = lb.feature.tonnetz(y=voice_data, sr=sampling_frequency)
        spectral_bandwidth = lb.feature.spectral_bandwidth(y=voice_data, sr=sampling_frequency, n_fft=256, hop_length=64)
        amplitude_envelope = self.calculate_amplitude_envelope(voice_data, 256, 64)
        root_mean_square = lb.feature.rms(y=voice_data, frame_length=256, hop_length=64)
        filename = file_path[14:23]


        for i in range(len(log_mel_spectrogram)):
            log_mel_spectrogram_mean.append(log_mel_spectrogram[i].mean())
            log_mel_spectrogram_var.append(log_mel_spectrogram[i].var())

        for i in range(len(mfccs)):
            mfccs_mean.append(mfccs[i].mean())
            mfccs_var.append(mfccs[i].var())

        for i in range(len(constant_q_transform)):
            cqt_mean.append(constant_q_transform[i].mean())
            cqt_var.append(constant_q_transform[i].var())

        for i in range(len(chroma)):
            chroma_mean.append(chroma[i].mean())
            chroma_var.append(chroma[i].var())

        #Calculate mean and variance of each frame of tone
        for i in range(len(tone)):
            tone_mean.append(tone[i].mean())
            tone_var.append(tone[i].var())
         
        self.features_array = np.hstack((
        chroma_mean, chroma_var, cqt_mean, cqt_var, mfccs_mean,
                                        mfccs_var, log_mel_spectrogram_mean, log_mel_spectrogram_var))
        
        # mean(amplitude_envelope), var(amplitude_envelope), mean(root_mean_square),
        # var(root_mean_square), mean(spectral_bandwidth), var(spectral_bandwidth), tone_mean, tone_var,
        
        if database_flag:
            self.database_features_array.append(self.features_array)
            self.file_names.append(filename)

    def train_model(self):
        # kn_classifier = KNeighborsClassifier(n_neighbors=5)
        # kn_classifier.fit(self.database_features_array, self.file_names)

        # dt_classifier = DecisionTreeClassifier(random_state=42)
        # dt_classifier.fit(self.database_features_array, self.file_names)

        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        result = rf_classifier.fit(self.database_features_array, self.file_names)

        return result

    def record_voice(self):
        if self.ui.Public_RadioButton.isChecked:
            duration = 3  # seconds
            
            self.recorded_voice = sd.rec(frames=int(self.sampling_frequency*duration), samplerate=self.sampling_frequency,
                                         channels=1, blocking=True, dtype='int16')
            sf.write("output.ogg", self.recorded_voice, self.sampling_frequency)
            self.recorded_voice, sampling_frequency = lb.load("output.ogg")
            self.ui.SpectrogramWidget.canvas.plot_spectrogram(self.recorded_voice, sampling_frequency)
            
            self.calculate_sound_features("output.ogg",False)
            model = self.train_model()
            rf_probabilities = model.predict_proba(self.features_array.reshape(1,-1))
            rf_predictions = model.predict(self.features_array.reshape(1,-1))

            print("Probability Scores for the First Test Sample:")
            print(rf_probabilities)
            print(rf_predictions)

            self.check_matching(rf_probabilities[0])

    def check_matching(self,probs):
        for i in range(3):
            sum = 0
            for j in range(4):
                sum += probs[i + j*3]    
            self.pass_sentences_progress_bars[i].setValue(int(sum*100))

        for i in range(4):
            sum = 0
            for j in range(3):
                sum += probs[i*3 : i*3 + 2]
            self.people_progress_bars[i].setValue(int(sum*100))

    def display_text(self):
        self.ui.VoiceRecognizedLabel.setText(self.recorded_voice_text)

    def set_icon(self,flag):
        self.ui.AccessLabel.setPixmap(self.icons[flag][0])
        self.ui.label_6.setText(f"Access {self.icons[flag][1]}")

    @staticmethod
    def formatting_features_lists(list_to_be_formatted: list):
        formatted_list = []
        for outer_list in list_to_be_formatted:
            for inner_lists in outer_list:
                for value in inner_lists:
                    formatted_list.append(value)
        return formatted_list

    @staticmethod
    def calculate_amplitude_envelope(audio, frame_length, hop_length):
        return np.array([max(audio[i:i + frame_length]) for i in range(0, len(audio), hop_length)])
