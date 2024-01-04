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
        self.pass_sentences_progress_bars = [ui.grant_progressBar, ui.open_progressBar, ui.unlock_progressBar]
        self.right_mark_icon = QPixmap("Assets/Correct.png")
        self.right_mark_icon = self.right_mark_icon.scaledToWidth(50)
        self.wrong_mark_icon = QPixmap("Assets/Wrong.png")
        self.wrong_mark_icon = self.wrong_mark_icon.scaledToWidth(50)
        self.features_array = None
        self.database_features_array = []
        self.file_names = []
        self.c = 1

    def switch_modes(self):
        self.fingerprint_mode = not self.fingerprint_mode 
    
    def create_database(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            for name in ("Omar","Hazem"):
                #for word in ("Key"):
                    for i in range(1, 31):
                        self.calculate_sound_features(f"Voice Dataset/{name}_Key ({i}).ogg")
                        
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
        self.check_matching([0,0,0])
        if self.ui.Public_RadioButton.isChecked:
            duration = 3  # seconds
            
            self.recorded_voice = sd.rec(frames=int(self.sampling_frequency*duration), samplerate=self.sampling_frequency,
                                         channels=1, blocking=True, dtype='int16')
            sf.write(f"Voice Dataset/Hazem_Access ({self.c}).ogg", self.recorded_voice, self.sampling_frequency)
            self.recorded_voice, sampling_frequency = lb.load("output.ogg")
            self.ui.SpectrogramWidget.canvas.plot_spectrogram(self.recorded_voice, sampling_frequency)

            print(f"Voice Dataset/Hazem_Access ({self.c}).ogg")
            self.c += 1

            # self.calculate_sound_features("output.ogg",False)
            # model = self.train_model()
            # rf_probabilities = model.predict_proba(self.features_array.reshape(1,-1))
            # rf_predictions = model.predict(self.features_array.reshape(1,-1))


            # print("Probability Scores for the First Test Sample:")
            # print(rf_probabilities)
            # print(rf_predictions)
            #self.check_matching(rf_probabilities[0])
            
            #print(model.predict(self.features_array.reshape(1,-1)))
            #print(model2.predict(self.features_array.reshape(1,-1)))
            # accuracy = model.score(X_test, y_test)
            # print(accuracy)

    def check_matching(self,probs):
        for i in range(len(self.pass_sentences_progress_bars)):
            self.pass_sentences_progress_bars[i].setValue(int(probs[i]*100))

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
