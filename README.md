Voice Authentication System
Welcome to the Voice Authentication System repository! This Python desktop application utilizes voice fingerprint and spectrogram technology to identify individuals based on their unique vocal characteristics. The system can be trained on up to 8 individuals and operates in two distinct modes to ensure secure access.

image

Features
Mode 1 – Security Voice Code:

Access is granted only if the user speaks a specific pass-code sentence.
Mode 2 – Security Voice Fingerprint:

Access is granted to specific individuals who say the valid pass-code sentence.
The program calculates matching probabilities and decides whether access should be granted or denied based on the input.

Installation
To run the Voice Authentication System, make sure you have the following libraries installed:

pip install PyQt5 matplotlib sounddevice soundfile librosa numpy scikit-learn
Usage
Clone the repository:
git clone https://github.com/MoHazem02/Voice-Authentication-System.git
Navigate to the project directory:
cd Voice-Authentication-System
Run the application:
python Voice_Recognizer.py
Dependencies
PyQt5
Matplotlib
Sounddevice
Soundfile
Librosa
Numpy
Scikit-learn
Contributing
If you would like to contribute to the project, please follow the contribution guidelines.

Acknowledgments
Special thanks to the developers of the libraries used in this project.

Happy coding!
