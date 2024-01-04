import sounddevice as sd
import soundfile as sf

# Set the desired audio parameters
sample_rate = 44100  # Sample rate in Hz
duration = 3  # Duration of the recording in seconds

# Print a message to indicate when to talk
print("Talk now...")

# Start the recording
recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2)

# Wait for the recording to complete
sd.wait()

# Save the recording as an OGG file
output_filename = "Taha_Access (30).ogg"
sf.write(output_filename, recording, sample_rate)

# Print a message indicating the recording is complete
print("Recording complete. You can stop talking now.")
