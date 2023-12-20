class ApplicationManger():
    def __init__(self,ui):
        self.ui = ui
        self.fingerprint_mode = False

    def switch_modes(self):
        self.fingerprint_mode = not self.fingerprint_mode 
    
    def record_voice(self):
        pass


