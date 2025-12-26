class DetectionLogger:
    def __init__(self, widget):
        self.widget = widget

    def log(self, message):
        self.widget.append(message)
