
class Experiment_parameter:
    def __init__(self, bounds: dict):
        """
        Dummy parameter class.
        @param name:
        @param bounds: dict {'name': (min, max)}
        @param measurement:
        """
        self.names = list(bounds.keys())
        self.bounds = list(bounds.values())
        self.value = None

    def set(self, val: list):
        self.value = dict(zip(self.names, val))

    def get(self):
        return self.value
