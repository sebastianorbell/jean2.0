
class Dummy_parameter:
    def __init__(self, name, init_value=None, measurement=lambda x: x):
        self.name = name
        self.value = init_value
        self.measurement_fn = measurement

    def set(self, val):
        self.value = val

    def get(self):
        return self.value

    def measurement(self):
        return self.measurement_fn(self.value)