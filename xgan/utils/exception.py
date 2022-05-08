import inspect

class GanException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value

class XAIConfigException(GanException):
    def __init__(self, value):
        super().__init__('XAI configuration error: ' + str(value))
