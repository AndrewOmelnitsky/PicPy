from PyQt5 import QtWidgets


class VelidateLineEdit(QtWidgets.QLineEdit):
    try_type = float
    default = 0
    validate = [None, None]

    def __init__(self, parent=None):
        QtWidgets.QLineEdit.__init__(self, parent)
        self.editingFinished.connect(self.param_validate)
        self.param_validate()

    def set_default(self, val):
        self.default = val
        self.setText(str(self.default))

    def set_try_type(self, val):
        self.try_type = val

    def set_validate(self, validate):
        self.validate = validate

    def param_validate(self):
        text = self.text()

        try:
            res = self.try_type(text)
            a, b = self.validate
            if a is None and b is None:
                return True
            elif a is None:
                if res < b:
                    return True
            elif b is None:
                if res > a:
                    return True
            else:
                if a < res < b:
                    return True
        except:
            pass

        self.setText(str(self.default))
        return False
