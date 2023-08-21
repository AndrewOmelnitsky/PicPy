with open("py_UI/filter_creator_ui.py", "r") as file:
    data = file.read()

data = "from widgets.validate_edit import VelidateLineEdit\n" + data

divider = "self.divider = "
data = data.replace(f"{divider}QtWidgets.QLineEdit", f"{divider}VelidateLineEdit")
color_shift = "self.color_shift = "
data = data.replace(
    f"{color_shift}QtWidgets.QLineEdit", f"{color_shift}VelidateLineEdit"
)

with open("py_UI/filter_creator_ui.py", "w") as file:
    file.write(data)
