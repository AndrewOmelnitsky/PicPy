from widgets.validate_edit import VelidateLineEdit

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/blackgolyb/Documents/PyMage/ui/filter_creator.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(662, 528)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.filterGB = QtWidgets.QGroupBox(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filterGB.sizePolicy().hasHeightForWidth())
        self.filterGB.setSizePolicy(sizePolicy)
        self.filterGB.setObjectName("filterGB")
        self.filterLayout = QtWidgets.QGridLayout(self.filterGB)
        self.filterLayout.setObjectName("filterLayout")
        self.horizontalLayout_2.addWidget(self.filterGB)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        spacerItem = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.verticalLayout_4.addItem(spacerItem)
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.filter_h = QtWidgets.QSpinBox(self.groupBox_4)
        self.filter_h.setMinimum(1)
        self.filter_h.setMaximum(11)
        self.filter_h.setSingleStep(2)
        self.filter_h.setObjectName("filter_h")
        self.horizontalLayout_3.addWidget(self.filter_h)
        self.filter_w = QtWidgets.QSpinBox(self.groupBox_4)
        self.filter_w.setMinimum(1)
        self.filter_w.setMaximum(11)
        self.filter_w.setSingleStep(2)
        self.filter_w.setObjectName("filter_w")
        self.horizontalLayout_3.addWidget(self.filter_w)
        self.verticalLayout_5.addLayout(self.horizontalLayout_3)
        self.clear_filter = QtWidgets.QPushButton(self.groupBox_4)
        self.clear_filter.setObjectName("clear_filter")
        self.verticalLayout_5.addWidget(self.clear_filter)
        self.load_filter = QtWidgets.QPushButton(self.groupBox_4)
        self.load_filter.setObjectName("load_filter")
        self.verticalLayout_5.addWidget(self.load_filter)
        self.verticalLayout_4.addWidget(self.groupBox_4)
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_3.sizePolicy().hasHeightForWidth())
        self.groupBox_3.setSizePolicy(sizePolicy)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox_3)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.filter_name = QtWidgets.QLineEdit(self.groupBox_3)
        self.filter_name.setObjectName("filter_name")
        self.verticalLayout_2.addWidget(self.filter_name)
        self.label = QtWidgets.QLabel(self.groupBox_3)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.divider = VelidateLineEdit(self.groupBox_3)
        self.divider.setObjectName("divider")
        self.verticalLayout_2.addWidget(self.divider)
        self.label_2 = QtWidgets.QLabel(self.groupBox_3)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.color_shift = VelidateLineEdit(self.groupBox_3)
        self.color_shift.setObjectName("color_shift")
        self.verticalLayout_2.addWidget(self.color_shift)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.save_filter = QtWidgets.QPushButton(self.groupBox_3)
        self.save_filter.setObjectName("save_filter")
        self.horizontalLayout.addWidget(self.save_filter)
        self.add_filter = QtWidgets.QPushButton(self.groupBox_3)
        self.add_filter.setObjectName("add_filter")
        self.horizontalLayout.addWidget(self.add_filter)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.verticalLayout_4.addWidget(self.groupBox_3)
        self.horizontalLayout_2.addLayout(self.verticalLayout_4)
        self.verticalLayout.addWidget(self.groupBox)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox.setTitle(_translate("Form", "Filter creator"))
        self.filterGB.setTitle(_translate("Form", "Filter"))
        self.groupBox_4.setTitle(_translate("Form", "Editor settings"))
        self.clear_filter.setText(_translate("Form", "Clear"))
        self.load_filter.setText(_translate("Form", "Load filter"))
        self.groupBox_3.setTitle(_translate("Form", "Filter settings"))
        self.label_3.setText(_translate("Form", "name"))
        self.label.setText(_translate("Form", "divider"))
        self.label_2.setText(_translate("Form", "color shift"))
        self.save_filter.setText(_translate("Form", "Save filter"))
        self.add_filter.setText(_translate("Form", "Add Filter"))
