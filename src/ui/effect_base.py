# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/blackgolyb/Documents/PyMage/ui/effect_base.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(150, 117)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.effectGroup = QtWidgets.QGroupBox(Form)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.effectGroup.sizePolicy().hasHeightForWidth())
        self.effectGroup.setSizePolicy(sizePolicy)
        self.effectGroup.setMinimumSize(QtCore.QSize(150, 0))
        self.effectGroup.setObjectName("effectGroup")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.effectGroup)
        self.horizontalLayout_2.setContentsMargins(6, 6, 6, 6)
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout_2.addItem(spacerItem)
        self.effectLayout = QtWidgets.QVBoxLayout()
        self.effectLayout.setObjectName("effectLayout")
        self.channelSelector = QtWidgets.QComboBox(self.effectGroup)
        self.channelSelector.setObjectName("channelSelector")
        self.effectLayout.addWidget(self.channelSelector)
        self.useEffectBtn = QtWidgets.QPushButton(self.effectGroup)
        self.useEffectBtn.setObjectName("useEffectBtn")
        self.effectLayout.addWidget(self.useEffectBtn)
        self.horizontalLayout_2.addLayout(self.effectLayout)
        spacerItem1 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout_2.addItem(spacerItem1)
        self.verticalLayout_2.addWidget(self.effectGroup)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.effectGroup.setTitle(_translate("Form", "Effect name"))
        self.useEffectBtn.setText(_translate("Form", "Use effect"))
