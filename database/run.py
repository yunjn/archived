# coding:utf-8
import os
import sys
import pandas as pd
from pymysql import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtGui import QPixmap, QFont, QIcon
from PyQt5.QtCore import QPoint, QPropertyAnimation, QUrl

#############################################################
########################### 全局  ###########################
############################################################
db = ''
cursor = ''
account = ''
password = ''
root_path = os.getcwd()
style_sheet = """
/*去掉item虚线边框*/
QListWidget, QListView, QTreeWidget, QTreeView {
    outline: 0px;
}
/*设置左侧选项的最小最大宽度,文字颜色和背景颜色*/
QListWidget {
    min-width: 120px;
    max-width: 120px;
    color: black;
    background: white;
}
/*被选中时的背景颜色和左边框颜色*/
QListWidget::item:selected {
    
    background: rgb(171, 220, 255);
    /*background: rgb(47, 169, 255);*/
    /*background: rgb(65, 105, 225);*/
    border-radius:18px;
    border-left: 0px solid rgb(9, 187, 7);
}
/*鼠标悬停颜色*/
HistoryPanel::item:hover {
    background: rgb(171, 220, 255);
}

/*右侧的层叠窗口的背景颜色*/
QStackedWidget {
    background: rgb(255, 255, 255);
}
/*模拟的页面*/
QLabel {
    color: white;
}
"""
window_list = []
############################################################


def list_to_html(table_name='sql', sql='123'):
    global db
    global cursor
    db = connect(host='localhost', user=account,
                 password=password, database='TSM', charset='utf8')
    cursor = db.cursor()
    if sql == '123':
        sql = f'select * from {table_name}'
    try:
        tmp = cursor.execute(sql)
        db.commit()
        data = cursor.fetchall()
        col = cursor.description
        cols = []
        for i in range(len(col)):
            cols.append(col[i][0])
        pd.set_option('display.width', 1000)
        pd.set_option('colheader_justify', 'center')
        data = pd.DataFrame(data, columns=cols)
        table = data.to_html(classes='mystyle')
        html_string = '''
        <html>
        <head><title>HTML Pandas Dataframe with CSS</title></head>
        <style type="text/css">
    .mystyle {
        font-size: 14pt; 
        width:100%;
        /* height:90%; */
        text-align: center;
        vertical-align:middle;
        font-family: Arial;
        border-collapse: collapse; 
        /* border: 1px solid rgb(194, 194, 194); */
        border: 1px solid #0396FF; 
    }
    .mystyle td, th {
        padding: 5px;
    }
    .mystyle tr:nth-child(even) {
        background: #ffffff;
    }
    .mystyle tr:hover {
        background: rgb(255, 255, 255);
        cursor: pointer;
    }
    </style>
        <body> ''' + f'''{table}</body></html>
    '''
        with open(f'static/html/{table_name}.html', 'w') as f:
            f.write(html_string)
    except:
        db.rollback()
    # cursor.close()

#   <link rel="stylesheet" type="text/css" href="{root_path}\static\css\\table.css"/>


def img_to_html(img_name: str):
    html_string = f'''
  <html>
    <head><title>分析</title></head><body>
      <img src="{root_path}/static/img/{img_name}" width="100%" height="100%"/>
    </body>
  </html>
  '''
    with open(f'static/html/{img_name.split(".")[0]}.html', 'w') as f:
        f.write(html_string)
###########################################################################


class LoginForm(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setObjectName("login_window")
        self.setStyleSheet('#login_window{background-color:white}')
        self.setFixedSize(650, 400)
        self.setWindowTitle("Login")
        self.setWindowIcon(QIcon('static/img/logo.png'))
        self.text = "家校家教服务管理系统"

        # 添加顶部logo图片
        pixmap = QPixmap("static/img/header.png")
        scared_pixmap = pixmap.scaled(650, 140)
        label = QLabel(self)
        label.setPixmap(scared_pixmap)

        # 绘制顶部文字
        lbl_logo = QLabel(self)
        lbl_logo.setText(self.text)
        lbl_logo.setStyleSheet(
            "QWidget{color:white;font-weight:600;background: transparent;font-size:30px;}")
        lbl_logo.setFont(QFont("Microsoft YaHei"))
        lbl_logo.move(170, 50)
        lbl_logo.setAlignment(Qt.AlignCenter)
        lbl_logo.raise_()

        # 登录表单内容部分
        login_widget = QWidget(self)
        login_widget.move(0, 140)
        login_widget.setGeometry(0, 140, 650, 260)
        hbox = QHBoxLayout()

        # 添加左侧logo
        logolb = QLabel(self)
        logopix = QPixmap("static/img/database.svg")
        logopix_scared = logopix.scaled(150, 150)
        logolb.setPixmap(logopix_scared)
        logolb.setAlignment(Qt.AlignCenter)
        hbox.addWidget(logolb, 1)

        # 添加右侧表单
        fmlayout = QFormLayout()
        lbl_workerid = QLabel("用户名")
        lbl_workerid.setFont(QFont("Microsoft YaHei"))
        self.led_workerid = QLineEdit()
        self.led_workerid.setFixedWidth(270)
        self.led_workerid.setFixedHeight(38)
        lbl_pwd = QLabel("密码")
        lbl_pwd.setFont(QFont("Microsoft YaHei"))
        self.led_pwd = QLineEdit()
        self.led_pwd.setEchoMode(QLineEdit.Password)
        self.led_pwd.setFixedWidth(270)
        self.led_pwd.setFixedHeight(38)

        btn_login = QPushButton("登录")
        btn_login.setFixedWidth(270)
        btn_login.setFixedHeight(40)
        btn_login.setFont(QFont("Microsoft YaHei"))
        btn_login.setObjectName("login_btn")
        btn_login.setStyleSheet(
            "#login_btn{background-color:#2c7adf;color:#fff;border:none;border-radius:4px;}")
        btn_login.clicked.connect(self.btn_login_fuc)

        fmlayout.addRow(lbl_workerid, self.led_workerid)
        fmlayout.addRow(lbl_pwd, self.led_pwd)
        fmlayout.addWidget(btn_login)
        hbox.setAlignment(Qt.AlignCenter)

        # 调整间距
        fmlayout.setHorizontalSpacing(20)
        fmlayout.setVerticalSpacing(12)
        hbox.addLayout(fmlayout, 2)
        login_widget.setLayout(hbox)
        self.center()
        self.show()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def btn_login_fuc(self):
        global account
        global password
        account = self.led_workerid.text()
        password = self.led_pwd.text()
        try:
            global db
            global app
            global window_list
            db = connect(host='localhost', user=account,
                         password=password, database='TSM', charset='utf8')
            # db = connect(host='localhost', user='root',
            #              password='samuel', database='TSM', charset='utf8')
            app.setStyleSheet(style_sheet)
            main_window = MainWindow()
            window_list.append(main_window)
            self.close()
            main_window.show()
        except:
            # QMessageBox.warning(self, 'ERROR', "账号或密码错误", QMessageBox.Yes)
            self.shake_window(self)

    def shake_window(self, target):
        if hasattr(target, '_shake_animation'):
            return
        animation = QPropertyAnimation(target, b'pos', target)
        target._shake_animation = animation
        animation.finished.connect(lambda: delattr(target, '_shake_animation'))
        pos = target.pos()
        x, y = pos.x(), pos.y()
        animation.setDuration(200)
        animation.setLoopCount(2)
        animation.setKeyValueAt(0, QPoint(x, y))
        animation.setKeyValueAt(0.09, QPoint(x + 2, y - 2))
        animation.setKeyValueAt(0.18, QPoint(x + 4, y - 4))
        animation.setKeyValueAt(0.27, QPoint(x + 2, y - 6))
        animation.setKeyValueAt(0.36, QPoint(x + 0, y - 8))
        animation.setKeyValueAt(0.45, QPoint(x - 2, y - 10))
        animation.setKeyValueAt(0.54, QPoint(x - 4, y - 8))
        animation.setKeyValueAt(0.63, QPoint(x - 6, y - 6))
        animation.setKeyValueAt(0.72, QPoint(x - 8, y - 4))
        animation.setKeyValueAt(0.81, QPoint(x - 6, y - 2))
        animation.setKeyValueAt(0.90, QPoint(x - 4, y - 0))
        animation.setKeyValueAt(0.99, QPoint(x - 2, y + 2))
        animation.setEndValue(QPoint(x, y))
        animation.start(animation.DeleteWhenStopped)


class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setStyleSheet('#login_window{background-color:white}')
        self.setWindowTitle("Aicy")
        self.setWindowIcon(QIcon('static/img/logo.png'))
        self.resize(1280, 720)

        layout = QHBoxLayout(self, spacing=0)
        layout.setContentsMargins(0, 0, 0, 0)

        # 左侧列表
        self.listWidget = QListWidget(self)
        layout.addWidget(self.listWidget)

        # 右侧层叠窗口
        self.stackedWidget = QStackedWidget(self)
        layout.addWidget(self.stackedWidget)
        self.init_ui()

    def sql_in(self, event):
        sql_text, ok = QInputDialog().getText(QWidget(), 'SQL', '输入文本:')
        sql_text.encode('utf-8')
        if ok and sql_text:
            list_to_html(sql=sql_text)
        sql_file = open('static/html/sql.html', 'r')
        data = sql_file.read()
        sql_file.close()

        if ok and data:
            self.sql_label.setText(data)
        self.data_to_img()
        self.update_data()

    def init_ui(self):
        self.listWidget.currentRowChanged.connect(
            self.stackedWidget.setCurrentIndex)
        # 去掉边框
        self.listWidget.setFrameShape(QListWidget.NoFrame)
        # 隐藏滚动条
        self.listWidget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.listWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # 数据库
        global db
        global cursor
        cursor = db.cursor()
        self.data_to_img()
        rows = cursor.execute('show tables from tsm')
        table_list = cursor.fetchall()
        # 分析页面
        img_to_html('an.jpg')
        an_item = QListWidgetItem(
            QIcon('static/img/an.png'), str('分析'), self.listWidget)
        self.listWidget.setIconSize(QSize(60, 60))
        an_item.setSizeHint(QSize(16777215, 75))
        an_item.setTextAlignment(Qt.AlignCenter)

        self.browser = QWebEngineView()
        self.browser.load(QUrl.fromLocalFile(
            f"{root_path}/static/html/an.html"))
        self.stackedWidget.addWidget(self.browser)

        for i in range(rows):
            item = QListWidgetItem(
                QIcon('static/img/table.svg'), str(table_list[i][0]), self.listWidget)
            item.setSizeHint(QSize(16777215, 75))
            item.setTextAlignment(Qt.AlignCenter)
            list_to_html(table_name=str(table_list[i][0]))

        for i in range(rows):
            browser = QWebEngineView()
            browser.load(QUrl.fromLocalFile(
                f'{root_path}/static/html/{table_list[i][0]}.html'))
            self.stackedWidget.addWidget(browser)

        # SQL页面
        sql_item = QListWidgetItem(
            QIcon('static/img/database.svg'), str('SQL'), self.listWidget)
        self.listWidget.setIconSize(QSize(60, 60))
        sql_item.setSizeHint(QSize(16777215, 75))
        sql_item.setTextAlignment(Qt.AlignCenter)

        self.sql_label = QLabel()
        self.sql_label.setAlignment(Qt.AlignCenter)
        self.sql_label.setStyleSheet('color:rgb(0,0,0);')
        self.sql_label.mouseDoubleClickEvent = self.sql_in
        sql_file = open('static/html/sql.html', 'r')
        data = sql_file.read()
        sql_file.close()

        self.sql_label.setText(data)

        self.stackedWidget.addWidget(self.sql_label)

        # self.browser = QWebEngineView()
        # self.browser.load(QUrl.fromLocalFile(
        #     f"{root_path}/static/html/sql.html"))
        # self.stackedWidget.addWidget(self.browser)

        # end
        cursor.close()
        db.close()
        # self.add_blog()
    def update_data(self):
        global db
        global cursor

        db = connect(host='localhost', user=account,
                     password=password, database='TSM', charset='utf8')
        cursor = db.cursor()
        rows = cursor.execute('show tables from tsm')
        table_list = cursor.fetchall()
        for i in range(rows):
            list_to_html(table_name=str(table_list[i][0]))
        pass

    def add_blog(self):
        blog = QListWidgetItem(
            QIcon('static/img/blog.png'), str('Blog'), self.listWidget)
        self.listWidget.setIconSize(QSize(60, 60))
        blog.setSizeHint(QSize(16777215, 75))
        blog.setTextAlignment(Qt.AlignCenter)
        self.browser = QWebEngineView()
        self.browser.load(QUrl('https://blog.xraw.top'))
        self.stackedWidget.addWidget(self.browser)

    def data_to_img(self):
        import matplotlib.pyplot as plt
        global db
        global cursor

        db = connect(host='localhost', user=account,
                     password=password, database='TSM', charset='utf8')
        cursor = db.cursor()

        data_num = cursor.execute('select * from 教师信息表;')
        data = cursor.fetchall()
        class_name = []
        for row in data:
            class_name.append(row[4])

        data_num = cursor.execute('select * from 职业登记表;')
        data = cursor.fetchall()
        profession = []
        for row in data:
            profession.append(row[1])
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(nrows=1, ncols=2)
        ax0, ax1 = axes.flatten()
        ax0.hist(class_name, bins=10, density=False,)
        ax0.set_title('课程需求')

        ax1.hist(profession, bins=10, density=False)
        ax1.set_title('职业需求')
        fig.tight_layout()
        plt.savefig('static/img/an.jpg')
        # plt.show()
        pass


input_stylesheet = """
#Custom_Widget {
    background: white;
    border-radius: 10px;
}

#closeButton {
    min-width: 30px;
    min-height: 30px;
    font-family: "Webdings";
    qproperty-text: "r";
    border-radius: 10px;
}
#closeButton:hover {
    color: white;
    background: red;
}
"""

app = QApplication(sys.argv)
login = LoginForm()
sys.exit(app.exec_())
