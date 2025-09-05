import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGridLayout, QPushButton, QLabel,
                             QTextEdit, QFileDialog, QMessageBox, QSplitter,
                             QGroupBox, QFrame, QScrollArea, QProgressBar,
                             QSlider, QSpinBox, QCheckBox, QInputDialog, QComboBox, QLineEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QEvent
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QPen, QIcon, QColor
from datetime import datetime
import threading

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 定义项目根路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

from app.core.detector import TrademarkDetector
from app.common.logger import LogManager
from app.core.voice_alarm import VoiceAlarm


# 自定义事件类
class UpdateResultEvent(QEvent):
    """更新检测结果事件"""
    def __init__(self, image, results):
        super().__init__(QEvent.User + 1)
        self.image = image  # 检测到的图像
        self.results = results  # 检测结果列表，包含每个检测到的商标信息


class ErrorEvent(QEvent):
    """错误事件"""
    def __init__(self, message):
        super().__init__(QEvent.User + 2)
        self.message = message  # 错误消息


class CameraThread(QThread):
    """摄像头线程"""
    frame_ready = pyqtSignal(np.ndarray)  # 摄像头信号，传递numpy数组格式的图像帧

    def __init__(self):
        super().__init__()
        self.cap = None  # 摄像头对象
        self.running = False  # 线程运行标志

    def start_camera(self):
        """启动摄像头"""
        self.cap = cv2.VideoCapture(0)  # 默认摄像头索引为0
        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if self.cap.isOpened():  # 检查摄像头是否成功打开
            self.running = True  # 设置运行标志
            self.start()  # 启动线程
            return True
        return False

    def stop_camera(self):
        """停止摄像头"""
        self.running = False  # 停止线程
        if self.cap:
            self.cap.release()  # 释放摄像头资源
        self.quit()  # 退出线程
        self.wait()  # 等待线程结束

    def run(self):
        """线程运行函数"""
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()  # 读取摄像头帧
            if ret:
                self.frame_ready.emit(frame)  # 发射信号传递帧数据
            self.msleep(30)  # 约30fps


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(600, 400)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #ddd; 
                border-radius: 8px;
                background-color: #f8f9fa;
            }
        """)
        self.setAlignment(Qt.AlignCenter)
        self.setText("请加载图片或打开摄像头")

        self.current_pixmap = None  # 当前显示的QPixmap
        self.original_image = None  # 原始图像（OpenCV格式）

    def set_image(self, cv_image):
        """设置显示的图像"""
        if cv_image is None:
            return

        self.original_image = cv_image.copy()  # 保存原图引用

        # 转换颜色空间
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        # 创建QImage
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 缩放图像以适应标签大小
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.current_pixmap = scaled_pixmap
        self.setPixmap(scaled_pixmap)


class TrademarkDetectionWindow(QMainWindow):
    """商标检测主窗口"""

    def __init__(self):
        super().__init__()

        # 强制设置应用程序样式为统一风格（跨平台一致性）
        app = QApplication.instance()  # 获取当前应用实例
        if app:
            app.setStyle('Fusion')  # 使用Fusion样式，在所有平台保持一致

        self.setWindowTitle("商标检测系统")  # 设置窗口标题
        self.setGeometry(100, 100, 1200, 750)  # 设置窗口大小和位置

        # 设置窗口图标和样式
        self.setStyleSheet(self.get_stylesheet())  # 设置统一样式表

        # 初始化变量
        self.current_image = None  # 当前显示的图像
        self.current_image_path = None  # 当前图像路径

        # 光电感应检测状态
        self.detecting_enabled = False  # 检测是否启用
        self.auto_clear_timer = QTimer()  # 自动清除输入框的定时器
        self.auto_clear_timer.setSingleShot(True)  # 单次触发
        self.auto_clear_timer.timeout.connect(self.clear_sensor_input)

        # 统计变量
        self.detection_stats = {
            'total_detections': 0, # 总检测次数
            'ok_count': 0, # 检测通过次数
            'ng_count': 0, # 检测失败次数
            'defect_types': {},  # 缺陷类型统计 {'NG标签': count, '魔术贴缺失': count}
            'session_start': datetime.now()
        }
        # 批次统计
        self.batch_current = 0  # 本批次数量
        self.batch_count = 0    # 批次数

        # 初始化组件
        self.logger = LogManager()  # 日志
        self.detector = TrademarkDetector(self.logger) # 检测器
        self.alarm = VoiceAlarm() # 语音报警
        self.camera_thread = CameraThread() # 摄像头线程

        # 创建界面
        self.setup_ui()

        # 连接信号
        self.setup_connections()

        # 添加状态栏
        self.statusBar().showMessage("系统就绪")

        self.add_log("系统初始化完成")

    def get_stylesheet(self):
        """从CSS文件加载样式表"""
        css_path = os.path.join(os.path.dirname(__file__), 'styles.css')
        try:
            with open(css_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            self.add_log(f"警告: 找不到CSS文件 {css_path}")
            return ""
        except Exception as e:
            self.add_log(f"加载CSS文件时出错: {str(e)}")
            return ""

    def setup_ui(self):
        """设置用户界面"""
        central_widget = QWidget()  # 创建中央控件
        self.setCentralWidget(central_widget)  # 设置中央控件

        # 主布局
        main_layout = QHBoxLayout(central_widget)  # 使用水平布局
        main_layout.setSpacing(15)  # 设置控件间距
        main_layout.setContentsMargins(15, 15, 15, 15)  # 设置边距

        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)  # 水平分割器
        main_layout.addWidget(splitter)  # 添加到主布局

        # 左侧控制面板
        self.setup_control_panel(splitter)

        # 右侧图像显示区域
        self.setup_image_area(splitter)

        # 设置分割器比例
        splitter.setSizes([350, 750])

    def setup_control_panel(self, parent):
        """设置控制面板"""
        control_widget = QWidget()  # 创建控制面板控件
        control_widget.setFixedWidth(350)  # 固定宽度
        control_layout = QVBoxLayout(control_widget)  # 垂直布局

        # 标题
        title_label = QLabel("商标检测控制台")  # 创建标题标签
        title_label.setObjectName("titleLabel")  # 设置样式名
        title_label.setAlignment(Qt.AlignCenter)  # 居中对齐
        control_layout.addWidget(title_label)  # 添加到布局

        # 产品选择组
        product_group = QGroupBox("产品配置")
        product_layout = QVBoxLayout(product_group)

        # 产品选择下拉框
        product_layout.addWidget(QLabel("选择产品类型:"))
        self.product_combo = QComboBox()
        self.product_combo.addItem("产品一 (商标1+魔术贴)", "product1")
        self.product_combo.addItem("产品二 (商标12+魔术贴)", "product2")
        self.product_combo.setCurrentIndex(0)  # 默认选择产品一
        product_layout.addWidget(self.product_combo)

        control_layout.addWidget(product_group)

        # 控制按钮组
        button_group = QGroupBox("操作控制")
        button_layout = QVBoxLayout(button_group)

        # 加载图片按钮
        self.load_btn = QPushButton("📁 加载图片")
        button_layout.addWidget(self.load_btn)

        # 摄像头按钮
        self.camera_btn = QPushButton("📷 打开摄像头")
        self.camera_btn.setObjectName("cameraBtn")
        button_layout.addWidget(self.camera_btn)

        # 开始检测按钮
        self.detect_btn = QPushButton("🔍 开始检测")
        self.detect_btn.setObjectName("detectBtn")
        button_layout.addWidget(self.detect_btn)

        # 光电感应输入框组
        sensor_layout = QVBoxLayout()
        sensor_layout.addWidget(QLabel("光电感应输入:"))

        self.sensor_input = QLineEdit()
        self.sensor_input.setPlaceholderText("等待光电感应输入...")
        self.sensor_input.setMaxLength(2)  # 限制最大输入长度为2
        self.sensor_input.setEnabled(False)  # 默认禁用
        self.sensor_input.setStyleSheet("""
            QLineEdit {
                font-size: 14px;
                padding: 8px;
                border: 2px solid #ddd;
                border-radius: 4px;
                background-color: #f5f5f5;
            }
            QLineEdit:enabled {
                background-color: white;
                border-color: #007ACC;
            }
            QLineEdit:focus {
                border-color: #0078d4;
                background-color: #fff;
            }
        """)
        sensor_layout.addWidget(self.sensor_input)

        button_layout.addLayout(sensor_layout)

        control_layout.addWidget(button_group)

        # 系统信息组
        info_group = QGroupBox("系统信息")
        info_layout = QVBoxLayout(info_group)


        self.status_info = QLabel("状态: 就绪")
        self.status_info.setObjectName("infoLabel")
        info_layout.addWidget(self.status_info)

        self.result_info = QLabel("检测结果: 无")
        self.result_info.setObjectName("infoLabel")
        info_layout.addWidget(self.result_info)

        control_layout.addWidget(info_group)

        # 日志显示组
        log_group = QGroupBox("操作日志")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        control_layout.addWidget(log_group)

        # 统计信息组
        stats_group = QGroupBox("检测统计")
        stats_layout = QVBoxLayout(stats_group)

        # 会话统计
        session_layout = QGridLayout()
        session_layout.addWidget(QLabel("本次会话:"), 0, 0)
        self.session_time_label = QLabel("00:00:00")
        self.session_time_label.setObjectName("infoLabel")
        session_layout.addWidget(self.session_time_label, 0, 1)

        session_layout.addWidget(QLabel("总检测次数:"), 1, 0)
        self.total_count_label = QLabel("0")
        self.total_count_label.setObjectName("infoLabel")
        session_layout.addWidget(self.total_count_label, 1, 1)

        session_layout.addWidget(QLabel("正常次数:"), 2, 0)
        self.ok_count_label = QLabel("0")
        self.ok_count_label.setObjectName("infoLabel")
        self.ok_count_label.setStyleSheet("QLabel { color: #4CAF50; font-weight: bold; }")
        session_layout.addWidget(self.ok_count_label, 2, 1)

        session_layout.addWidget(QLabel("异常次数:"), 3, 0)
        self.ng_count_label = QLabel("0")
        self.ng_count_label.setObjectName("infoLabel")
        self.ng_count_label.setStyleSheet("QLabel { color: #F44336; font-weight: bold; }")
        session_layout.addWidget(self.ng_count_label, 3, 1)

        # 批次统计
        session_layout.addWidget(QLabel("本批次数量:"), 4, 0)
        self.batch_current_label = QLabel("0 / 20")
        self.batch_current_label.setObjectName("infoLabel")
        session_layout.addWidget(self.batch_current_label, 4, 1)

        session_layout.addWidget(QLabel("批次数:"), 5, 0)
        self.batch_count_label = QLabel("0")
        self.batch_count_label.setObjectName("infoLabel")
        session_layout.addWidget(self.batch_count_label, 5, 1)

        stats_layout.addLayout(session_layout)

        control_layout.addWidget(stats_group)

        # 启动统计更新定时器
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_session_time)
        self.stats_timer.start(1000)  # 每秒更新一次

        parent.addWidget(control_widget)

    def setup_image_area(self, parent):
        """设置图像显示区域"""
        image_widget = QWidget()  # 创建图像显示控件
        image_layout = QVBoxLayout(image_widget)  # 垂直布局

        # 图像显示标签（主显示区：用于图片与检测结果）
        self.image_label = ImageLabel()
        self.image_label.setMinimumSize(600, 520)
        image_layout.addWidget(self.image_label)

        # 摄像头预览区域（小窗口，独立显示摄像头实时画面）
        preview_group = QGroupBox("摄像头预览")
        preview_layout = QVBoxLayout(preview_group)
        self.camera_preview = QLabel()
        self.camera_preview.setFixedSize(360, 240)
        self.camera_preview.setStyleSheet("QLabel { background: #111; border: 1px solid #333; }")
        self.camera_preview.setAlignment(Qt.AlignCenter)
        self.camera_preview.setText("摄像头未开启")
        preview_layout.addWidget(self.camera_preview)
        image_layout.addWidget(preview_group)

        parent.addWidget(image_widget)

    def setup_connections(self):
        """设置信号连接"""
        # 按钮连接
        self.load_btn.clicked.connect(self.load_image)  # 加载图片

        # 光电感应输入框连接
        self.sensor_input.textChanged.connect(self.on_sensor_input_changed)
        self.camera_btn.clicked.connect(self.toggle_camera)  # 切换摄像头状态
        self.detect_btn.clicked.connect(self.start_detection) # 检测按钮
        self.camera_thread.frame_ready.connect(self.update_camera_frame)  # 摄像头帧更新

        # 产品选择下拉框连接
        self.product_combo.currentTextChanged.connect(self.on_product_changed)

    def add_log(self, message):
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")  # 获取当前时间戳
        log_message = f"[{timestamp}] {message}"  # 格式化日志消息

        self.log_text.append(log_message)  # 添加到日志文本框
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )  # 滚动到最新日志

        # 记录到日志文件
        self.logger.info(message)

    def load_image(self):
        """加载图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有文件 (*)"
        )

        if file_path:  # 如果选择了文件
            try:
                self.current_image = cv2.imread(file_path) # 使用 opencv 加载选择的图片
                self.current_image_path = file_path # 当前图片的路径
                self.image_label.set_image(self.current_image) # 页面显示

                filename = os.path.basename(file_path)  # 获取图片文件名
                self.add_log(f"加载图片: {filename}") # 日志区域显示内容
                self.status_info.setText("状态: 图片已加载")
                self.statusBar().showMessage(f"已加载: {filename}")

                # 停止摄像头
                if self.camera_thread.running: # 显示图片的时候不打开摄像头
                    self.stop_camera()  # 关闭摄像头

            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法加载图片: {str(e)}")
                self.add_log(f"加载图片失败: {str(e)}")

    def toggle_camera(self):
        """切换摄像头状态"""
        if not self.camera_thread.running: # 如果未打开摄像头
            self.start_camera() # 打开摄像头
        else:
            self.stop_camera() # 关闭摄像头

    def start_camera(self):
        """启动摄像头"""
        if self.camera_thread.start_camera():
            self.camera_btn.setText("📷 关闭摄像头") # 切换按钮显示内容
            self.add_log("摄像头已启动")
            self.status_info.setText("状态: 摄像头运行中")
            self.statusBar().showMessage("摄像头运行中")
            self.camera_preview.setText("")
        else:
            QMessageBox.critical(self, "错误", "无法打开摄像头")
            print(cv2.getBuildInformation())


    def stop_camera(self):
        """停止摄像头"""

        self.camera_thread.stop_camera()  # 停止摄像头线程
        self.camera_btn.setText("📷 打开摄像头")  # 切换按钮显示内容
        self.add_log("摄像头已关闭")  # 日志区域显示内容
        self.status_info.setText("状态: 就绪")  # 更新状态信息
        self.statusBar().showMessage("就绪")  # 更新状态栏信息
        self.camera_preview.clear()
        self.camera_preview.setText("摄像头未开启")

    def update_camera_frame(self, frame):
        """更新摄像头帧"""
        self.current_image = frame
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            scaled = pix.scaled(self.camera_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_preview.setPixmap(scaled)
        except Exception:
            pass

    def start_detection(self):
        """开始检测（图片：直接检测；摄像头：启用光电感应输入监听）"""
        # 摄像头模式下：启用/禁用光电感应输入监听
        if self.camera_thread.running:  # 如果摄像头线程开启
            if not self.detecting_enabled:  # 开始监听
                self.detecting_enabled = True
                self.sensor_input.setEnabled(True)  # 启用输入框
                self.sensor_input.setFocus()  # 设置焦点到输入框
                self.detect_btn.setText("🛑 停止检测")
                self.add_log("光电感应检测已启用")
                self.status_info.setText("状态: 光电感应检测中")
                self.statusBar().showMessage("光电感应检测中...")
            else:
                # 停止监听
                self.detecting_enabled = False
                self.sensor_input.setEnabled(False)  # 禁用输入框
                self.sensor_input.clear()  # 清空输入框
                self.detect_btn.setText("🔍 开始检测")
                self.add_log("光电感应检测已停止")
                self.status_info.setText("状态: 摄像头运行中")
                self.statusBar().showMessage("摄像头运行中")
            return

        # 静态图片模式：按原逻辑
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return

        self.add_log("开始检测...")
        self.status_info.setText("状态: 检测中")
        self.statusBar().showMessage("正在检测...")
        self.detect_btn.setEnabled(False)

        detection_thread = threading.Thread(target=self.perform_detection)  # 独立线程执行检测
        detection_thread.setDaemon(True)  # 守护线程
        detection_thread.start()  # 启动线程

    def on_foot_signal(self):
        """收到脚踏 01 信号：抓取当前帧进行一次检测，并将检测结果显示在主图像区域"""
        # 使用独立线程，避免阻塞
        def _detect_once():
            try:
                img = None
                # 拷贝当前帧
                if self.current_image is not None:
                    img = self.current_image.copy()
                if img is None:
                    return
                results = self.detector.detect_in_image(img)
                result_image = self.detector.draw_results(img, results)
                QApplication.postEvent(self, UpdateResultEvent(result_image, results))
            except Exception as e:
                QApplication.postEvent(self, ErrorEvent(f"脚踏触发检测失败: {e}"))
        t = threading.Thread(target=_detect_once, daemon=True)  # 守护线程 检测一次
        t.start()

    def perform_detection(self):
        """执行检测"""
        try:
            # 执行检测
            results = self.detector.detect_in_image(self.current_image)

            # 在图像上绘制结果
            result_image = self.detector.draw_results(self.current_image, results)

            # 更新界面（在主线程中）
            QApplication.postEvent(self, UpdateResultEvent(result_image, results))

        except Exception as e:
            self.logger.error(f"检测过程中发生错误: {str(e)}")
            QApplication.postEvent(self, ErrorEvent(f"检测失败: {str(e)}"))
        finally:
            # 确保检测状态正确重置
            if hasattr(self, '_detecting'):
                self._detecting = False

    def customEvent(self, event):
        """处理自定义事件"""
        if isinstance(event, UpdateResultEvent):  # 更新检测结果事件
            self.image_label.set_image(event.image)  # 更新图像显示
            self.update_detection_results(event.results)  # 更新检测结果
        elif isinstance(event, ErrorEvent):
            self.add_log(event.message)  # 日志区域显示内容
            self.status_info.setText("状态: 检测失败")
            self.statusBar().showMessage("检测失败")
            self.detect_btn.setEnabled(True)

    def update_detection_results(self, results):
        """更新检测结果"""
        # 更新检测统计
        self.detection_stats['total_detections'] += 1  # 总检测次数加1
        self.total_count_label.setText(str(self.detection_stats['total_detections']))  # 更新总检测次数

        # 分析检测结果
        if results and len(results) > 0:
            analysis_all = []
            analyze = {'overall_status': '', 'defect_reasons': [], 'overall_status_temp': []}  # 整体分析结果
            # 获取整体分析结果
            for result in results:
                analysis = result.get('analysis', {})
                analysis_all.append(analysis)

            for analysis in analysis_all:
                analyze['overall_status_temp'].append(analysis.get('overall_status', None))  # 整体状态
                analyze['defect_reasons'].extend(analysis.get('defect_reasons', []))  # 缺陷

            status = analyze.get('overall_status_temp', None)
            defects = analyze.get('defect_reasons', [])
            if 'ng' in status:  # 如果有任何结果为NG
                overall_status = 'ng'  # 整体状态为NG
            else:
                overall_status = 'ok'
            analyze['overall_status'] = overall_status

            defects_only = []  # 只保留缺陷原因
            for defect in defects:
                if defect not in defects_only:
                    defects_only.append(defect)
            analyze['defect_reasons'] = defects_only  # 更新缺陷原因列表

            overall_status = analyze.get('overall_status', 'unknown')  # 整体状态
            defect_reasons = analyze.get('defect_reasons', [])  # 缺陷原因列表

            # 更新统计计数
            if overall_status == 'ok':  # 如果整体状态为OK
                self.detection_stats['ok_count'] += 1
                self.batch_current += 1
                if self.batch_current == 20:
                    try:
                        self.alarm.trigger_alarm(["本批次已满"])
                    except Exception:
                        pass
                    self.batch_current = 0
                    self.batch_count += 1  # 批次数+1
                # 更新批次显示
                self.batch_current_label.setText(f"{self.batch_current} / 20")
                self.batch_count_label.setText(str(self.batch_count))

                self.result_info.setText("检测结果: ✅ 产品正常")
                self.add_log("✅ 检测完成 - 产品状态：正常")
            else:
                self.detection_stats['ng_count'] += 1
                self.result_info.setText("检测结果: ❌ 产品异常")
                self.add_log("❌ 检测完成 - 产品状态：异常")

                # 触发报警
                self.alarm.trigger_alarm(defect_reasons)

                # 记录缺陷详情
                for reason in defect_reasons:
                    self.add_log(f"   ⚠️ 缺陷原因: {reason}")
        else: # 如果没有检测到任何结果 即 results 为空
            self.detection_stats['ng_count'] += 1
            reasons = ["商标缺失", "魔术贴缺失"]
            self.result_info.setText("检测结果: ❌ 产品异常")
            self.add_log("❌ 检测完成 - 未检测到任何标签/魔术贴")
            self.alarm.trigger_alarm(reasons)

        # 更新统计显示
        self.ok_count_label.setText(str(self.detection_stats['ok_count']))
        self.ng_count_label.setText(str(self.detection_stats['ng_count']))
        # self.update_defect_stats_display()

        self.status_info.setText("状态: 检测完成")
        self.statusBar().showMessage("检测完成")
        self.detect_btn.setEnabled(True)

    def update_defect_stats_display(self):
        """更新缺陷统计显示"""
        defect_text = ""
        if self.detection_stats['defect_types']:
            for defect_type, count in self.detection_stats['defect_types'].items():
                defect_text += f"{defect_type}: {count}次\n"
        else:
            defect_text = "暂无缺陷记录"

        self.defect_stats_text.setPlainText(defect_text)

    def update_session_time(self):
        """更新会话时间"""
        elapsed = datetime.now() - self.detection_stats['session_start']
        self.session_time_label.setText(str(elapsed).split('.')[0])  # 格式化为 HH:MM:SS

    def on_product_changed(self):
        """产品选择变化时的处理"""
        current_product = self.product_combo.currentData()

        if current_product == "product1":
            # 产品一：只需要ok1商标和魔术贴
            self.detector.set_product_type("product1")
            self.add_log("切换到产品一检测模式：需要ok1商标和魔术贴")
        elif current_product == "product2":
            # 产品二：需要ok1和ok2商标以及魔术贴
            self.detector.set_product_type("product2")
            self.add_log("切换到产品二检测模式：需要ok1、ok2商标和魔术贴")

    def on_sensor_input_changed(self, text):
        """光电感应输入框文本变化处理"""
        if not self.detecting_enabled:
            return

        # 检查是否输入了"01"
        if text == "01":
            self.add_log("收到光电感应信号: 01")
            # 触发检测
            self.trigger_sensor_detection()
            # 启动0.5秒后自动清除定时器
            self.auto_clear_timer.start(500)  # 500毫秒 = 0.5秒

    def trigger_sensor_detection(self):
        """光电感应触发的检测"""
        # 使用独立线程，避免阻塞
        def _detect_once():
            try:
                img = None
                # 拷贝当前帧
                if self.current_image is not None:
                    img = self.current_image.copy()
                if img is None:
                    return
                results = self.detector.detect_in_image(img)
                result_image = self.detector.draw_results(img, results)
                QApplication.postEvent(self, UpdateResultEvent(result_image, results))
            except Exception as e:
                QApplication.postEvent(self, ErrorEvent(f"光电感应触发检测失败: {e}"))

        t = threading.Thread(target=_detect_once, daemon=True)  # 守护线程 检测一次
        t.start()

    def clear_sensor_input(self):
        """清除光电感应输入框"""
        if self.sensor_input.isEnabled():
            self.sensor_input.clear()
            self.sensor_input.setFocus()  # 重新设置焦点到输入框
