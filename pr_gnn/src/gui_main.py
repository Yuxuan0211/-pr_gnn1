# src/gui_main.py
import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import pandas as pd
from pr_gnn.src.dataset import FlowDataset
from pr_gnn.src.pr_gnn_model import PRGNN
from pr_gnn.src.trainer import PRGNNTrainer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
class PRGNN_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PR-GNN 流场预测系统")
        self.setGeometry(100, 100, 800, 600)
        self.model = None
        self.data = None
        self.config = None
        self.init_ui()
    def update_loss_plot(self, epoch, total_loss, physics_loss):
        """更新训练损失曲线图"""
        self.loss_history['total'].append(total_loss)
        self.loss_history['physics'].append(physics_loss)
        
        # 创建matplotlib图表
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.loss_history['total'], label='总损失')
        ax.plot(self.loss_history['physics'], label='物理损失')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        # 将图表转换为QPixmap并显示
        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        qimage = canvas.buffer_rgba()
        pixmap = QPixmap.fromImage(qimage)
        self.loss_plot.setPixmap(pixmap)
        
        plt.close(fig)
    def init_ui(self):
        font = QFont("SimHei", 10)
        self.setFont(font)

        container = QWidget()
        layout = QVBoxLayout()

        title = QLabel("PR-GNN: 物理区域化图神经网络")
        title.setFont(QFont("SimHei", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.btn_adj = QPushButton("选择 图邻接 CSV")
        self.btn_adj.clicked.connect(self.load_adj_csv)
        layout.addWidget(self.btn_adj)

        self.btn_csv = QPushButton("选择节点数据 CSV(可多选)")
        self.btn_csv.clicked.connect(self.load_csv)
        layout.addWidget(self.btn_csv)

        # 参数配置面板
        param_group = QGroupBox("训练参数配置")
        param_layout = QFormLayout()
        
        # 基础训练参数
        self.lr_input = QLineEdit("0.001")
        self.batch_size_input = QLineEdit("800")
        self.pre_epochs_input = QLineEdit("100")
        self.fine_epochs_input = QLineEdit("500")
        
        # 神经网络结构参数
        self.hidden_dim_input = QLineEdit("32")
        self.num_layers_input = QLineEdit("2")
        self.dropout_input = QLineEdit("0.1")
        self.batch_norm_check = QCheckBox("启用批归一化")
        self.batch_norm_check.setChecked(True)
        
        # 损失权重
        self.w_thermo = QLineEdit("1.0")
        self.w_vorticity = QLineEdit("10.0")
        self.w_noslip = QLineEdit("5.0")
        self.w_wake = QLineEdit("1.0")
        self.w_energy = QLineEdit("1.0")
        self.w_inviscid = QLineEdit("0.5")
        self.w_freestream = QLineEdit("10.0")
        self.lambda_phys = QLineEdit("1.0")
        
        # 添加控件
        # 添加神经网络结构配置
        param_layout.addRow(QLabel("神经网络结构"))
        param_layout.addRow("隐藏层维度", self.hidden_dim_input)
        param_layout.addRow("网络层数", self.num_layers_input)
        param_layout.addRow("Dropout率", self.dropout_input)
        param_layout.addRow(self.batch_norm_check)
        
        # 添加训练参数
        param_layout.addRow(QLabel("训练参数"))
        param_layout.addRow("学习率", self.lr_input)
        param_layout.addRow("Batch Size", self.batch_size_input)
        param_layout.addRow("预训练轮数", self.pre_epochs_input)
        param_layout.addRow("微调轮数", self.fine_epochs_input)
        
        # 添加损失权重
        param_layout.addRow(QLabel("损失权重"))
        param_layout.addRow("热力学损失", self.w_thermo)
        param_layout.addRow("涡量损失", self.w_vorticity)
        param_layout.addRow("无滑移损失", self.w_noslip)
        param_layout.addRow("尾流损失", self.w_wake)
        param_layout.addRow("能量损失", self.w_energy)
        param_layout.addRow("无粘区损失", self.w_inviscid)
        param_layout.addRow("来流区损失", self.w_freestream)
        param_layout.addRow("物理损失权重", self.lambda_phys)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # 训练模式选择
        hbox = QHBoxLayout()
        self.radio_pre = QRadioButton("预训练")
        self.radio_fine = QRadioButton("微调")
        self.radio_fine.setChecked(True)
        hbox.addWidget(self.radio_pre)
        hbox.addWidget(self.radio_fine)
        layout.addLayout(hbox)

        self.btn_train = QPushButton("开始训练")
        self.btn_train.clicked.connect(self.start_training)
        layout.addWidget(self.btn_train)

        # 模型操作按钮
        btn_hbox = QHBoxLayout()
        self.btn_load = QPushButton("加载模型")
        self.btn_load.clicked.connect(self.load_model)
        self.btn_save = QPushButton("保存模型")
        self.btn_save.clicked.connect(self.save_model)
        self.btn_predict = QPushButton("预测流场")
        self.btn_predict.clicked.connect(self.predict_flow)
        btn_hbox.addWidget(self.btn_load)
        btn_hbox.addWidget(self.btn_save)
        btn_hbox.addWidget(self.btn_predict)
        layout.addLayout(btn_hbox)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_adj_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 图邻接 CSV 文件", "", "*.csv")
        if path:
            from pr_gnn.src.csv_graph_loader import save_adjacency_csv, load_adjacency_from_csv
            import traceback
            try:
                adj = load_adjacency_from_csv(path)
                save_adjacency_csv(adj, "data/processed/adjacency.csv")
                self.log.append(f"已生成邻接矩阵: adjacency.csv")
            except Exception as e:
                error_msg = f"❌ CSV 解析失败:\n{traceback.format_exc()}"
                self.log.append(error_msg)
                QMessageBox.critical(self, "错误", f"CSV解析失败:\n{str(e)}")

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "models/saved", "*.pth")
        if path:
            import traceback
            try:
                if self.model is None:
                    # 如果模型未初始化，先创建默认模型
                    config = self._get_config_from_ui()
                    self.model = PRGNN(in_channels=7, hidden_channels=64).to(config['device'])
                    self.trainer = PRGNNTrainer(self.model, config)
                
                self.trainer.load_model(path)
                self.log.append(f"模型已加载: {path}")
            except Exception as e:
                error_msg = f"❌ 模型加载失败:\n{traceback.format_exc()}"
                self.log.append(error_msg)
                QMessageBox.critical(self, "错误", f"模型加载失败:\n{str(e)}")

    def _get_config_from_ui(self):
        return {
            'free_stream': {'V': 100, 'P': 101325, 'rho': 1.225, 'h': 300e3},
            'P_inf': 101325, 'rho_inf': 1.225, 'V_inf': 100, 'h_inf': 300e3,
            'loss_weights': {
                'thermo': float(self.w_thermo.text()),
                'vorticity': float(self.w_vorticity.text()),
                'energy': float(self.w_energy.text()),
                'noslip': float(self.w_noslip.text()),
                'wake': float(self.w_wake.text()),
                'inviscid': float(self.w_inviscid.text()),
                'freestream': float(self.w_freestream.text())
            },
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            # 模型结构参数
            'hidden_channels': int(self.hidden_dim_input.text()),
            'num_layers': int(self.num_layers_input.text()),
            'dropout': float(self.dropout_input.text()),
            'batch_norm': self.batch_norm_check.isChecked(),
            
            'training': {
                'mixed_precision': True,
                'lr': float(self.lr_input.text()),
                'weight_decay': 0.01,
                'warmup_epochs': 10,
                'cosine_epochs': 100,
                'batch_size': int(self.batch_size_input.text()),
                'neighbor_sampling': False,  # 默认关闭邻居采样
                'num_neighbors': 25,  # 邻居采样数量
                'early_stopping_patience': 20,  # 早停耐心值
                'min_epochs': 50  # 最小训练轮数
            },
            'pre_epochs': int(self.pre_epochs_input.text()),
            'lambda_phys': float(self.lambda_phys.text())
        }

    def load_csv(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择节点数据 CSV 文件", "", "*.csv")
        if paths:
            import traceback
            try:
                config = self._get_config_from_ui()
                total_nodes = 0
                loaded_files = 0
                
                for i, path in enumerate(paths):
                    dataset = FlowDataset("data/processed/adjacency.csv", path, config)
                    data, scaler_x, scaler_y = dataset.load_data()
                    
                    if i == 0:
                        self.data = data
                        self.scaler_x = scaler_x
                        self.scaler_y = scaler_y
                    else:
                        # 合并数据，保持Data对象结构
                        self.data.x = torch.cat([self.data.x, data.x], dim=0)
                        self.data.y = torch.cat([self.data.y, data.y], dim=0)
                        # 更新节点数
                        self.data.num_nodes = self.data.x.size(0)
                    
                    total_nodes += data.num_nodes
                    loaded_files += 1
                    self.log.append(f"[{i+1}/{len(paths)}] 已加载文件: {os.path.basename(path)} (节点数: {data.num_nodes})")
                
                if self.model is None:  # 如果模型未加载
                    self.model = PRGNN(in_channels=7, hidden_channels=64).to(config['device'])
                    self.trainer = PRGNNTrainer(self.model, config)
                
                self.log.append(f"数据加载完成 - 共加载 {loaded_files} 个文件，总节点数: {total_nodes}")
                QMessageBox.information(self, "加载完成", f"成功加载 {loaded_files} 个CSV文件\n总节点数: {total_nodes}")
            except Exception as e:
                error_msg = f"数据加载失败:\n{traceback.format_exc()}"
                self.log.append(error_msg)
                QMessageBox.critical(self, "错误", f"数据加载失败:\n{str(e)}")

    def convert_stl_to_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 STL 文件", "", "*.stl")
        if path:
            import traceback
            try:
                from pr_gnn.src.stl_to_graph import convert_stl_to_csv
                output_path = os.path.join("data/raw", os.path.basename(path).replace('.stl', '.csv'))
                convert_stl_to_csv(path, output_path)
                self.log.append(f"STL 转换完成: {output_path}")
            except Exception as e:
                error_msg = f"STL 转换失败:\n{traceback.format_exc()}"
                self.log.append(error_msg)
                QMessageBox.critical(self, "错误", f"STL转换失败:\n{str(e)}")

    def start_training(self):
        if self.data is None:
            QMessageBox.warning(self, "错误", "请先加载数据！")
            return
        try:
            # 初始化损失记录
            self.loss_history = {'total': [], 'physics': []}
            
            if self.radio_pre.isChecked():
                self.trainer.regional_pretrain(self.data)
                self.log.append("预训练完成")
            else:
                epochs = int(self.fine_epochs_input.text())
                self.trainer.global_finetune(self.data, epochs=epochs)
                self.log.append("微调完成")
        except Exception as e:
            import traceback
            error_msg = f"❌ 训练失败\n\n错误类型: {type(e).__name__}\n\n错误详情: {str(e)}\n\n调用栈:\n"
            error_msg += "".join(traceback.format_exception(type(e), e, e.__traceback__))
            
            error_dialog = QDialog(self)
            error_dialog.setWindowTitle("错误详情")
            error_dialog.setMinimumSize(600, 400)
            
            layout = QVBoxLayout()
            text_edit = QTextEdit()
            text_edit.setPlainText(error_msg)
            text_edit.setReadOnly(True)
            
            close_btn = QPushButton("关闭")
            close_btn.clicked.connect(error_dialog.close)
            
            layout.addWidget(text_edit)
            layout.addWidget(close_btn)
            error_dialog.setLayout(layout)
            error_dialog.exec_()
            
            self.log.append(f"❌ 训练失败: {str(e)}")

    def predict_flow(self):
        if self.model is None or self.data is None:
            QMessageBox.warning(self, "错误", "请先加载模型和数据！")
            return
        
        # 获取配置
        self.config = self._get_config_from_ui()
        
        # 获取多个马赫数输入
        mach_str, ok = QInputDialog.getText(self, "输入马赫数", 
                                         "输入多个马赫数(用逗号分隔):", 
                                         text="0.6,0.8,1.0,1.2")
        if not ok:
            return
            
        try:
            mach_numbers = [float(m.strip()) for m in mach_str.split(",")]
        except:
            QMessageBox.warning(self, "错误", "请输入有效的马赫数列表！")
            return
        
        try:
            # 选择保存目录
            save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录")
            if not save_dir:
                return
                
            self.log.append(f"开始批量预测马赫数: {mach_numbers}")
            
            for mach in mach_numbers:
                try:
                    # 复制原始数据并确保在正确设备上
                    pred_data = self.data.clone().to(self.config['device'])
                    
                    # 根据马赫数调整来流条件
                    scale_factor = mach / (self.config['V_inf'] / 340.0)
                    
                    # 调整来流参数
                    pred_data.x[:, 0] *= scale_factor  # 速度
                    pred_data.x[:, 6] *= scale_factor**2  # 压力
                    
                    # 预测
                    self.model.eval()
                    with torch.no_grad():
                        # 确保edge_index也在正确设备上
                        edge_index = pred_data.edge_index.to(self.config['device'])
                        pred, _ = self.model(pred_data.x, edge_index)
                        
                    # 保存预测结果
                    save_path = os.path.join(save_dir, f"pred_M{mach:.2f}.csv")
                    
                    # 创建包含节点位置和预测流场数据的DataFrame
                    pred_np = pred.cpu().numpy()
                    data_np = pred_data.x.cpu().numpy()
                    
                    # 获取节点编号和名称（假设第0列是编号，第3列是名称）
                    node_ids = data_np[:, 0].astype(int)
                    node_names = data_np[:, 3].astype(str)
                    
                    result_data = {
                        'NodeID': node_ids,
                        'NodeName': node_names,
                        'X': data_np[:, 1],
                        'Y': data_np[:, 2], 
                        'Z': data_np[:, 3],
                        'Vx': pred_np[:, 0],
                        'Vy': pred_np[:, 1],
                        'Vz': pred_np[:, 2],
                        'Pressure': pred_np[:, 6],
                        'Density': pred_np[:, 7],
                        'Enthalpy': pred_np[:, 9]
                    }
                    
                    # 使用UTF-8编码保存，确保中文正常显示
                    pd.DataFrame(result_data).to_csv(save_path, index=False, encoding='utf-8-sig')
                    self.log.append(f"马赫数 {mach:.2f} 预测结果已保存")
                    
                except Exception as e:
                    self.log.append(f"马赫数 {mach:.2f} 预测失败: {str(e)}")
                    continue
                    
        except Exception as e:
            self.log.append(f"预测流程出错: {str(e)}")

    def save_model(self):
        if self.model is None: return
        path, _ = QFileDialog.getSaveFileName(self, "保存模型", "models/saved/prgnn.pth", "*.pth")
        if path:
            self.trainer.save_model(path)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PRGNN_GUI()
    window.show()
    sys.exit(app.exec_())
