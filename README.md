# 操作系统指纹识别系统

这是一个基于网络流量分析的操作系统指纹识别系统，通过分析网络数据包的特征来识别不同的操作系统类型。

[English Version](README.en.md)

## 功能特点

- 自动生成多种类型的网络流量数据
- 提取 PCAP 文件中的网络流量特征
- 支持多种操作系统的识别（CentOS、Ubuntu、Debian、Windows、macOS）
- 使用机器学习模型进行操作系统分类
- 集成多个分类器提高识别准确率

## 项目结构

```
project/
├── send_https_requests.sh # 网络流量生成脚本
├── pcap_extractor.py # PCAP 文件特征提取
├── process_features.py # 特征处理和数据预处理
├── model_training.py # 模型训练和评估
├── predict.py # 使用训练好的模型进行预测
├── feature_files/ # 特征文件存储目录
└── pcap_files/ # PCAP 文件存储目录
```

## 技术栈

- Python 3.x
- scikit-learn
- pandas
- numpy
- pyshark
- loguru
- bash 脚本

## 安装说明

1. 安装依赖包：

```bash
pip install pandas numpy scikit-learn pyshark loguru
```

2. 安装系统依赖：

```bash
sudo apt-get install tcpdump tshark
```

## 使用方法

1. 生成网络流量数据：

```bash
chmod +x send_https_requests.sh
./send_https_requests.sh
```

2. 提取 PCAP 文件特征：

```bash
python pcap_extractor.py
```

3. 处理特征数据：

```bash
python process_features.py
```

4. 训练模型：

```bash
python model_training.py
```

5. 使用模型进行预测：

```bash
python predict.py --pcap <path_to_pcap_file> --model <path_to_model_file>
```

## 特征说明

系统提取的主要特征包括：
- IP 包长度统计特征（最大值、最小值、平均值、标准差）
- TTL 值统计特征
- TCP 窗口大小统计特征
- TCP 标志位统计
- TCP 时间戳特征

## 模型说明

使用集成学习方法，包含以下分类器：
- K近邻分类器 (KNN)
- 支持向量机 (SVM)
- 随机森林 (Random Forest)
- 逻辑回归 (Logistic Regression)

## 注意事项

1. 运行流量生成脚本需要 root 权限
2. 确保有足够的磁盘空间存储 PCAP 文件
3. 建议在测试环境中运行流量生成脚本

## 文件下载

需要的文件可以从以下链接下载：[Google Drive](https://drive.google.com/drive/folders/1JDthSZJlkorWXAPdICTXd7PGNyvbqntV?usp=drive_link)

## 许可证

MIT License

## 贡献指南

1. Fork 该项目
2. 创建特性分支
3. 提交更改
4. 发起 Pull Request

## 联系方式

如有问题，请提交 Issue 或 Pull Request。

