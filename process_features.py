import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
import numpy as np


class FeatureProcessor:
    def __init__(self, csv_file, scaler_file="scaler.pkl"):
        self.csv_file = csv_file
        self.scaler_file = scaler_file
        self.scaler = None

    def load_features(self):
        """加载CSV文件中的特征"""
        try:
            df = pd.read_csv(self.csv_file)
            logger.info(f"成功加载特征文件: {self.csv_file}")
            return df
        except FileNotFoundError:
            logger.error(f"文件未找到: {self.csv_file}")
            return None
        except pd.errors.EmptyDataError:
            logger.error(f"文件为空: {self.csv_file}")
            return None

    def normalize_features(self, df, fit=True):
        """对特征进行归一化处理"""
        if df is None:
            logger.error("数据框为空，无法进行归一化")
            return None

        # 保存 label 列
        label = df["label"] if "label" in df.columns else None

        df = df.drop(columns=["flow_key", "label"], errors="ignore")
        # 填充 DataFrame 中的 NaN 值为 0
        df = df.fillna(0)

        # 获取需要进行 Min-Max 归一化的特征
        min_max_features = ["tcp_ACK", "tcp_PSH", "tcp_RST", "tcp_SYN", "tcp_FIN"]
        # 获取需要进行 Z-Score 归一化的特征
        z_score_features = ["ip.ttl_max", "ip.ttl_min", "ip.ttl_mean", "ip.ttl_std", 
                            "ip.len_max", "ip.len_min", "ip.len_mean", "ip.len_std"]
        # 获取需要进行 Log 变换的特征
        log_transform_features = ["tcp.window_size_value_max", "tcp.window_size_value_min", 
                                  "tcp.window_size_value_mean", "tcp.window_size_value_std"]

        # try:
        #     if fit:
        #         # Min-Max 归一化
        #         self.scaler = MinMaxScaler()
        #         df[min_max_features] = self.scaler.fit_transform(df[min_max_features])
        #         # Z-Score 归一化
        #         df[z_score_features] = (df[z_score_features] - df[z_score_features].mean()) / df[z_score_features].std()
        #         # Log 变换
        #         df[log_transform_features] = df[log_transform_features].map(np.log1p)
                
        #         joblib.dump(self.scaler, self.scaler_file)
        #         logger.info("特征归一化处理完成并保存缩放器")
        #     else:
        #         if self.scaler is None:
        #             self.scaler = joblib.load(self.scaler_file)
        #             logger.info("加载缩放器完成")
        #         df[min_max_features] = self.scaler.transform(df[min_max_features])
        #         df[z_score_features] = (df[z_score_features] - df[z_score_features].mean()) / df[z_score_features].std()
        #         df[log_transform_features] = df[log_transform_features].map(np.log1p)
        #         logger.info("特征归一化处理完成")
        # except ValueError as e:
        #     logger.info(f"归一化处理失败：{str(e)}")
        #     return df

        # 将 label 加回 DataFrame
        if label is not None:
            df["label"] = label

        return df

    def label_data(self, df):
        """根据操作系统对数据进行标注"""
        filename = os.path.basename(self.csv_file).lower()
        if "centos" in filename:
            label = 0
        elif "ubuntu" in filename:
            label = 1
        elif "debian" in filename:
            label = 2
        elif "windows" in filename:
            label = 3
        elif "macos" in filename:
            label = 4
        else:
            label = -1

        df["label"] = label
        logger.info(f"数据标注完成，标签为: {label}")
        return df

    def split_data_stratified(self, df):
        """将数据分为训练集和测试集，保持每个标签的比例一致"""
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["label"]
        )
        logger.info("数据分割完成: 训练集和测试集，保持标签比例一致")
        return train_df, test_df


def main():
    feature_dir = "feature_files"
    all_dfs = []  # 用于存储所有的 DataFrame

    for filename in os.listdir(feature_dir):
        if filename.endswith(".csv"):
            csv_file = os.path.join(feature_dir, filename)
            processor = FeatureProcessor(csv_file)
            df = processor.load_features()

            if df is not None:
                df = processor.label_data(df)  # 先标注数据
                all_dfs.append(df)

    # 合并所有的数据
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # 对合并后的数据进行归一化处理
        processor = FeatureProcessor("")  # 不需要特定文件名
        combined_df = processor.normalize_features(combined_df, fit=True)

        # 分割合并后的数据为训练集和测试集，保持标签比例一致
        train_df, test_df = processor.split_data_stratified(combined_df)

        # 打乱训练集和测试集
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info("所有特征文件的数据已合并、归一化并打乱")

        # 保存训练集和测试集
        train_df.to_csv("combined_train.csv", index=False)
        test_df.to_csv("combined_test.csv", index=False)
        logger.info("合并后的训练集和测试集已保存")


if __name__ == "__main__":
    main()
