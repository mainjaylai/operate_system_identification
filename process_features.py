import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import joblib


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

        # 填充 DataFrame 中的 NaN 值为 0
        df = df.fillna(0)

        features_to_normalize = [
            "ip.len_max",
            "ip.len_min",
            "ip.len_mean",
            "ip.len_std",
            "ip.ttl_max",
            "ip.ttl_min",
            "ip.ttl_mean",
            "ip.ttl_std",
            "tcp.window_size_value_max",
            "tcp.window_size_value_min",
            "tcp.window_size_value_mean",
            "tcp.window_size_value_std",
            "tcp.options_timestamp_tsval_mean",
            "tcp.options_timestamp_tsval_std",
        ]

        if fit:
            self.scaler = MinMaxScaler()
            df[features_to_normalize] = self.scaler.fit_transform(
                df[features_to_normalize]
            )
            joblib.dump(self.scaler, self.scaler_file)
            logger.info("特征归一化处理完成并保存缩放器")
        else:
            if self.scaler is None:
                self.scaler = joblib.load(self.scaler_file)
                logger.info("加载缩放器完成")
            df[features_to_normalize] = self.scaler.transform(df[features_to_normalize])
            logger.info("特征归一化处理完成")

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
        else:
            label = -1

        df["label"] = label
        logger.info(f"数据标注完成，标签为: {label}")
        return df

    def split_data(self, df):
        """将数据分为训练集和测试集"""
        df = df.drop(columns=["flow_key"], errors="ignore")

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        logger.info("数据分割完成: 训练集和测试集")
        return train_df, test_df


def main():
    feature_dir = "feature_files"
    all_train_dfs = []  # 用于存储所有的训练集 DataFrame
    all_test_dfs = []   # 用于存储所有的测试集 DataFrame

    for filename in os.listdir(feature_dir):
        if filename.endswith(".csv"):
            csv_file = os.path.join(feature_dir, filename)
            processor = FeatureProcessor(csv_file)
            df = processor.load_features()

            if df is not None:
                df = processor.normalize_features(df, fit=True)
                df = processor.label_data(df)

                # 分割每个文件为训练集和测试集
                train_df, test_df = processor.split_data(df)
                all_train_dfs.append(train_df)
                all_test_dfs.append(test_df)

    # 合并所有的训练集和测试集
    if all_train_dfs and all_test_dfs:
        combined_train_df = pd.concat(all_train_dfs, ignore_index=True)
        combined_test_df = pd.concat(all_test_dfs, ignore_index=True)

        # 打乱合并后的训练集和测试集
        combined_train_df = combined_train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        combined_test_df = combined_test_df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info("所有特征文件的训练集和测试集已合并并打乱")

        # 保存训练集和测试集
        combined_train_df.to_csv(os.path.join(feature_dir, "combined_train.csv"), index=False)
        combined_test_df.to_csv(os.path.join(feature_dir, "combined_test.csv"), index=False)
        logger.info("合并后的训练集和测试集已保存")

if __name__ == "__main__":
    main()
