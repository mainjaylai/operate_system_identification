import argparse
from pcap_extractor import PcapFeatureExtractor
from process_features import FeatureProcessor
import joblib
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PcapPredictor:
    def __init__(self, model_path, scaler_path="scaler.pkl"):
        """初始化预测器
        Args:
            model_path: 训练好的模型文件路径
            scaler_path: 特征缩放器文件路径
        """
        self.model = joblib.load(model_path)
        self.pcap_extractor = PcapFeatureExtractor()
        self.feature_processor = FeatureProcessor(
            csv_file=None, scaler_file=scaler_path
        )
        self.feature_processor.scaler = joblib.load(scaler_path)
        # 添加完整的类别标签映射
        self.labels = {
            0: "CentOS",
            1: "Ubuntu",
            2: "Debian",
            3: "Windows",
            4: "macOS",
            -1: "未知系统",
        }

    def predict(self, pcap_file):
        """对输入的pcap文件进行预测
        Args:
            pcap_file: pcap文件路径
        Returns:
            预测结果
        """
        try:
            # 提取pcap特征
            logger.info(f"正在处理pcap文件: {pcap_file}")
            flows_df = self.pcap_extractor.process_pcap(pcap_file)

            # 处理特征
            features = self.feature_processor.normalize_features(flows_df, fit=False)

            # 预测
            predictions = self.model.predict(features)
            probabilities = self.model.predict_proba(features)

            return predictions, probabilities

        except Exception as e:
            logger.error(f"预测过程发生错误: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description="预测pcap文件中的流量类型")
    parser.add_argument("--pcap", required=True, help="输入pcap文件路径")
    parser.add_argument(
        "--model",
        default="voting_classifier.pkl",
        help="模型文件路径 (voting_classifier.pkl)",
    )
    args = parser.parse_args()

    predictor = PcapPredictor(args.model)
    predictions, probabilities = predictor.predict(args.pcap)

    # 输出预测结果
    logger.info("预测结果:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        os_name = predictor.labels.get(pred, f"未知类别({pred})")
        logger.info(f"流量 {i+1}: 操作系统 = {os_name}, 置信度 = {max(prob):.3f}")


if __name__ == "__main__":
    main()
