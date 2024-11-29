import pyshark
import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger
from collections import defaultdict


class PcapFeatureExtractor:

    def __init__(self):
        self.statistic_cols = ["ip.len", "ip.ttl", "tcp.window_size_value"]

    def process_tcp_timestamp_diff(self, tcp_timestamp_list):
        tcp_timestamp_list = np.array(tcp_timestamp_list).astype(float)
        tcp_timestamp_list = np.sort(tcp_timestamp_list)
        timestamp_diffs = np.diff(tcp_timestamp_list)
        return np.mean(timestamp_diffs), np.std(timestamp_diffs)

    def parse_tcp_flags(self, flags_value):
        # 将十六进制表示的 flags 值与标志位比较
        flags = {
            "URG": 0x20,
            "ACK": 0x10,
            "PSH": 0x08,
            "RST": 0x04,
            "SYN": 0x02,
            "FIN": 0x01,
        }

        active_flags = []

        for flag, bit in flags.items():
            if flags_value & bit:  # 如果该位被设置
                active_flags.append(flag)

        return active_flags

    def get_flow_key(self, packet):
        """生成流的唯一标识符"""
        try:
            if hasattr(packet, "ip") and hasattr(packet, "tcp"):
                # 使用五元组作为流标识
                src_ip = packet.ip.src
                dst_ip = packet.ip.dst
                src_port = packet.tcp.srcport
                dst_port = packet.tcp.dstport
                protocol = "TCP"

                # 确保源IP/端口较小的在前，保证双向流量的一致性
                if src_ip < dst_ip or (src_ip == dst_ip and src_port < dst_port):
                    return f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
                else:
                    return f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
            return None
        except AttributeError:
            return None

    def extract_field_safely(self, packet, field_path):
        """安全地从数据包中提取字段值"""
        try:
            parts = field_path.split(".")
            current = packet
            for part in parts:
                current = getattr(current, part, None)
                if current is None:
                    return None
            return current
        except AttributeError:
            return None

    def process_packet(self, packet):
        """处理单个数据包并提取特征"""
        if not hasattr(packet, "tcp"):
            return None

        feature_dict = {}

        # IP相关特征
        ip_fields = [
            "ip.flags_df",
            "ip.ttl",
            "ip.len",
            # "ip.dsfield_dscp",
            # "ip.frag_offset",
            # "ip.proto",
            # "ip.dsfield_ecn",
        ]

        # TCP相关特征
        tcp_fields = [
            "tcp.flags",
            "tcp.window_size_value",
            "tcp.options_timestamp_tsval",
        ]

        # 提取所有特征
        for field in ip_fields + tcp_fields:
            feature_dict[field] = self.extract_field_safely(packet, field)

        return feature_dict

    def process_flow(self, packets):
        """处理单个流的所有数据包并提取特征"""
        flow_features = defaultdict(list)

        # 收集流中所有包的特征
        for packet in packets:
            features = self.process_packet(packet)
            if features:
                for key, value in features.items():
                    flow_features[key].append(value)

        # 如果没有提取到特征，返回None
        if not flow_features:
            return None

        # 对流特征进行统计
        flow_stats = {
            "tcp_ACK": 0,
            "tcp_PSH": 0,
            "tcp_RST": 0,
            "tcp_SYN": 0,
            "tcp_FIN": 0,
        }
        for key, values in flow_features.items():
            values = [v for v in values if v is not None]
            if not values:
                continue

            if key == "ip.flags_df":
                flow_stats[key] = 1 if values[0] == "True" else 0
                continue
            if key == "ip.frag_offset":
                flow_stats[key] = max(list(map(float, values)))
                continue
            if key == "tcp.flags":
                for v in values:
                    active_flags = self.parse_tcp_flags(int(v, 16))
                    for flag in active_flags:
                        if flag == "URG":
                            continue
                        flow_stats[f"tcp_{flag}"] += 1
                continue
            if key == "tcp.options_timestamp_tsval":
                if len(values) > 1:  # 确保至少有两个值才计算
                    mean, std = self.process_tcp_timestamp_diff(values)
                    flow_stats[f"{key}_mean"] = mean
                    flow_stats[f"{key}_std"] = std
                else:
                    flow_stats[f"{key}_mean"] = float(values[0]) if values else 0
                    flow_stats[f"{key}_std"] = 0
                continue

            if key in self.statistic_cols:
                if values:  # 确保有值才进行计算
                    values = list(map(float, values))
                    flow_stats[f"{key}_max"] = np.max(values)
                    flow_stats[f"{key}_min"] = np.min(values)
                    flow_stats[f"{key}_mean"] = np.mean(values)
                    flow_stats[f"{key}_std"] = np.std(values) if len(values) > 1 else 0
                else:
                    flow_stats[f"{key}_max"] = 0
                    flow_stats[f"{key}_min"] = 0
                    flow_stats[f"{key}_mean"] = 0
                    flow_stats[f"{key}_std"] = 0
                continue
            # 对于非数值型特征，保留第一个非空值
            flow_stats[key] = values[0]

        return flow_stats

    def process_pcap(self, pcap_file):
        """处理PCAP文件并按流提取特征"""
        flows = defaultdict(list)
        cap = pyshark.FileCapture(pcap_file)

        # 按流分组
        for packet in cap:
            flow_key = self.get_flow_key(packet)
            if flow_key:
                flows[flow_key].append(packet)

        # 处理每个流
        flow_features = []
        for flow_key, packets in flows.items():
            features = self.process_flow(packets)
            if features:
                features["flow_key"] = flow_key
                flow_features.append(features)

        cap.close()
        return pd.DataFrame(flow_features)

    def extract_and_process(self, pcap_dir):
        """提取和处理所有PCAP文件的特征，并将每个PCAP文件的特征单独存储到CSV文件中"""
        for filename in os.listdir(pcap_dir):
            if filename.endswith(".pcap") or filename.endswith(".pcapng"):
                pcap_path = os.path.join(pcap_dir, filename)
                logger.info(f"处理文件: {pcap_path}")

                df = self.process_pcap(pcap_path)

                # 将特征存储到单独的CSV文件中
                output_file = os.path.join(
                    "feature_files", f"{os.path.splitext(filename)[0]}_features.csv"
                )
                df.to_csv(output_file, index=False)
                logger.info(f"特征已保存到: {output_file}")


def main():
    pcap_dir = "pcap_files"
    extractor = PcapFeatureExtractor()
    extractor.extract_and_process(pcap_dir)


if __name__ == "__main__":
    main()
