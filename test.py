import pyshark

cap = pyshark.FileCapture("pcap_files/test2.pcapng")

# 按流分组
for packet in cap:
    if hasattr(packet, "tls") and hasattr(packet, "tcp"):
        # 打印所有层的属性
        for layer in packet.layers:
            print(f"\n=== {layer.layer_name} Layer ===")
            # 获取并打印该层的所有字段
            for field_name in layer.field_names:
                print(f"{field_name}: {getattr(layer, field_name)}")
        
        