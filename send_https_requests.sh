#!/bin/bash

# 定义目标网站数组
declare -a TARGET_HOSTS=(
    "www.baidu.com"
    "www.taobao.com"
    "www.jd.com"
    "www.qq.com"
    "www.163.com"
    "www.sina.com.cn"
    "www.weibo.com"
    "www.zhihu.com"
    "www.bilibili.com"
    "www.aliyun.com"
    "www.tencent.com"
    "www.douyin.com"
    "www.xiaomi.com"
    "www.huawei.com"
    "www.360.cn"
    "www.ctrip.com"
    "www.meituan.com"
    "www.pinduoduo.com"
    "www.suning.com"
    "www.iqiyi.com"
)

# 日志文件
LOG_FILE="https_requests.log"

# 清空日志文件
> $LOG_FILE

# 对每个目标网站发送请求
for target in "${TARGET_HOSTS[@]}"; do
    echo "开始对 $target 发送请求..."
    
    for i in {1..50}; do  # 每个网站发送50次请求，总共1000次
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$target] Request $i - $timestamp"
        
        # 发送请求并记录结果
        curl -k -s \
             -w "[$target] HTTP Status: %{http_code}, Total time: %{time_total}s\n" \
             -H "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
             -H "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8" \
             -H "Accept-Language: zh-CN,zh;q=0.9,en;q=0.8" \
             --connect-timeout 5 \
             --max-time 10 \
             "https://$target" > /dev/null
        
        # 记录到日志文件
        echo "$timestamp - [$target] Request $i completed" >> $LOG_FILE
        
        # 随机延迟0.1到0.5秒
        sleep $(awk -v min=0.1 -v max=0.5 'BEGIN{srand(); print min+rand()*(max-min)}')
        
        # 每10个请求显示进度
        if [ $((i % 10)) -eq 0 ]; then
            echo "[$target] Completed $i requests"
        fi
    done
    
    echo "完成对 $target 的请求"
    echo "-----------------------------------"
done

echo "所有请求已完成"
echo "详细日志请查看: $LOG_FILE" 



sudo tcpdump -i any \
    -w captured_traffic.pcap \
    -n \
    -s 0 \
    -v \
    'tcp'

# 生成DNS流量
echo "生成DNS流量..."
for target in "${TARGET_HOSTS[@]}"; do
    dig $target >> $LOG_FILE
done

# 生成FTP流量
echo "生成FTP流量..."
ftp -inv <<EOF
open ftp.example.com
user anonymous ""
ls
bye
EOF

# 生成SMTP流量
echo "生成SMTP流量..."
swaks --to test@example.com --from user@example.com --server smtp.example.com

# 生成POP3流量
echo "生成POP3流量..."
openssl s_client -connect pop3.example.com:995 <<EOF
USER your_username
PASS your_password
QUIT
EOF

# 生成IMAP流量
echo "生成IMAP流量..."
openssl s_client -connect imap.example.com:993 <<EOF
a login your_username your_password
a logout
EOF