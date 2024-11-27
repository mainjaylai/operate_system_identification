# 定义目标网站数组
$TARGET_HOSTS = @(
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
$LOG_FILE = "https_requests.log"

# 清空日志文件
Clear-Content -Path $LOG_FILE -ErrorAction SilentlyContinue
if (-not (Test-Path $LOG_FILE)) {
    New-Item -Path $LOG_FILE -ItemType File
}

# 设置安全协议
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12, [Net.SecurityProtocolType]::Tls13
[System.Net.ServicePointManager]::ServerCertificateValidationCallback = {$true}

foreach ($target in $TARGET_HOSTS) {
    Write-Host "开始对 $target 发送请求..."
    
    for ($i = 1; $i -le 50; $i++) {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Write-Host "[$target] Request $i - $timestamp"
        
        $maxRetries = 3
        $retryCount = 0
        
        do {
            try {
                $start = Get-Date
                
                # 使用 WebClient 替代 Invoke-WebRequest
                $webClient = New-Object System.Net.WebClient
                $webClient.Headers.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                $webClient.Headers.Add("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9")
                $webClient.Headers.Add("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
                
                # 下载内容
                $content = $webClient.DownloadString("https://$target")
                
                $end = Get-Date
                $duration = ($end - $start).TotalSeconds
                
                Write-Host "[$target] Success, Total time: $($duration)s"
                Add-Content -Path $LOG_FILE -Value "$timestamp - [$target] Request $i completed"
                break
            }
            catch {
                $retryCount++
                Write-Host "[$target] Error: $($_.Exception.Message)"
                Add-Content -Path $LOG_FILE -Value "$timestamp - [$target] Request $i failed - Error: $($_.Exception.Message)"
                
                if ($retryCount -eq $maxRetries) {
                    Write-Host "[$target] Max retries reached, moving to next request"
                    break
                }
                
                # 重试前等待2秒
                Start-Sleep -Seconds 2
            }
            finally {
                if ($webClient) {
                    $webClient.Dispose()
                }
            }
        } while ($retryCount -lt $maxRetries)
        
        # 增加延迟时间到1-2秒
        $delay = Get-Random -Minimum 1 -Maximum 2
        Start-Sleep -Seconds $delay
        
        # 每10个请求显示进度
        if ($i % 10 -eq 0) {
            Write-Host "[$target] Completed $i requests"
        }
    }
    
    Write-Host "完成对 $target 的请求"
    Write-Host "-----------------------------------"
}

Write-Host "所有请求已完成"
Write-Host "详细日志请查看: $LOG_FILE" 