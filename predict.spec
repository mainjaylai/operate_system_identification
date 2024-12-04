# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['predict.py'],
    pathex=['/usr/local/lib'],  # 添加共享库路径
    binaries=[],
    datas=[
        ('voting_classifier.pkl', '.'),  # 将 voting_classifier.pkl 包含在打包中
        ('scaler.pkl', '.'),             # 将 scaler.pkl 包含在打包中
        ('pcap_extractor.py', '.'),      # 手动包含 pcap_extractor.py
        ('process_features.py', '.'),    # 手动包含 process_features.py
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='predict',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='predict',
)
