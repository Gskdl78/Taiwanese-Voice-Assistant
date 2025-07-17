#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台語語音對話系統 - 配置文件
管理系統設置，避免硬編碼敏感信息
"""

import os
from typing import Optional

class Config:
    """系統配置類"""
    
    def __init__(self):
        """初始化配置"""
        self.load_config()
    
    def load_config(self):
        """載入配置"""
        # 遠端TTS服務配置
        self.REMOTE_TTS_HOST = os.getenv('REMOTE_TTS_HOST', 'YOUR_REMOTE_TTS_HOST')
        self.REMOTE_TTS_PORT = int(os.getenv('REMOTE_TTS_PORT', '5000'))
        
        # 意傳科技API配置
        self.ITHUAN_API_BASE_URL = os.getenv('ITHUAN_API_BASE_URL', 'https://hokbu.ithuan.tw')
        
        # Ollama配置
        self.OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'gemma3:4b')
        
        # 系統配置
        self.CLEANUP_FILES = os.getenv('CLEANUP_FILES', 'true').lower() == 'true'
        self.DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    
    def get_remote_tts_url(self) -> str:
        """獲取遠端TTS服務URL"""
        if self.REMOTE_TTS_HOST == 'YOUR_REMOTE_TTS_HOST':
            raise ValueError("請設置 REMOTE_TTS_HOST 環境變數或修改 config.py")
        return f"http://{self.REMOTE_TTS_HOST}:{self.REMOTE_TTS_PORT}"
    
    def get_remote_tts_display_name(self) -> str:
        """獲取遠端TTS服務顯示名稱（不包含敏感信息）"""
        if self.REMOTE_TTS_HOST == 'YOUR_REMOTE_TTS_HOST':
            return "遠端TTS服務（未配置）"
        return f"自訓練遠端TTS服務（{self.REMOTE_TTS_HOST}:{self.REMOTE_TTS_PORT}）"
    
    def is_remote_tts_configured(self) -> bool:
        """檢查遠端TTS服務是否已配置"""
        return self.REMOTE_TTS_HOST != 'YOUR_REMOTE_TTS_HOST'

# 創建全域配置實例
config = Config() 