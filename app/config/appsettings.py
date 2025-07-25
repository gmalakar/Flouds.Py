# =============================================================================
# File: appsettings.py
# Date: 2025-06-10
# Copyright (c) 2024 Goutam Malakar. All rights reserved.
# =============================================================================
import os
from typing import Optional

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str = Field(default="Flouds AI")
    debug: bool = Field(default=False)
    working_dir: str = Field(default=os.getcwd())
    is_production: bool = Field(default=True)


class ServerConfig(BaseModel):
    type: str = Field(default="uvicorn")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=5001)
    model_session_provider: str = Field(default="CPUExecutionProvider")
    model_config = {"protected_namespaces": ()}


class OnnxConfigSection(BaseModel):
    model_path: str = None
    config_file: str = Field(default="onnx_config.json")


class AppSettings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    onnx: OnnxConfigSection = Field(default_factory=OnnxConfigSection)
