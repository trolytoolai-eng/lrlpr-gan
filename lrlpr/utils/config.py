# -*- coding: utf-8 -*-
"""
Đọc file cấu hình yaml (config/...) làm giá trị MẶC ĐỊNH cho các tham số dòng lệnh.
Khóa nào trùng tên tham số thì được dùng; khóa lạ bị bỏ qua. Tham số gõ trên dòng lệnh luôn thắng.
"""

import argparse

import yaml


def apply_yaml_defaults(ap, argv=None):
    """Nếu có --config <file>, nạp yaml và đặt làm mặc định cho parser ap (gọi TRƯỚC ap.parse_args())."""
    ap.add_argument("--config", default="", help="file yaml cấu hình; tham số dòng lệnh ghi đè giá trị trong file")
    pre_ap = argparse.ArgumentParser(add_help=False)   # bộ đọc riêng, chỉ để lấy --config (parser chính còn tham số bắt buộc)
    pre_ap.add_argument("--config", default="")
    pre, _ = pre_ap.parse_known_args(argv)
    if not pre.config:
        return {}
    with open(pre.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    known = {a.dest for a in ap._actions}
    used = {k: v for k, v in cfg.items() if k in known}
    ap.set_defaults(**used)
    return used
