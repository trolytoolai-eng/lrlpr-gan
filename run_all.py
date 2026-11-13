# -*- coding: utf-8 -*-
"""
run_all.py: chạy toàn bộ quy trình LRLPR theo ba vòng, từ MỘT chỗ duy nhất.

  Vòng 1: tách dữ liệu, M0/M1/HR, OCR fine-tune, M3 (fine-tune), M4 (có tăng cường), đánh giá, test, báo cáo.
  Vòng 2: loss OCR có gradient (M5a, M5b, M5c), dạy lại OCR trên ảnh GAN (nhánh có dấu +), đánh giá, test, báo cáo.
  Vòng 3: mạng nhiều frame (M6a), đánh giá, test, báo cáo.

Ví dụ:
  python run_all.py --round all                         # chạy cả ba vòng liền nhau
  python run_all.py --round 2                           # chỉ vòng 2 (cần kết quả vòng 1 trên Drive)
  python run_all.py --round 1 --stage m3,eval_m3        # chỉ vài bước của vòng 1
  python run_all.py --round 1 --set epochs=1 --set workers=0   # ghi đè tham số (chạy thử nhanh)
  python run_all.py --round 3 --set m6a.init_from=M5a   # ghi đè riêng cho một nhánh

Các bước (--stage) của từng vòng:
  vòng 1: split, baselines, ocr, m3, eval_m3, m4, eval_m4, choose, test, report, bundle
  vòng 2: check, m5, eval_m5, ocr_sr, eval_plus, choose, test, report, bundle
  vòng 3: check, m6, eval_m6, choose, test, report, bundle

Quy ước: mọi tham số nằm trong config/default.yaml (chung) và config/LRLPR/*.yaml (từng nhánh).
Mọi kết quả và bằng chứng lưu trong drive_dir: results/, evidence/, ckpt/, splits/.
Tập test chỉ chạy MỘT lần mỗi vòng (khóa evidence/test_run*.lock); muốn chạy lại phải xóa khóa thủ công.
"""

import argparse
import os
import shutil
import sys
import zipfile

import pandas as pd
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lrlpr.models import setup_repo  # noqa: E402
from lrlpr.utils import api as C  # noqa: E402
from lrlpr.utils_test import eval_lib as E  # noqa: E402
from lrlpr.utils_test import report_lib as R  # noqa: E402

PY = sys.executable
ROUND_STAGES = {
    1: ["split", "baselines", "ocr", "m3", "eval_m3", "m4", "eval_m4", "choose", "test", "report", "bundle"],
    2: ["check", "m5", "eval_m5", "ocr_sr", "eval_plus", "choose", "test", "report", "bundle"],
    3: ["check", "m6", "eval_m6", "choose", "test", "report", "bundle"],
}


# ----------------------------------------------------------------------------
# Đọc cấu hình
# ----------------------------------------------------------------------------
def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_overrides(items):
    """['epochs=1', 'm6a.init_from=M5a'] -> ({'epochs': 1}, {'m6a': {'init_from': 'M5a'}})"""
    glob, per = {}, {}
    for it in items or []:
        k, v = it.split("=", 1)
        v = yaml.safe_load(v)
        if "." in k:
            b, kk = k.split(".", 1)
            per.setdefault(b, {})[kk] = v
        else:
            glob[k] = v
    return glob, per


class Ctx:
    """Giữ đường dẫn, cấu hình và các hàm dùng chung của cả ba vòng."""

    def __init__(self, args):
        self.args = args
        self.cfg = read_yaml(os.path.join(HERE, args.config))
        self.glob, self.per = parse_overrides(args.set)
        for k, v in self.glob.items():
            if k in self.cfg:
                self.cfg[k] = v
        c = self.cfg
        self.DRIVE = c["drive_dir"]
        self.SEED = int(c["seed"])
        self.WORKERS = int(c["workers"])
        self.SPLIT_DIR = f"{self.DRIVE}/splits"
        self.RESULTS_DIR = f"{self.DRIVE}/results"
        self.EVID_DIR = f"{self.DRIVE}/evidence"
        self.CKPT_DIR = f"{self.DRIVE}/ckpt"
        self.DATA_DIR = f"{c['local_dir']}/data"
        self.REPO_DIR = f"{c['local_dir']}/lpsrgan"
        self.ZIP_PATH = f"{self.DRIVE}/{c['zip_name']}"
        self.OCR_V1 = f"{self.CKPT_DIR}/gplpr_ft.pth"     # OCR vòng 1 (dạy bằng ảnh HR)
        self.OCR_V2 = f"{self.CKPT_DIR}/gplpr_sr.pth"     # OCR vòng 2 (dạy lại trên ảnh GAN)
        self.only = set(args.only.split(",")) if args.only else None
        self.stages = set(args.stage.split(",")) if args.stage else None
        self.DATA_ROOT = None
        self.PRETRAINED = None
        self.rows = {}
        self.ocr1 = self.ocr2 = None
        self.MS = {}
        self.timing = "timing.csv"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- cấu hình từng nhánh
    def branch_cfg(self, name):
        b = read_yaml(os.path.join(HERE, "config", "LRLPR", f"{name}.yaml"))
        for k, v in self.glob.items():
            if k in b:
                b[k] = v
        b.update(self.per.get(name, {}))
        return b

    def branches(self, names):
        """Danh sách tên file yaml, lọc theo --only (nếu có)."""
        return [n for n in names if self.only is None or n in self.only]

    def want(self, stage):
        return self.stages is None or stage in self.stages

    # ---- chuẩn bị chung
    def start(self, env_tag):
        try:  # nối Google Drive (chỉ chạy được trên Colab)
            from google.colab import drive
            drive.mount("/content/drive")
        except ImportError:
            print("Không chạy trên Colab: bỏ qua bước nối Google Drive.")
        for d in (self.SPLIT_DIR, self.RESULTS_DIR, self.EVID_DIR, self.CKPT_DIR):
            os.makedirs(d, exist_ok=True)
        C.set_seed(self.SEED)
        print("Thiết bị:", self.device, "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHÔNG có GPU")
        if self.device.type == "cpu":
            print("Cảnh báo: không có GPU! Chạy trên CPU sẽ rất chậm (Colab: Runtime > Change runtime type > T4 GPU).")
        C.save_environment(self.EVID_DIR, env_tag)
        self.DATA_ROOT = C.ensure_dataset(self.ZIP_PATH, self.DATA_DIR)
        print("DATA_ROOT =", self.DATA_ROOT)

    def prepare_repo(self):
        setup_repo.setup(self.REPO_DIR, self.EVID_DIR)    # tải repo LPSRGAN đúng commit + áp 2 bản vá nhỏ
        self.PRETRAINED = self.cfg.get("pretrained_path") or setup_repo.download(
            setup_repo.PRETRAINED_URL, f"{self.CKPT_DIR}/pretrained/best_model_Epoch_116.pth", f"{self.EVID_DIR}/patches.log")

    def load_rows(self):
        self.rows = {s: C.index_tracks(C.read_split_csv(f"{self.SPLIT_DIR}/split_{s}.csv"), self.DATA_ROOT) for s in ("val", "test")}

    def timer(self, name):
        return C.StepTimer(name, self.EVID_DIR, self.timing)

    def ckpt_of(self, branch):
        """File trọng số mạng làm nét của một nhánh ('M2' = trọng số công bố)."""
        return self.PRETRAINED if branch == "M2" else f"{self.CKPT_DIR}/lpsrgan_{branch}/best.pth"

    # ---- chạy các script huấn luyện (lệnh con, để thấy tiến độ ngay)
    def train_sr(self, name, ocr_ckpt, title):
        b = self.branch_cfg(name)
        init = self.ckpt_of("M2") if b["init_from"] == "pretrained" else self.ckpt_of(b["init_from"])
        out = f"{self.CKPT_DIR}/lpsrgan_{b['branch']}"
        cmd = [PY, f"{HERE}/train.py", "--config", f"{HERE}/config/LRLPR/{name}.yaml", "--repo", self.REPO_DIR,
               "--data_root", self.DATA_ROOT, "--split_dir", self.SPLIT_DIR, "--init", init, "--out", out,
               "--ocr_ckpt", ocr_ckpt, "--workers", self.WORKERS, "--seed", self.SEED,
               "--vgg_pretrained", int(self.cfg["vgg_pretrained"])]
        if self.cfg.get("max_train_tracks"):
            cmd += ["--max_train_tracks", self.cfg["max_train_tracks"]]
        for k, v in {**self.glob, **self.per.get(name, {})}.items():   # ghi đè gõ trên dòng lệnh thắng file yaml
            if k in b and k not in ("branch", "init_from"):
                cmd += [f"--{k}", v]
        with self.timer(title):
            C.run_script(cmd)
        print("Checkpoint tốt nhất:", f"{out}/best.pth")

    # ---- đánh giá
    def eval_sr(self, split, branches):
        """Nhánh dùng mạng làm nét: chạy mạng, lưu ảnh, đọc bằng OCR vòng 1."""
        for b in branches:
            ck = self.ckpt_of(b)
            if not os.path.exists(ck):
                print(f"Bỏ qua {b}: chưa có {ck}")
                continue
            _, ms = E.evaluate_branch(b, split, self.rows[split], self.RESULTS_DIR, self.ocr1, self.device, self.REPO_DIR, ck)
            if ms:
                self.MS[b] = ms

    def eval_plus(self, split, branches):
        """Nhánh '+': đọc lại ảnh đã có của nhánh gốc bằng OCR vòng 2."""
        for b in branches:
            if os.path.exists(f"{self.RESULTS_DIR}/{b}/sr_outputs_{split}.npz"):
                E.evaluate_with_ocr(b + "+", b, split, self.rows[split], self.RESULTS_DIR, self.ocr2, self.device, self.OCR_V2)
            else:
                print(f"Bỏ qua {b}+: chưa có ảnh {b} trên tập {split}")

    def choose(self, fname):
        """Chốt cách gộp và mô hình tốt nhất, CHỈ dựa vào val."""
        summ = R.gather_summary(self.RESULTS_DIR)
        cands = [b for b in R.available_branches(self.RESULTS_DIR) if b != "HR"]
        chosen = E.choose_on_val(summ, cands)
        C.write_json(f"{self.RESULTS_DIR}/{fname}", chosen)
        print("Cách gộp tốt nhất (val):", chosen["cach_gop_tot_nhat"])
        print("Mô hình tốt nhất (val):", chosen["mo_hinh_tot_nhat"], f"({chosen['exact_match_val_cua_mo_hinh_tot_nhat'] * 100:.2f}%)")
        print(pd.DataFrame(chosen["bang_val"]).to_string(index=False))
        return cands

    def need(self, items, hint):
        missing = {k: v for k, v in items.items() if not os.path.exists(v)}
        assert not missing, f"{hint}: " + str(missing)

    def keep_copy(self, names, suffix):
        for name in names:
            src, dst = f"{self.RESULTS_DIR}/{name}", f"{self.RESULTS_DIR}/{name.replace('.', suffix + '.')}"
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copyfile(src, dst)
                print("Đã giữ bản cũ:", dst)

    def report(self, suffix, title, chosen_name, ms_name, train_branches, comparison, with_val_plot, evidence_readme=False):
        sfx = suffix
        with self.timer(title):
            summary = R.gather_summary(self.RESULTS_DIR, out_csv=f"{self.RESULTS_DIR}/summary.csv")
            chosen = C.read_json(f"{self.RESULTS_DIR}/{chosen_name}")
            fusion, best = chosen["cach_gop_tot_nhat"], chosen["mo_hinh_tot_nhat"]
            ms = C.read_json(f"{self.RESULTS_DIR}/{ms_name}") if os.path.exists(f"{self.RESULTS_DIR}/{ms_name}") else {}
            R.write_comparison_md(f"{self.RESULTS_DIR}/{comparison}", summary, chosen, f"{self.EVID_DIR}/{self.timing}", ms)
            R.plot_overview(summary, fusion, f"{self.RESULTS_DIR}/so_sanh_tong_quan{sfx}.png")
            R.plot_groups(summary, fusion, "test", f"{self.RESULTS_DIR}/so_sanh_theo_nhom_test{sfx}.png")
            if with_val_plot:
                R.plot_groups(summary, fusion, "val", f"{self.RESULTS_DIR}/so_sanh_theo_nhom_val{sfx}.png")
            logs = {b: f"{self.CKPT_DIR}/lpsrgan_{b}/train_log.csv" for b in train_branches}
            R.plot_training(logs, f"{self.RESULTS_DIR}/duong_cong_huan_luyen{sfx}.png")
            R.confusion_pairs(self.RESULTS_DIR, fusion, f"{self.RESULTS_DIR}/confusion_pairs{sfx}.csv")
            store = R.ResultStore(self.RESULTS_DIR)
            R.make_examples(store, "test", fusion, best, f"{self.RESULTS_DIR}/examples{sfx}")
            if evidence_readme:
                R.write_evidence_readme(f"{self.EVID_DIR}/README.md")
        print(open(f"{self.RESULTS_DIR}/{comparison}", encoding="utf-8").read())

    def bundle(self, zip_name):
        roots = {"results": self.RESULTS_DIR, "evidence": self.EVID_DIR, "splits": self.SPLIT_DIR, "ckpt": self.CKPT_DIR}
        n = R.write_manifest(roots, f"{self.EVID_DIR}/MANIFEST.json")
        print("MANIFEST.json:", n, "file")
        path = f"{self.DRIVE}/{zip_name}"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:   # không gồm ảnh .npz và checkpoint cho nhẹ
            for name, root in roots.items():
                for dp, _, fns in os.walk(root):
                    for fn in fns:
                        if fn.endswith((".npz", ".pth", ".tmp")):
                            continue
                        p = os.path.join(dp, fn)
                        z.write(p, os.path.join(name, os.path.relpath(p, root)))
        print(f"Đã tạo {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


# ----------------------------------------------------------------------------
# VÒNG 1
# ----------------------------------------------------------------------------
def round1(x):
    x.timing = "timing.csv"
    x.start("full")
    t = f"{x.EVID_DIR}/timing.csv"
    if x.cfg.get("start_fresh") and x.stages is None and os.path.exists(t):   # chạy trọn vòng: cất bảng thời gian cũ đi
        os.replace(t, f"{x.EVID_DIR}/timing_cu_{int(__import__('time').time())}.csv")
        print("Đã cất bảng thời gian cũ.")

    if x.want("split"):
        with x.timer("00 · Giải nén dữ liệu & tách train/val/test"):
            C.run_script([PY, f"{HERE}/split.py", "--dataset", x.DATA_ROOT, "--out", x.SPLIT_DIR, "--seed", x.SEED])
        tr = pd.read_csv(f"{x.SPLIT_DIR}/split_train.csv")
        va = pd.read_csv(f"{x.SPLIT_DIR}/split_val.csv")
        te = pd.read_csv(f"{x.SPLIT_DIR}/split_test.csv")
        ex = pd.read_csv(f"{x.SPLIT_DIR}/excluded.csv")
        print("train / val / test =", len(tr), len(va), len(te))
        print("Track bị loại khỏi train/val:", ex["reason"].value_counts().to_dict() if len(ex) else "không có")
        assert not (set(tr.track) & set(va.track)), "trùng track train/val"
        assert not (set(tr.plate_text) & set(va.plate_text)), "rò rỉ biển số train/val"
        assert not ((set(tr.plate_text) | set(va.plate_text)) & set(te.plate_text)), "rò rỉ biển số sang test"
        print("Không rò rỉ giữa train / val / test")

    x.prepare_repo()
    x.load_rows()

    if x.want("baselines"):
        with x.timer("01 · Nhánh M0/M1/HR (CPU)"):
            for split in ("val", "test"):
                rows = C.index_tracks(C.read_split_csv(f"{x.SPLIT_DIR}/split_{split}.csv"), x.DATA_ROOT)
                jobs = [("M0", C.prep_plain_u8, "lr"),
                        ("M1", lambda im: C.prep_plain_u8(C.cv_pipeline_rgb(im)), "lr"),
                        ("HR", C.prep_hr_u8, "hr")]
                for br, fn, kind in jobs:
                    if br == "HR" and split == "test":
                        continue  # tập test không có ảnh HR
                    frames = E.build_precomputed_frames(rows, fn, kind, workers=4, label=f"{br} {split}")
                    E.save_frames_npz(f"{x.RESULTS_DIR}/{br}/sr_outputs_{split}.npz", rows, frames)

    if x.want("ocr"):
        oc = read_yaml(os.path.join(HERE, "config", "LRLPR", "ocr_ft.yaml"))
        cmd = [PY, f"{HERE}/train_ocr.py", "--config", f"{HERE}/config/LRLPR/ocr_ft.yaml", "--repo", x.REPO_DIR,
               "--data_root", x.DATA_ROOT, "--split_dir", x.SPLIT_DIR, "--init", f"{x.DRIVE}/{x.cfg['rodosol']}",
               "--out", x.OCR_V1, "--workers", x.WORKERS, "--seed", x.SEED]
        if x.cfg.get("max_train_tracks"):
            cmd += ["--max_train_tracks", x.cfg["max_train_tracks"]]
        for k, v in x.glob.items():
            if k in oc:
                cmd += [f"--{k}", v]
        with x.timer("02 · Fine-tune OCR GP-LPR"):
            C.run_script(cmd)
        s = C.read_json(f"{x.CKPT_DIR}/ocr_summary.json")
        print(f"OCR trên ảnh HR (val): {s['exact_match_val_hr_truoc'] * 100:.2f}% -> {s['exact_match_val_hr_sau'] * 100:.2f}%")
        assert s["cong_kiem_tra_70_phan_tram"] or x.cfg["allow_low_ocr"], (
            "OCR sau fine-tune < 70% trên ảnh HR: kiểm tra lại tiền xử lý trước khi chạy tiếp (hoặc đặt allow_low_ocr: true).")

    x.ocr1, _ = C.load_gplpr(x.REPO_DIR, x.OCR_V1, x.device)   # OCR đóng băng, dùng chung cho mọi nhánh

    if x.want("m3") and x.branches(["m3"]):
        x.train_sr("m3", x.OCR_V1, "03 · Fine-tune LPSRGAN — M3")
    if x.want("eval_m3"):
        with x.timer("04a · Đánh giá trên VAL: M0, M1, HR, M2, M3"):
            for b in ("M0", "M1", "HR"):
                if os.path.exists(f"{x.RESULTS_DIR}/{b}/sr_outputs_val.npz"):
                    E.evaluate_branch(b, "val", x.rows["val"], x.RESULTS_DIR, x.ocr1, x.device, x.REPO_DIR, None)
                else:
                    print(f"Bỏ qua {b}: chưa có ảnh đã sinh (chạy bước baselines trước)")
            x.eval_sr("val", ["M2", "M3"])
    if x.want("m4") and x.branches(["m4"]):
        x.train_sr("m4", x.OCR_V1, "03 · Fine-tune LPSRGAN — M4")
    if x.want("eval_m4"):
        with x.timer("04a · Đánh giá trên VAL: M4"):
            x.eval_sr("val", ["M4"])

    cands = None
    if x.want("choose"):
        cands = x.choose("chosen_on_val.json")
    if x.want("test"):
        cands = cands or [b for b in R.available_branches(x.RESULTS_DIR) if b != "HR"]
        E.acquire_test_lock(x.EVID_DIR, f"{x.RESULTS_DIR}/chosen_on_val.json")
        with x.timer("04c · Đánh giá trên TEST (một lần)"):
            for b in cands:
                if b in ("M0", "M1"):
                    E.evaluate_branch(b, "test", x.rows["test"], x.RESULTS_DIR, x.ocr1, x.device, x.REPO_DIR, None)
            x.eval_sr("test", [b for b in cands if b in ("M2", "M3", "M4")])
        C.write_json(f"{x.RESULTS_DIR}/speed_ms_per_image.json", x.MS)
        print("Tốc độ mạng làm nét (ms/ảnh):", x.MS)
    if x.want("report"):
        x.report("", "05 · Tổng hợp kết quả & bằng chứng", "chosen_on_val.json", "speed_ms_per_image.json",
                 ["M3", "M4"], "comparison.md", with_val_plot=False, evidence_readme=True)
    if x.want("bundle"):
        x.bundle("ket_qua_gui_thay.zip")


# ----------------------------------------------------------------------------
# VÒNG 2
# ----------------------------------------------------------------------------
def round2(x):
    x.timing = "timing_v2.csv"
    x.start("v2")
    x.prepare_repo()
    x.load_rows()
    if x.want("check"):
        x.need({"danh sách track (splits)": f"{x.SPLIT_DIR}/split_train.csv", "OCR vòng 1": x.OCR_V1,
                "LPSRGAN M3": x.ckpt_of("M3"), "LPSRGAN M4": x.ckpt_of("M4"),
                "kết quả M2 trên test": f"{x.RESULTS_DIR}/M2/pred_test.csv",
                "ảnh M0 trên val": f"{x.RESULTS_DIR}/M0/sr_outputs_val.npz",
                "ảnh M2 trên test": f"{x.RESULTS_DIR}/M2/sr_outputs_test.npz"},
               "Thiếu kết quả vòng 1 (chạy --round 1 trước)")
        print("Kết quả vòng 1 còn đủ.")
        x.keep_copy(["comparison.md", "summary.csv"], "_v1")   # vòng 2 sẽ ghi đè bằng bản đầy đủ hơn
    x.ocr1, _ = C.load_gplpr(x.REPO_DIR, x.OCR_V1, x.device)

    names = x.branches(["m5a", "m5b", "m5c"])
    M5 = {x.branch_cfg(n)["branch"]: x.branch_cfg(n)["ocr_weight"] for n in names}
    sr_branches = ["M2", "M3", "M4"] + list(M5)

    if x.want("m5"):
        for n in names:
            b = x.branch_cfg(n)
            x.train_sr(n, x.OCR_V1, f"V1 · Fine-tune LPSRGAN + loss OCR — {b['branch']} (trọng số {b['ocr_weight']})")
    if x.want("eval_m5"):
        with x.timer("V1 · Đánh giá trên VAL: các nhánh M5"):
            x.eval_sr("val", list(M5))

    oc = x.branch_cfg("ocr_sr")
    if x.want("ocr_sr"):
        srs = [f"{b}={x.ckpt_of(b)}" for b in oc["adapt_from"] if os.path.exists(x.ckpt_of(b))]
        print("Dạy OCR trên ảnh của:", srs)
        cmd = [PY, f"{HERE}/train_ocr_sr.py", "--config", f"{HERE}/config/LRLPR/ocr_sr.yaml", "--repo", x.REPO_DIR,
               "--data_root", x.DATA_ROOT, "--split_dir", x.SPLIT_DIR, "--ocr_init", x.OCR_V1, "--out", x.OCR_V2,
               "--workers", x.WORKERS, "--seed", x.SEED, "--sr_ckpts"] + srs
        if x.cfg.get("max_train_tracks"):
            cmd += ["--max_train_tracks", x.cfg["max_train_tracks"]]
        for k, v in x.glob.items():
            if k in oc and k != "adapt_from":
                cmd += [f"--{k}", v]
        with x.timer("V2 · Dạy lại OCR trên ảnh GAN"):
            C.run_script(cmd)
        s = C.read_json(f"{x.CKPT_DIR}/ocr_sr_summary.json")
        print("Epoch tốt nhất:", s["epoch_tot_nhat"], "| EM ảnh HR (val):",
              f"{s['em_hr_val_truoc'] * 100:.1f}% -> {s['em_hr_val_sau'] * 100:.1f}%")
    x.ocr2, _ = C.load_gplpr(x.REPO_DIR, x.OCR_V2, x.device)
    if x.want("eval_plus"):
        with x.timer("V2 · Đánh giá trên VAL: các nhánh có dấu +"):
            x.eval_plus("val", sr_branches)

    if x.want("choose"):
        x.choose("chosen_on_val_v2.json")
    if x.want("test"):
        E.acquire_test_lock(x.EVID_DIR, f"{x.RESULTS_DIR}/chosen_on_val_v2.json", "test_run_v2.lock")
        with x.timer("V3 · Đánh giá trên TEST vòng 2 (một lần)"):
            x.eval_sr("test", list(M5))
            x.eval_plus("test", sr_branches)
        C.write_json(f"{x.RESULTS_DIR}/speed_ms_per_image_v2.json", x.MS)
        print("Tốc độ mạng làm nét (ms/ảnh):", x.MS)
    if x.want("report"):
        x.report("_v2", "V4 · Tổng hợp kết quả vòng 2", "chosen_on_val_v2.json", "speed_ms_per_image_v2.json",
                 ["M3", "M4"] + list(M5), "comparison_v2.md", with_val_plot=True)
    if x.want("bundle"):
        x.bundle("ket_qua_v2_gui_thay.zip")


# ----------------------------------------------------------------------------
# VÒNG 3
# ----------------------------------------------------------------------------
def round3(x):
    x.timing = "timing_v3.csv"
    x.start("v3")
    x.prepare_repo()
    x.load_rows()
    names = x.branches(["m6a"])
    M6 = {x.branch_cfg(n)["branch"]: x.branch_cfg(n)["ocr_weight"] for n in names}
    if x.want("check"):
        need = {"danh sách track (splits)": f"{x.SPLIT_DIR}/split_train.csv", "OCR vòng 1": x.OCR_V1,
                "OCR vòng 2 (dạy lại trên ảnh GAN)": x.OCR_V2, "kết quả vòng 2": f"{x.RESULTS_DIR}/chosen_on_val_v2.json",
                "ảnh M0 trên val": f"{x.RESULTS_DIR}/M0/sr_outputs_val.npz"}
        for n in names:
            init = x.branch_cfg(n)["init_from"]
            if init != "pretrained" and init != "M2":
                need[f"mạng khởi tạo {init}"] = x.ckpt_of(init)
        x.need(need, "Thiếu kết quả các vòng trước (chạy --round 1 và --round 2 trước)")
        print("Kết quả các vòng trước còn đủ.")
        x.keep_copy(["summary.csv"], "_v2")   # vòng 3 sẽ ghi đè bằng bản đầy đủ hơn
    x.ocr1, _ = C.load_gplpr(x.REPO_DIR, x.OCR_V1, x.device)
    x.ocr2, _ = C.load_gplpr(x.REPO_DIR, x.OCR_V2, x.device)

    if x.want("m6"):
        for n in names:
            b = x.branch_cfg(n)
            x.train_sr(n, x.OCR_V1, f"W1 · Huấn luyện mạng nhiều frame — {b['branch']} (loss OCR {b['ocr_weight']})")
    if x.want("eval_m6"):
        with x.timer("W1 · Đánh giá trên VAL: các nhánh M6 và M6+"):
            x.eval_sr("val", list(M6))
            x.eval_plus("val", list(M6))
    if x.want("choose"):
        x.choose("chosen_on_val_v3.json")
    if x.want("test"):
        E.acquire_test_lock(x.EVID_DIR, f"{x.RESULTS_DIR}/chosen_on_val_v3.json", "test_run_v3.lock")
        with x.timer("W2 · Đánh giá trên TEST vòng 3 (một lần)"):
            x.eval_sr("test", list(M6))
            x.eval_plus("test", list(M6))
        C.write_json(f"{x.RESULTS_DIR}/speed_ms_per_image_v3.json", x.MS)
        print("Tốc độ mạng làm nét (ms/ảnh):", x.MS)
    if x.want("report"):
        x.report("_v3", "W3 · Tổng hợp kết quả vòng 3", "chosen_on_val_v3.json", "speed_ms_per_image_v3.json",
                 ["M5c"] + list(M6), "comparison_v3.md", with_val_plot=True)
    if x.want("bundle"):
        x.bundle("ket_qua_v3_gui_thay.zip")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", default="all", help="1, 2, 3 hoặc all (mặc định: all)")
    ap.add_argument("--stage", default="", help="chỉ chạy vài bước, ví dụ m3,eval_m3 (xem danh sách ở đầu file)")
    ap.add_argument("--only", default="", help="chỉ huấn luyện các nhánh này, ví dụ m5a,m5c")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--set", action="append", help="ghi đè tham số: key=value hoặc nhánh.key=value (có thể lặp lại)")
    args = ap.parse_args()
    rounds = [1, 2, 3] if args.round == "all" else [int(args.round)]
    for r in rounds:
        print(f"\n{'=' * 20} VÒNG {r} {'=' * 20}")
        {1: round1, 2: round2, 3: round3}[r](Ctx(args))


if __name__ == "__main__":
    main()
