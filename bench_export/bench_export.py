"""Benchmark nnU-Net's export (resample the logits to the image, argmax, write): two source trees side by side.

No model, no GPU and no data: the script makes deterministic random float16 logits with the shapes of real CTs for
the TotalSegmentator organ model (Dataset291: 25 channels, 1.5 mm, default plans with resample_data_or_seg_to_shape
and NibabelIOWithReorient; its plans are embedded below) and runs export_prediction_from_logits on them, as the
export workers of nnUNetv2_predict do after the network.

1) one export at a time: thick_105 (512x512x105, 4 mm slices, separate z) and thin_212 (512x512x212, 1.5 mm);
2) three exports of thin_40 at the same time, like the 3 export workers of nnUNetv2_predict on a folder (-nps 3):
   base, branch with its default threads (num_threads_torch = default_num_processes per export) and branch with
   the threads capped to cpu_count // 3 per export.
Each variant runs 3 times, interleaved, in a fresh process. The written segmentations are compared byte by byte.

usage: python bench_export.py [--base DIR] [--branch DIR] [--only single|parallel] [--pause SECONDS]
--pause waits before every run so that a laptop can cool down; before every run the script also times a small fixed
single-threaded job (cpu_probe_s), which grows when the CPU throttles.
DIR is a checkout of nnU-Net (the directory that contains nnunetv2/); the installed nnunetv2 only provides the
dependencies. Writes results_<host>_<time>.json next to this script.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SINGLE = {"cases": ["thick_105", "thin_212"],
          "orders": [("base", "branch"), ("branch", "base"), ("base", "branch")]}
PARALLEL = {"case": "thin_40", "n": 3,
            "orders": [("base", "branch", "capped"), ("capped", "branch", "base"), ("branch", "base", "capped")]}
CAPPED_THREADS = max(1, (os.cpu_count() or 1) // PARALLEL["n"])

# plans.json of TotalSegmentator's Dataset291, 3d_fullres only
PLANS = json.loads("""{"dataset_name": "Dataset291_TotalSegmentator_part1_organs_1559subj", "plans_name": "nnUNetPlans",
 "original_median_spacing_after_transp": [1.5, 1.5, 1.5], "original_median_shape_after_transp": [227, 227, 240],
 "image_reader_writer": "NibabelIOWithReorient", "transpose_forward": [0, 1, 2], "transpose_backward": [0, 1, 2],
 "configurations": {"3d_fullres": {"data_identifier": "nnUNetPlans_3d_fullres", "preprocessor_name": "DefaultPreprocessor",
  "batch_size": 2, "patch_size": [128, 128, 128], "median_image_size_in_voxels": [227.0, 227.0, 239.0],
  "spacing": [1.5, 1.5, 1.5], "normalization_schemes": ["CTNormalization"], "use_mask_for_norm": [false],
  "UNet_class_name": "PlainConvUNet", "UNet_base_num_features": 32, "n_conv_per_stage_encoder": [2, 2, 2, 2, 2, 2],
  "n_conv_per_stage_decoder": [2, 2, 2, 2, 2], "num_pool_per_axis": [5, 5, 5],
  "pool_op_kernel_sizes": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
  "conv_kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], "unet_max_num_features": 320,
  "resampling_fn_data": "resample_data_or_seg_to_shape", "resampling_fn_seg": "resample_data_or_seg_to_shape",
  "resampling_fn_data_kwargs": {"is_seg": false, "order": 3, "order_z": 0, "force_separate_z": null},
  "resampling_fn_seg_kwargs": {"is_seg": true, "order": 1, "order_z": 0, "force_separate_z": null},
  "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
  "resampling_fn_probabilities_kwargs": {"is_seg": false, "order": 1, "order_z": 0, "force_separate_z": null},
  "batch_dice": true}},
 "experiment_planner_used": "ExperimentPlanner", "label_manager": "LabelManager",
 "foreground_intensity_properties_per_channel": {"0": {"max": 3606.0, "mean": -370.00039267657144, "median": -249.0,
  "min": -3139.0, "percentile_00_5": -1024.0, "percentile_99_5": 276.0, "std": 436.5998675471528}}}""")
LABELS = ["background", "spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach", "pancreas",
          "adrenal_gland_right", "adrenal_gland_left", "lung_upper_lobe_left", "lung_lower_lobe_left",
          "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right", "esophagus", "trachea",
          "thyroid_gland", "small_bowel", "duodenum", "colon", "urinary_bladder", "prostate", "kidney_cyst_left",
          "kidney_cyst_right"]
DATASET = {"labels": {n: i for i, n in enumerate(LABELS)}, "channel_names": {"0": "CT"}, "file_ending": ".nii.gz",
           "overwrite_image_reader_writer": "NibabelIOWithReorient"}
# image spacing (z, y, x), image shape, logits shape (the network output at 1.5 mm), affine of the NIfTI
CASES = {
    "thick_105": ([4.0, 0.8500000238418579, 0.8500000238418579], (105, 512, 512), (25, 280, 290, 290),
                  [[0.8500000238418579, 0, 0, -234.154999], [0, 0.8500000238418579, 0, -216.577011], [0, 0, 4.0, 1319.0]]),
    "thin_212": ([1.5, 0.744140625, 0.744140625], (212, 512, 512), (25, 212, 254, 254),
                 [[0.744140625, 0, 0, -192.12792969], [0, 0.744140625, 0, -59.12792969], [0, 0, 1.5, 200.0]]),
    "thin_40": ([1.5, 0.744140625, 0.744140625], (40, 512, 512), (25, 40, 254, 254),
                [[0.744140625, 0, 0, -192.12792969], [0, 0.744140625, 0, -59.12792969], [0, 0, 1.5, 200.0]]),
}


def properties(case):
    import numpy as np
    spacing, shape, _, affine = CASES[case]
    affine = np.array(affine + [[0, 0, 0, 1]], dtype=float)
    return {"nibabel_stuff": {"original_affine": affine, "reoriented_affine": affine.copy()}, "spacing": spacing,
            "shape_before_cropping": shape, "bbox_used_for_cropping": [[0, s] for s in shape],
            "shape_after_cropping_and_before_resampling": shape}


def child(case, logits_file, out_trunc, threads):
    import numpy as np
    t0 = time.perf_counter()
    import nnunetv2
    from nnunetv2.inference.export_prediction import export_prediction_from_logits
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    t_import = time.perf_counter() - t0
    pm = PlansManager(PLANS)
    cm = pm.get_configuration("3d_fullres")
    logits = np.load(logits_file)
    kwargs = {"num_threads_torch": int(threads)} if threads != "default" else {}
    t1 = time.perf_counter()
    export_prediction_from_logits(logits, properties(case), cm, pm, DATASET, out_trunc, False, **kwargs)
    print(json.dumps({"code": str(Path(nnunetv2.__file__).parent), "t_import": t_import,
                      "t_export": time.perf_counter() - t1}))


def make_logits(shape, path):
    import numpy as np
    if path.exists():
        return
    rng = np.random.default_rng(1234)
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    for c in range(shape[0]):
        out[c] = (rng.standard_normal(shape[1:], dtype=np.float32) * 3).astype(np.float16)
    out.flush()
    del out


def tree_memory(procs):
    """Working set and private bytes (Windows) or RSS and PSS (elsewhere) of the processes and their children."""
    import psutil
    rss = own = 0
    for proc in procs:
        try:
            tree = [proc] + proc.children(recursive=True)
        except psutil.Error:
            continue
        for p in tree:
            try:
                if sys.platform == "win32":
                    mi = p.memory_info()
                    rss += mi.rss
                    own += mi.private
                else:
                    mi = p.memory_full_info()
                    rss += mi.rss
                    own += getattr(mi, "pss", mi.uss)
            except psutil.Error:
                continue
    return rss, own


def cpu_probe():
    """Best of 3 timings of a fixed single-threaded scipy zoom; it gets slower when the CPU throttles."""
    import numpy as np
    from scipy.ndimage import zoom
    x = np.random.default_rng(0).standard_normal((40, 127, 127), dtype=np.float32)
    best = float("inf")
    for _ in range(3):
        t = time.perf_counter()
        zoom(x, 2, order=1)
        best = min(best, time.perf_counter() - t)
    return best


def run(src, variant, case, logits_file, work, n=1):
    """Start n exports of the same logits at the same time and wait for all of them; wall time and peak memory
    are for all of them together. variant: base, branch (default threads) or capped (branch, CAPPED_THREADS)."""
    import psutil
    side, threads = ("branch", str(CAPPED_THREADS)) if variant == "capped" else (variant, "default")
    env = dict(os.environ, PYTHONPATH=str(src[side]), PYTHONUNBUFFERED="1")
    for k in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        env.setdefault(k, str(work / k))
    outs = [work / f"out_{case}_{variant}_{i}" for i in range(n)]
    swap0 = psutil.swap_memory().used
    t0 = time.time()
    procs, readers, lines = [], [], []
    for out_trunc in outs:
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child", case, str(logits_file),
                              str(out_trunc), threads],
                             cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs.append(p)
        lines.append([])
        readers.append(threading.Thread(target=lambda p=p, ln=lines[-1]: ln.extend(p.stdout), daemon=True))
        readers[-1].start()
    ps = [psutil.Process(p.pid) for p in procs]
    peak_rss = peak_own = 0
    min_avail = psutil.virtual_memory().available
    while any(p.poll() is None for p in procs):
        rss, own = tree_memory(ps)
        peak_rss, peak_own = max(peak_rss, rss), max(peak_own, own)
        min_avail = min(min_avail, psutil.virtual_memory().available)
        time.sleep(0.2)
    wall = time.time() - t0
    infos = []
    for p, r, ln in zip(procs, readers, lines):
        r.join()
        out = "".join(ln)
        if p.returncode != 0:
            print(out[-3000:])
            raise SystemExit(f"{variant} {case} failed with exit code {p.returncode}")
        infos.append(json.loads(out.strip().splitlines()[-1]))
        if not Path(infos[-1]["code"]).resolve().is_relative_to(src[side]):
            raise SystemExit(f"{variant} imported nnunetv2 from {infos[-1]['code']}, not from {src[side]}. "
                             f"Put this script outside of the nnU-Net checkouts.")
    digests = []
    for out_trunc in outs:
        f = Path(str(out_trunc) + ".nii.gz")
        digests.append(hashlib.sha256(f.read_bytes()).hexdigest())
        f.unlink()
    rec = {"variant": variant, "case": case, "parallel": n, "threads_per_export": threads,
           "wall_s": round(wall, 1), "t_export_s": [round(i["t_export"], 1) for i in infos],
           "t_import_s": [round(i["t_import"], 1) for i in infos], "peak_rss_gb": round(peak_rss / 1e9, 2),
           "peak_private_or_pss_gb": round(peak_own / 1e9, 2), "min_available_gb": round(min_avail / 1e9, 2),
           "swap_growth_gb": round((psutil.swap_memory().used - swap0) / 1e9, 2), "code": infos[0]["code"]}
    return rec, digests


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=str(HERE / "src_base"), help="nnU-Net checkout to compare against (master)")
    ap.add_argument("--branch", default=str(HERE / "src_export"), help="nnU-Net checkout with the change")
    ap.add_argument("--only", choices=["single", "parallel"], help="run only one of the two parts")
    ap.add_argument("--pause", type=float, default=0, help="seconds to wait before every run (default 0)")
    args = ap.parse_args()
    src = {"base": Path(args.base).resolve(), "branch": Path(args.branch).resolve()}
    for k, d in src.items():
        if not (d / "nnunetv2" / "__init__.py").exists():
            raise SystemExit(f"--{k} {d}: no nnunetv2/ in there")
    import numpy, scipy, skimage, psutil, torch
    vm = psutil.virtual_memory()
    meta = {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "ram_total_gb": round(vm.total / 1e9, 1), "ram_available_gb_at_start": round(vm.available / 1e9, 1),
            "python": sys.version.split()[0], "numpy": numpy.__version__, "scipy": scipy.__version__,
            "skimage": skimage.__version__, "torch": torch.__version__,
            "nnUNet_def_n_proc": os.environ.get("nnUNet_def_n_proc", "8 (default)"),
            "capped_threads_per_export": CAPPED_THREADS, "base": str(src["base"]), "branch": str(src["branch"]),
            "only": args.only, "pause_s": args.pause}
    print(json.dumps(meta, indent=1), flush=True)
    if vm.available < 9e9:
        print(f"Only {vm.available / 1e9:.1f} GB of memory is free; master needs up to 7 GB for thin_212. "
              f"Close other programs if you can; the script goes on anyway.", flush=True)
    work = HERE / "work"
    work.mkdir(exist_ok=True)
    results, identical = [], {}
    plan = [(case, SINGLE["orders"], 1) for case in SINGLE["cases"] if args.only != "parallel"] + \
           [(PARALLEL["case"], PARALLEL["orders"], PARALLEL["n"])] * (args.only != "single")
    for case, orders, n in plan:
        key = case if n == 1 else f"{case} x{n} at the same time"
        logits_file = work / f"{case}_logits.npy"
        make_logits(CASES[case][2], logits_file)
        for rep, order in enumerate(orders):
            digests = {}
            for variant in order:
                time.sleep(args.pause)
                probe = cpu_probe()
                r, digests[variant] = run(src, variant, case, logits_file, work, n)
                r["rep"], r["cpu_probe_s"] = rep, round(probe, 3)
                results.append(r)
                print(f"{key} run {rep + 1}/{len(orders)} {variant}: all done after {r['wall_s']} s "
                      f"(export {', '.join(map(str, r['t_export_s']))} s), peak working set {r['peak_rss_gb']} GB, "
                      f"private/PSS {r['peak_private_or_pss_gb']} GB, swap growth {r['swap_growth_gb']} GB, "
                      f"cpu probe {r['cpu_probe_s']} s", flush=True)
            identical.setdefault(key, []).append(all(d == digests["base"][0] for v in order for d in digests[v]))
        print(f"{key}: segmentations byte-identical in all runs: {all(identical[key])}", flush=True)
        logits_file.unlink()
    name = HERE / f"results_{platform.node()}_{time.strftime('%Y%m%d-%H%M')}.json"
    name.write_text(json.dumps({"meta": meta, "byte_identical": identical, "runs": results}, indent=1))
    shutil.rmtree(work, ignore_errors=True)
    print(f"\nWrote {name.name}. Send this file back.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(*sys.argv[2:6])
    else:
        main()
