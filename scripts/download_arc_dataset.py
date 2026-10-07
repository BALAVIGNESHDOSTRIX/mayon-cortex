# ==============================================================================
# Copyright (c) 2026 INFIDOS LLP. All Rights Reserved.
# Developer: BALAVIGNESH M
# Company: INFIDOS LLP
# System: Mayon-Cortex Cognitive Architecture
#
# DUAL LICENSED: GNU Affero General Public License v3.0 (AGPL-3.0)
# or Commercial License.
#
# Open-source use is governed by the AGPL-3.0 license. For proprietary
# or commercial applications seeking exemption from copyleft requirements,
# a commercial license must be obtained from INFIDOS LLP.
# Contact: balavignesh@infidos.com | https://infidos.com
# ==============================================================================

"""
ARC-AGI 400 Evaluation Dataset Ingestion Utility
=================================================
Downloads and extracts the complete official François Chollet ARC-AGI
400 evaluation tasks into data/arc_evaluation/ for local benchmarking.
"""

import io
import os
import sys
import zipfile
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIR = os.path.join(REPO_ROOT, "data", "arc_evaluation")
ARC_ZIP_URL = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"


def download_arc_evaluation_set(target_dir: str = TARGET_DIR) -> int:
    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading official ARC-AGI repository from {ARC_ZIP_URL}...")
    req = urllib.request.Request(
        ARC_ZIP_URL,
        headers={"User-Agent": "Mayon-Cortex-Ingestion/1.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()

    print(f"Downloaded {len(content) / (1024 * 1024):.2f} MB. Extracting 400 evaluation tasks...")
    count = 0
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for name in z.namelist():
            if "/data/evaluation/" in name and name.endswith(".json"):
                base_name = os.path.basename(name)
                if not base_name:
                    continue
                file_bytes = z.read(name)
                dest_path = os.path.join(target_dir, base_name)
                with open(dest_path, "wb") as f_out:
                    f_out.write(file_bytes)
                count += 1

    print(f"Successfully extracted {count} evaluation tasks into: {target_dir}")
    return count


if __name__ == "__main__":
    count = download_arc_evaluation_set()
    sys.exit(0 if count == 400 else 1)
