import argparse
import re
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Rename Hikvision NVR-exported clips from their raw numeric "
                "filenames to a sortable timestamp, using the file list .txt "
                "exported alongside them."
)
parser.add_argument(
    "folder", nargs="?", default=".",
    help="Camera folder containing a data/ subfolder with the videos and "
         "'New Text Document.txt'. Defaults to the current directory, so "
         "running with no argument behaves like before (expects ./data)."
)
args = parser.parse_args()

folder = Path(args.folder) / "data"

txt_file = folder / "New Text Document.txt"

timestamp_pattern = re.compile(
    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
)

rename_list = []

with open(txt_file, encoding="utf-8") as f:
    for line in f:

        line = line.strip()

        match = timestamp_pattern.search(line)

        if not match:
            continue

        # timestamp แรกในบรรทัด
        timestamp = match.group()

        # ส่วนก่อน timestamp
        prefix = line[:match.start()]

        # ดึงเลขท้าย 17 หลัก
        file_id = re.search(
            r"(\d{17})$",
            prefix
        )

        if not file_id:
            print("SKIP:", line)
            continue

        old_name = (
                file_id.group(1)
                + ".mp4"
        )

        new_name = (
                timestamp
                .replace(":", "")
                .replace(" ", "_")
                + ".mp4"
        )

        rename_list.append(
            (old_name, new_name)
        )

print(
    f"พบรายการ {len(rename_list)} รายการ"
)

for old_name, new_name in rename_list:

    old_file = folder / old_name
    new_file = folder / new_name

    if old_file.exists():

        print(
            f"{old_name}"
            f" -> "
            f"{new_name}"
        )

        old_file.rename(
            new_file
        )

    else:

        print(
            f"NOT FOUND: {old_name}"
        )
