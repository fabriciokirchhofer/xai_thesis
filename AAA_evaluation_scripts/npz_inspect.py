#!/usr/bin/env python3

import argparse
import os
import sys
from typing import Optional

import numpy as np


def list_npz_contents(npz_path: str) -> None:
	with np.load(npz_path, allow_pickle=False) as data:
		print(f"File: {npz_path}")
		if len(data.files) == 0:
			print("(no arrays found)")
			return
		name_col_width = max(len(name) for name in data.files)
		print("Key".ljust(name_col_width), "\tShape\tDType\tMin\tMax")
		for name in data.files:
			arr = data[name]
			try:
				arr_min = np.nanmin(arr)
				arr_max = np.nanmax(arr)
			except Exception:
				arr_min = "-"
				arr_max = "-"
			print(
				f"{name.ljust(name_col_width)}\t{tuple(arr.shape)}\t{arr.dtype}\t{arr_min}\t{arr_max}"
			)


def select_array(npz_path: str, key: Optional[str]) -> tuple[str, np.ndarray]:
	data = np.load(npz_path, allow_pickle=False)
	if key is None:
		if len(data.files) == 0:
			data.close()
			raise ValueError("No arrays in NPZ file.")
		if len(data.files) > 1:
			available = ", ".join(data.files)
			data.close()
			raise ValueError(
				"Multiple arrays in NPZ; specify --key. Available: " + available
			)
		key = data.files[0]
	arr = data[key]
	return key, arr


def export_npy(arr: np.ndarray, out_path: str) -> None:
	out_dir = os.path.dirname(out_path)
	if out_dir and not os.path.exists(out_dir):
		os.makedirs(out_dir, exist_ok=True)
	np.save(out_path, arr)
	print(f"Saved NPY: {out_path}")


def save_png(arr: np.ndarray, out_path: str, cmap: str, vmin: Optional[float], vmax: Optional[float]) -> None:
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt  # noqa: E402

	if arr.ndim == 3 and arr.shape[-1] in (3, 4):
		image = arr
		use_cmap = None
	else:
		image = arr.squeeze()
		use_cmap = cmap

	if image.ndim not in (2, 3):
		raise ValueError(
			f"Can only save 2D grayscale or 3D RGB(A) arrays as PNG. Got shape {image.shape}"
		)
\n+	fig, ax = plt.subplots(figsize=(6, 6))
	ax.axis("off")
	im = ax.imshow(image, cmap=use_cmap, vmin=vmin, vmax=vmax)
	if use_cmap is not None:
		fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	out_dir = os.path.dirname(out_path)
	if out_dir and not os.path.exists(out_dir):
		os.makedirs(out_dir, exist_ok=True)
	plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
	plt.close(fig)
	print(f"Saved PNG: {out_path}")


def show_image(arr: np.ndarray, cmap: str, vmin: Optional[float], vmax: Optional[float]) -> None:
	import matplotlib.pyplot as plt

	if arr.ndim == 3 and arr.shape[-1] in (3, 4):
		image = arr
		use_cmap = None
	else:
		image = arr.squeeze()
		use_cmap = cmap

	if image.ndim not in (2, 3):
		raise ValueError(
			f"Can only display 2D grayscale or 3D RGB(A) arrays. Got shape {image.shape}"
		)

	plt.figure(figsize=(6, 6))
	plt.imshow(image, cmap=use_cmap, vmin=vmin, vmax=vmax)
	plt.axis("off")
	if use_cmap is not None:
		plt.colorbar(fraction=0.046, pad=0.04)
	plt.show()


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Inspect and visualize NumPy .npz files (list, show, save)."
	)
	parser.add_argument("path", help="Path to .npz file")
	parser.add_argument("--list", action="store_true", help="List arrays and exit")
	parser.add_argument("--key", help="Array key to use (if multiple arrays)")
	parser.add_argument("--summary", action="store_true", help="Print array stats")
	parser.add_argument("--export-npy", metavar="OUT.npy", help="Export selected array to .npy")
	parser.add_argument("--save-png", metavar="OUT.png", help="Save selected array as PNG")
	parser.add_argument("--show", action="store_true", help="Display image window (matplotlib)")
	parser.add_argument("--cmap", default="inferno", help="Matplotlib colormap for 2D arrays (default: inferno)")
	parser.add_argument("--vmin", type=float, help="Color scale minimum")
	parser.add_argument("--vmax", type=float, help="Color scale maximum")

	args = parser.parse_args()
	if not os.path.isfile(args.path):
		print(f"File not found: {args.path}", file=sys.stderr)
		return 1

	if args.list:
		list_npz_contents(args.path)
		return 0

	try:
		key, arr = select_array(args.path, args.key)
		print(f"Selected array: {key} shape={arr.shape} dtype={arr.dtype}")
		if args.summary:
			if np.issubdtype(arr.dtype, np.number):
				print(
					f"min={np.nanmin(arr)} max={np.nanmax(arr)} mean={np.nanmean(arr)} std={np.nanstd(arr)}"
				)
			else:
				print("(non-numeric dtype; summary limited)")

		if args.export_npy:
			export_npy(arr, args.export_npy)
		if args.save_png:
			save_png(arr, args.save_png, args.cmap, args.vmin, args.vmax)
		if args.show:
			show_image(arr, args.cmap, args.vmin, args.vmax)
		if not (args.export_npy or args.save_png or args.show or args.summary):
			print("Nothing to do. Use --list, --summary, --save-png, --export-npy, or --show.")
		return 0
	except Exception as exc:
		print(f"Error: {exc}", file=sys.stderr)
		return 2


if __name__ == "__main__":
	sys.exit(main())



