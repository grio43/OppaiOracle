from pathlib import Path

shard = Path('L:/Dab/OppaiOracle/test_downsample/shard_00000')

jpg = list(shard.glob('*.jpg')) + list(shard.glob('*.jpeg'))
png = list(shard.glob('*.png'))
webp = list(shard.glob('*.webp'))
bmp = list(shard.glob('*.bmp'))

print(f"Current file counts in shard_00000:")
print(f"  JPEGs: {len(jpg)}")
print(f"  PNGs: {len(png)}")
print(f"  WebP: {len(webp)}")
print(f"  BMP: {len(bmp)}")
print(f"  TOTAL: {len(jpg) + len(png) + len(webp) + len(bmp)}")
