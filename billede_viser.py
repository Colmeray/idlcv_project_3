from PIL import Image
import matplotlib.pyplot as plt


billede_path = "DRIVE/test/images/01_test.tif"

path = "/dtu/datasets1/02516/" + billede_path

mask = Image.open(path).convert("L")

plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.savefig("pictures/mask_preview.png")
plt.show()

# skriv i terminal:
# uv run --no-project --with pillow --with matplotlib billede_viser.py