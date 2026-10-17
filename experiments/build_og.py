"""Build the 1200x630 social preview card (web/public/og.png)."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONTS = Path("C:/Windows/Fonts")


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for cand in (FONTS / name, Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if cand.exists():
            return ImageFont.truetype(str(cand), size)
    return ImageFont.load_default(size)


def main() -> None:
    w, h = 1200, 630
    card = Image.new("RGB", (w, h), "#fbfaf7")
    d = ImageDraw.Draw(card)
    d.rectangle([0, 0, w, 10], fill="#2a78d6")
    d.text((60, 48), "TwinLens", font=font("georgiab.ttf", 64), fill="#14171c")
    d.text(
        (60, 132),
        "Attribution graphs drop two derivatives. Restoring them keeps the graph linear",
        font=font("arial.ttf", 27),
        fill="#3a3f47",
    )
    d.text(
        (60, 170),
        "and lifts counterfactual accuracy: \u03c1 0.20 \u2192 0.53 (GPT-2), replicated on Qwen3.",
        font=font("arial.ttf", 27),
        fill="#3a3f47",
    )
    fig = Image.open(ROOT / "experiments/figures/F5_fix.png").convert("RGB")
    fw = w - 120
    fig = fig.resize((fw, int(fig.height * fw / fig.width)), Image.LANCZOS)
    card.paste(fig, (60, 230))
    d.text(
        (60, h - 52),
        "spiliosdimakopoulos.github.io/twinlens  \u00b7  DOI 10.5281/zenodo.23437556",
        font=font("arial.ttf", 22),
        fill="#6b7280",
    )
    out = ROOT / "web/public/og.png"
    card.save(out, optimize=True)
    print(out, card.size)


if __name__ == "__main__":
    main()
