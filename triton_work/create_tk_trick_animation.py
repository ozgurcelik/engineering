"""Generate the causal-attention Tk loop-bound animation used by flash_attention.md."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1200, 760
N_TILES = 6
BG = "#0b1020"
PANEL = "#121a2e"
GRID = "#31405f"
TEXT = "#f2f5ff"
MUTED = "#9ba9c6"
VISIBLE = "#35c58b"
ACTIVE = "#ffd166"
WASTED = "#ef6a73"
SKIPPED = "#27314a"
QUERY = "#65a7ff"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


TITLE = load_font(38, bold=True)
SUBTITLE = load_font(22)
PANEL_TITLE = load_font(24, bold=True)
LABEL = load_font(18, bold=True)
SMALL = load_font(16)
MONO = ImageFont.truetype(
    "/System/Library/Fonts/Menlo.ttc"
    if Path("/System/Library/Fonts/Menlo.ttc").exists()
    else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    17,
)


def centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill: str) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), text, font=font, fill=fill)


def rounded(draw: ImageDraw.ImageDraw, box, fill, outline=None, width=1, radius=14) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_matrix(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    title: str,
    query_tile: int,
    phase: int,
    optimized: bool,
) -> None:
    panel_w, panel_h = 535, 480
    rounded(draw, (x, y, x + panel_w, y + panel_h), PANEL, GRID, 2, 18)
    draw.text((x + 24, y + 20), title, font=PANEL_TITLE, fill=TEXT)

    cell = 48
    gx, gy = x + 120, y + 124
    centered(draw, (gx + N_TILES * cell / 2, gy - 45), "key tile  j  →", LABEL, MUTED)
    draw.text((x + 18, gy + N_TILES * cell / 2 - 18), "query", font=LABEL, fill=MUTED)
    draw.text((x + 31, gy + N_TILES * cell / 2 + 7), "tile i", font=LABEL, fill=MUTED)

    for j in range(N_TILES):
        centered(draw, (gx + j * cell + cell / 2, gy - 17), str(j), SMALL, MUTED)
    for i in range(N_TILES):
        centered(draw, (gx - 22, gy + i * cell + cell / 2), str(i), SMALL, QUERY if i == query_tile else MUTED)

    for i in range(N_TILES):
        for j in range(N_TILES):
            box = (gx + j * cell, gy + i * cell, gx + (j + 1) * cell - 3, gy + (i + 1) * cell - 3)
            # Keep the tile classification fixed throughout the animation. In
            # particular, upper-triangle crosses are drawn once and persist;
            # only the active-cell outline moves.
            if j < i:
                fill = VISIBLE if optimized else ACTIVE
            elif j == i:
                fill = ACTIVE
            else:
                fill = SKIPPED if optimized else WASTED
            rounded(draw, box, fill, BG, 1, 6)
            if j > i:
                draw.line((box[0] + 12, box[1] + 12, box[2] - 12, box[3] - 12), fill=BG, width=3)
                draw.line((box[2] - 12, box[1] + 12, box[0] + 12, box[3] - 12), fill=BG, width=3)

    # Emphasize the row currently owned by this Triton program.
    row_y = gy + query_tile * cell
    draw.rounded_rectangle((gx - 5, row_y - 5, gx + N_TILES * cell + 1, row_y + cell + 1), radius=8, outline=QUERY, width=4)

    # The baseline cursor walks through all key tiles. The optimized cursor
    # disappears once phase reaches Tk(i), showing that no loop iteration runs.
    active_j = phase if (not optimized or phase <= query_tile) else None
    if active_j is not None:
        cell_x = gx + active_j * cell
        draw.rounded_rectangle(
            (cell_x + 3, row_y + 3, cell_x + cell - 6, row_y + cell - 6),
            radius=6,
            outline=TEXT,
            width=4,
        )

    if optimized:
        tk = query_tile + 1
        code = f"Tk(i) = i + 1 = {tk}    →    range({tk})"
        if phase < query_tile:
            result = "fully visible → tl.dot only; mask branch is skipped"
            result_color = VISIBLE
        elif phase == query_tile:
            result = "causal boundary → tl.dot + elementwise mask"
            result_color = ACTIVE
        else:
            result = "loop already ended → no load, dot, or mask"
            result_color = MUTED
    else:
        code = f"Tk = {N_TILES}             →    range({N_TILES})"
        if phase > query_tile:
            result = "future tile → loaded and scored, then entirely masked"
            result_color = WASTED
        elif phase < query_tile:
            result = "fully visible, but the elementwise mask is still built"
            result_color = ACTIVE
        else:
            result = "causal boundary → tl.dot + elementwise mask"
            result_color = ACTIVE
    rounded(draw, (x + 24, y + 426, x + panel_w - 24, y + 463), "#0a0f1d", None, radius=8)
    draw.text((x + 38, y + 435), code, font=MONO, fill=ACTIVE if optimized else TEXT)
    draw.text((x + 24, y + 54), result, font=SMALL, fill=result_color)


def make_frame(query_tile: int, phase: int) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    centered(draw, (WIDTH / 2, 44), "The causal Tk trick", TITLE, TEXT)
    centered(
        draw,
        (WIDTH / 2, 83),
        f"Query tile i = {query_tile}: no query in this tile can see a key tile after j = {query_tile}",
        SUBTITLE,
        MUTED,
    )
    draw_matrix(draw, 45, 118, "Baseline causal loop", query_tile, phase, False)
    draw_matrix(draw, 620, 118, "Tk bound + mask guard", query_tile, phase, True)

    legend_y = 632
    legend = [(VISIBLE, "dot · no mask"), (ACTIVE, "dot + mask"), (WASTED, "computed · all −∞"), (SKIPPED, "skipped")]
    lx = 125
    for color, label in legend:
        rounded(draw, (lx, legend_y, lx + 24, legend_y + 24), color, radius=5)
        draw.text((lx + 34, legend_y + 2), label, font=SMALL, fill=TEXT)
        lx += 235 if label != "computed · all −∞" else 275

    centered(
        draw,
        (WIDTH / 2, 707),
        "Tk(i) = min(⌈(i + 1) Bq / Bk⌉, ⌈Nkeys / Bk⌉)  ·  equal tile sizes shown",
        MONO,
        MUTED,
    )
    return image


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "figures"
    frames: list[Image.Image] = []
    durations: list[int] = []
    for query_tile in range(N_TILES):
        for phase in range(N_TILES):
            frames.append(make_frame(query_tile, phase))
            durations.append(360 if phase < N_TILES - 1 else 900)

    gif_path = out_dir / "flash_attention_tk_trick.gif"
    webp_path = out_dir / "flash_attention_tk_trick.webp"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    frames[0].save(webp_path, save_all=True, append_images=frames[1:], duration=durations, loop=0, quality=88, method=6)
    print(gif_path)
    print(webp_path)


if __name__ == "__main__":
    main()
