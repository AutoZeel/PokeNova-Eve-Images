import os
from pathlib import Path

from PIL import Image
from google import genai
from google.genai import types


REFERENCE_DIR = Path("Glacier")
OUTPUT_DIR = Path("generated")
OUTPUT_FILE = OUTPUT_DIR / "test-output.png"

MODEL = "gemini-3.1-flash-image"


def find_reference():
    extensions = {".png", ".jpg", ".jpeg", ".webp"}

    files = sorted(
        p for p in REFERENCE_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not files:
        raise RuntimeError("No Pokémon reference image found in Glacier/")

    return files[0]


def main():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    reference = find_reference()

    print(f"Using reference: {reference}")

    image = Image.open(reference)

    prompt = """
    Transform the PROVIDED Pokémon official artwork into a premium Glacier skin.

IMPORTANT: Use the uploaded Pokémon official artwork as the ONLY character reference.

CRITICAL POKÉMON PRESERVATION:
• Preserve the EXACT original Pokémon pose.
• Preserve the EXACT silhouette.
• Preserve the EXACT anatomy.
• Preserve the EXACT proportions.
• Preserve the EXACT facial features and expression.
• Preserve the EXACT body position and camera angle.
• Preserve the EXACT official artwork composition.
• Preserve all species-defining traits.
• Preserve tail shape, wing shape, ears, claws, horns, fins, fur, scales, markings, and body structure.
• Keep the Pokémon instantly recognizable.
• Do NOT redesign the Pokémon.
• Do NOT generate a new pose.
• Do NOT modify anatomy.
• This is a texture and cosmetic skin conversion only.

GLACIER SKIN DESIGN:
• Transform the Pokémon into a premium glacier-inspired skin.
• Apply beautiful icy blue, glacier cyan, aqua, white, pale turquoise, and frosted silver coloration.
• Use subtle glacial ice textures across the body.
• Add gentle crystalline reflections and soft frosted gradients.
• Integrate elegant glacier details.
• Small transparent ice crystal accents.
• Tiny snowflake accents.
• Delicate frost patterns.
• Thin natural ice veins.
• Soft frozen-glass appearance.
• Smooth glossy icy highlights.
• Delicate crystalline shimmer.
• Premium collectible skin quality.
• Clean, elegant, frozen aesthetic.

STYLE REFINEMENTS:
• Use a soft watercolor finish.
• Use a subtle painterly airbrush effect.
• Soft color transitions.
• Smooth gradients.
• Clean artwork readability.
• Elegant and minimal design language.
• Focus on color transformation rather than heavy decoration.
• Keep the Pokémon looking clean and premium.
• Decorations should be tasteful accents only.

DECORATION LIMITS:
• Maximum 2–4 visible ice crystal accents.
• Very minimal snowflake details.
• Very minimal frost patterns.
• No giant icicles.
• No heavy ice formations.
• No snow piles.
• No blizzards.
• No clutter.
• Most of the Pokémon body must remain clearly visible.

ICE EFFECTS:
• Subtle frosted crystal patterns.
• Gentle icy glow.
• Light reflective highlights.
• Soft crystalline shimmer.
• Slight translucent ice appearance.
• Body should NOT become fully transparent.
• Body should NOT appear made entirely of ice.
• Preserve clear Pokémon readability.

ART STYLE:
• Official Pokémon artwork quality.
• Professional Pokémon Unite-quality skin artwork.
• Clean linework.
• Soft cel shading.
• Smooth airbrushed rendering.
• High detail.
• Vibrant but tasteful colors.
• Highly polished premium cosmetic skin.

BACKGROUND:
• Pure white background (#FFFFFF).
• Clean studio presentation.
• No scenery.
• No mountains.
• No glaciers.
• No snow landscape.
• No ice cave.
• No props.
• No text.
• No frame.
• No shadows on the background.

NEGATIVE PROMPT:
Change of pose, anatomy changes, silhouette changes, expression changes, redesigned Pokémon, cropped Pokémon, excessive ice crystals, giant icicles, frozen environment, snowy landscape, blizzard effects, cluttered composition, too many accessories, heavy transparency, fully crystalline body, excessive glow, armor, clothing, weapons, loss of Pokémon readability.

GOAL:
Create a premium Pokémon Unite-style Glacier skin that preserves the exact original artwork while adding a subtle, elegant frozen theme with soft airbrushed rendering, icy gradients, delicate frost textures, and minimal glacier decorations.
"""

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=MODEL,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            response_format={
                "image": {
                    "aspect_ratio": "1:1",
                    "image_size": "1K",
                }
            },
        ),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved = False

    for part in response.parts:
        if part.thought:
            continue

        if image := part.as_image():
            image.save(OUTPUT_FILE)
            saved = True
            break

    if not saved:
        raise RuntimeError("Gemini did not return an image")

    print(f"Generated: {OUTPUT_FILE}")
    print(f"Reference: {reference}")


if __name__ == "__main__":
    main()
