import sys
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from integration.util import load_model_and_processor


def main(argv):
    image_path, sentence = argv
    model, processor = load_model_and_processor(
        "microsoft/kosmos-2-patch14-224", padding_side="right"
    )

    image = Image.open(image_path)
    inputs = processor(
        text="<grounding> " + sentence, images=image, return_tensors="pt"
    ).to("cuda", torch.float16)

    processed_image_raw = inputs["pixel_values"][0].permute(1, 2, 0).cpu().numpy()
    processed_image_raw = (
        (processed_image_raw - processed_image_raw.min())
        / (processed_image_raw.max() - processed_image_raw.min())
        * 255
    )
    processed_image_raw = processed_image_raw.astype(np.uint8)
    processed_image = Image.fromarray(processed_image_raw)

    tokens = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    token_results = []
    attentions = outputs.projection_attentions[0].mean(dim=0)
    for i, token in enumerate(tokens):
        token_results.append((token, attentions[:, i].cpu().numpy().reshape(8, 8)))

    started_sentence = False
    i = 0
    for token, attention in token_results:
        if token == "<grounding>":
            started_sentence = True
        else:
            if not started_sentence:
                continue

            i += 1
            normed_attention = (
                (attention - attention.min())
                / (attention.max() - attention.min())
                * 255
            )
            normed_attention = normed_attention.astype(np.uint8)
            attention_img = Image.fromarray(normed_attention).resize((224, 224))

            plt.imshow(processed_image)
            plt.imshow(attention_img, cmap="jet", alpha=0.5)
            plt.title(token)
            plt.tight_layout()
            plt.savefig(f"token-{i}.jpg")


if __name__ == "__main__":
    main(sys.argv[1:])
    # This works, but I'm not completely the attentions are actually done
    # right and correspond to the image regions and tokens indicated.
