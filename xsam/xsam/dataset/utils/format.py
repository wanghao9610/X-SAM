import random
import re


def format_cat_name(cat_name):
    cat_name = re.sub(r"\(.*\)", "", cat_name)
    cat_name = re.sub(r"\.", "", cat_name)
    cat_name = re.sub(r"_", " ", cat_name)
    cat_name = re.sub(r"-", " ", cat_name)
    cat_name = re.sub(r"  ", " ", cat_name)
    cat_name = cat_name.strip().lower()
    return cat_name


def format_caption(caption):
    caption = re.sub(r" +", " ", caption)
    caption = caption.strip()
    return caption


def format_parts_of_cat_name(cat_name):
    cat_splits = cat_name.strip().split(":")
    if len(cat_splits) == 1:
        cat_name = cat_splits[0].strip().split("_(")[0]
    if len(cat_splits) > 1:
        assert len(cat_splits) == 2
        main, part = cat_splits
        main = main.split("_(")[0].replace("_", " ").replace("-", " ").strip()
        part = part.split("_(")[0].replace("_", " ").replace("-", " ").strip()
        if random.random() < 0.5:
            cat_name = f"{main} {part}"
        else:
            cat_name = f"the {part} of the {main}"
    return cat_name
