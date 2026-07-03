from .img_vgdseg_map_fn import img_vgdseg_map_fn


def img_intseg_map_fn(example, output_ids_with_output=True, cond_type="phrase", image_token=None):
    return img_vgdseg_map_fn(example, output_ids_with_output, cond_type, image_token)
