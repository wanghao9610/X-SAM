from .vgdseg_map_fn import vgdseg_map_fn


def intseg_map_fn(example, output_ids_with_output=True, cond_type="phrase"):
    return vgdseg_map_fn(example, output_ids_with_output, cond_type)
