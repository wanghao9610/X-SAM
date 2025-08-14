from .vgd_seg_map_fn import vgd_seg_map_fn


def inter_seg_map_fn(example, output_ids_with_output=True, cond_type="phrase"):
    return vgd_seg_map_fn(example, output_ids_with_output, cond_type)
