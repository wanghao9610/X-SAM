from .image_processing_sam import SamImageProcessor


class Sam2ImageProcessor(SamImageProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_image(self, image):
        return super().process_image(image)
