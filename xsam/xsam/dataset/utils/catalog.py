import copy
import types
from collections import UserDict


# Copied and modified from detectron2.data.catalog.Metadata
class Metadata(types.SimpleNamespace):
    """
    A class that supports simple attribute setter/getter.
    It is intended for storing metadata of a dataset and make it accessible globally.

    Examples:
    ::
        # somewhere when you load the data:
        MetadataCatalog.get("mydataset").thing_classes = ["person", "dog"]

        # somewhere when you print statistics or visualize:
        classes = MetadataCatalog.get("mydataset").thing_classes
    """

    # the name of the dataset
    # set default to N/A so that `self.name` in the errors will not trigger getattr again
    name: str = "N/A"

    def __getattr__(self, key):
        # "name" exists in every metadata
        if len(self.__dict__) > 1:
            raise AttributeError(
                "Attribute '{}' does not exist in the metadata of dataset '{}'. Available "
                "keys are {}.".format(key, self.name, str(self.__dict__.keys()))
            )
        else:
            raise AttributeError(
                f"Attribute '{key}' does not exist in the metadata of dataset '{self.name}': " "metadata is empty."
            )

    def __setattr__(self, key, val):
        # Ensure that metadata of the same name stays consistent
        try:
            oldval = getattr(self, key)
            assert (
                oldval == val
            ), "Attribute '{}' in the metadata of '{}' cannot be set " "to a different value!\n{} != {}".format(
                key, self.name, oldval, val
            )
        except AttributeError:
            super().__setattr__(key, val)

    def as_dict(self):
        """
        Returns all the metadata as a dict.
        Note that modifications to the returned dict will not reflect on the Metadata object.
        """
        return copy.copy(self.__dict__)

    def set(self, **kwargs):
        """
        Set multiple metadata with kwargs.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def get(self, key, default=None):
        """
        Access an attribute and return its value if exists.
        Otherwise return default.
        """
        try:
            return getattr(self, key)
        except AttributeError:
            return default


# Copied and modified from detectron2.data.catalog.MetadataCatalog
class _MetadataCatalog(UserDict):
    """
    MetadataCatalog is a global dictionary that provides access to
    :class:`Metadata` of a given dataset.

    The metadata associated with a certain name is a singleton: once created, the
    metadata will stay alive and will be returned by future calls to ``get(name)``.

    It's like global variables, so don't abuse it.
    It's meant for storing knowledge that's constant and shared across the execution
    of the program, e.g.: the class names in COCO.
    """

    def get(self, name):
        """
        Args:
            name (str): name of a dataset (e.g. coco_2014_train).

        Returns:
            Metadata: The :class:`Metadata` instance associated with this name,
            or create an empty one if none is available.
        """
        assert len(name)
        r = super().get(name, None)
        if r is None:
            r = self[name] = Metadata(name=name)
        return r

    def list(self):
        """
        List all registered metadata.

        Returns:
            list[str]: keys (names of datasets) of all registered metadata
        """
        return list(self.keys())

    def remove(self, name):
        """
        Alias of ``pop``.
        """
        self.pop(name)

    def reset(self):
        """
        Reset the metadata catalog.
        """
        self.clear()

    def __str__(self):
        return "MetadataCatalog(registered metadata: {})".format(", ".join(self.keys()))


MetadataCatalog = _MetadataCatalog()
MetadataCatalog.__doc__ = (
    _MetadataCatalog.__doc__
    + """
    .. automethod:: MetadataCatalog.get
"""
)
