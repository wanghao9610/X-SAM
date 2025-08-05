"""
This interface provides access to four datasets:
1) refclef
2) refcoco
3) refcoco+
4) refcocog
split by unc and google

The following API functions are defined:
REFER      - REFER api class
getRefIds  - get ref ids that satisfy given filter conditions.
getAnnIds  - get ann ids that satisfy given filter conditions.
getImgIds  - get image ids that satisfy given filter conditions.
getCatIds  - get category ids that satisfy given filter conditions.
loadRefs   - load refs with the specified ref ids.
loadAnns   - load anns with the specified ann ids.
loadImgs   - load images with the specified image ids.
loadCats   - load category names with the specified category ids.
getRefBox  - get ref's bounding box [x, y, w, h] given the ref_id
showRef    - show image, segmentation or box of the referred object with the ref
getMask    - get mask and area of the referred object given ref
showMask   - show mask of the referred object given ref
"""

import json
import os.path as osp
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from pycocotools import mask
from skimage import io

from xsam.utils.logging import print_log


# Copied and modified from https://github.com/lichengunc/refer/blob/master/refer.py
class REFER:
    def __init__(self, data_root, dataset="refcoco"):
        # print_log(f"Loading {dataset} dataset into memory...", logger="current")
        self.ROOT_DIR = osp.abspath(osp.dirname(__file__))
        self.DATA_DIR = osp.join(data_root, dataset)
        self.IMAGE_DIR = self._get_image_dir(data_root, dataset)

        splitBy = "umd" if dataset == "refcocog" else "unc"
        self.data = self._load_data(splitBy)
        self.createIndex()

    def _get_image_dir(self, data_root, dataset):
        if dataset in ["refcoco", "refcoco+", "refcocog"]:
            return osp.join(data_root, "images/train2014")
        elif dataset == "refclef":
            return osp.join(data_root, "images/saiapr_tc-12")
        else:
            raise ValueError(f"No refer dataset is called [{dataset}]")

    def _load_data(self, splitBy):
        tic = time.time()
        ref_file = osp.join(self.DATA_DIR, f"refs({splitBy}).p")
        instances_file = osp.join(self.DATA_DIR, "instances.json")

        data = {
            "refs": pickle.load(open(ref_file, "rb")),
            **json.load(open(instances_file, "r")),
        }
        # print_log(f"DONE (t={time.time() - tic:.2f}s)", logger="current")
        return data

    def createIndex(self):
        # print_log("Creating index...", logger="current")

        # Create mappings
        self.Refs = {ref["ref_id"]: ref for ref in self.data["refs"]}
        self.Anns = {ann["id"]: ann for ann in self.data["annotations"]}
        self.Imgs = {img["id"]: img for img in self.data["images"]}
        self.Cats = {cat["id"]: cat["name"] for cat in self.data["categories"]}

        # Create reverse mappings
        self.imgToRefs = self._create_img_to_refs()
        self.imgToAnns = self._create_img_to_anns()
        self.refToAnn = {ref["ref_id"]: self.Anns[ref["ann_id"]] for ref in self.data["refs"]}
        self.annToRef = {ref["ann_id"]: ref for ref in self.data["refs"] for _ in range(len(ref["sentences"]))}
        self.catToRefs = self._create_cat_to_refs()

        # Create sentence related mappings
        self.Sents = {}
        self.sentToRef = {}
        self.sentToTokens = {}
        for ref in self.data["refs"]:
            for sent in ref["sentences"]:
                self.Sents[sent["sent_id"]] = sent
                self.sentToRef[sent["sent_id"]] = ref
                self.sentToTokens[sent["sent_id"]] = sent["tokens"]

        # print_log("Index created.", logger="current")

    def _create_img_to_refs(self):
        imgToRefs = {}
        for ref in self.data["refs"]:
            imgToRefs.setdefault(ref["image_id"], []).append(ref)
        return imgToRefs

    def _create_img_to_anns(self):
        imgToAnns = {}
        for ann in self.data["annotations"]:
            imgToAnns.setdefault(ann["image_id"], []).append(ann)
        return imgToAnns

    def _create_cat_to_refs(self):
        catToRefs = {}
        for ref in self.data["refs"]:
            catToRefs.setdefault(ref["category_id"], []).append(ref)
        return catToRefs

    def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=""):
        image_ids = [image_ids] if isinstance(image_ids, int) else image_ids
        cat_ids = [cat_ids] if isinstance(cat_ids, int) else cat_ids
        ref_ids = [ref_ids] if isinstance(ref_ids, int) else ref_ids

        refs = self.data["refs"]

        if image_ids:
            refs = [ref for image_id in image_ids for ref in self.imgToRefs.get(image_id, [])]
        if cat_ids:
            refs = [ref for ref in refs if ref["category_id"] in cat_ids]
        if ref_ids:
            refs = [ref for ref in refs if ref["ref_id"] in ref_ids]
        if split:
            if split in ["testA", "testB", "testC"]:
                refs = [ref for ref in refs if split[-1] in ref["split"]]
            elif split in ["testAB", "testBC", "testAC"]:
                refs = [ref for ref in refs if ref["split"] == split]
            elif split == "test":
                refs = [ref for ref in refs if "test" in ref["split"]]
            elif split in ["train", "val"]:
                refs = [ref for ref in refs if ref["split"] == split]
            else:
                raise ValueError(f"No such split [{split}]")

        return [ref["ref_id"] for ref in refs]

    def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
        image_ids = [image_ids] if isinstance(image_ids, int) else image_ids
        cat_ids = [cat_ids] if isinstance(cat_ids, int) else cat_ids
        ref_ids = [ref_ids] if isinstance(ref_ids, int) else ref_ids

        if not (image_ids or cat_ids or ref_ids):
            return [ann["id"] for ann in self.data["annotations"]]

        if image_ids:
            anns = [ann for image_id in image_ids for ann in self.imgToAnns.get(image_id, [])]
        else:
            anns = self.data["annotations"]

        if cat_ids:
            anns = [ann for ann in anns if ann["category_id"] in cat_ids]

        ann_ids = [ann["id"] for ann in anns]

        if ref_ids:
            ids = set(ann_ids) & set(self.Refs[ref_id]["ann_id"] for ref_id in ref_ids)
            return list(ids)

        return ann_ids

    def getImgIds(self, ref_ids=[]):
        ref_ids = [ref_ids] if isinstance(ref_ids, int) else ref_ids
        if ref_ids:
            return sorted(list(set(self.Refs[ref_id]["image_id"] for ref_id in ref_ids)))
        return sorted(list(self.Imgs.keys()))

    def getCatIds(self):
        return list(self.Cats.keys())

    def loadRefs(self, ref_ids):
        if isinstance(ref_ids, int):
            return [self.Refs[ref_ids]]
        return [self.Refs[ref_id] for ref_id in ref_ids]

    def loadAnns(self, ann_ids):
        if isinstance(ann_ids, (int, str)):
            return [self.Anns[ann_ids]]
        return [self.Anns[ann_id] for ann_id in ann_ids]

    def loadImgs(self, image_ids):
        if isinstance(image_ids, int):
            return [self.Imgs[image_ids]]
        return [self.Imgs[image_id] for image_id in image_ids]

    def loadCats(self, cat_ids):
        if isinstance(cat_ids, int):
            return [self.Cats[cat_ids]]
        return [self.Cats[cat_id] for cat_id in cat_ids]

    def getRefBox(self, ref_id):
        ann = self.refToAnn[ref_id]
        return ann["bbox"]  # [x, y, w, h], where x, y is the lower-left point

    def showRef(self, ref, seg_box="seg"):
        ax = plt.gca()
        image = self.Imgs[ref["image_id"]]
        I = io.imread(osp.join(self.IMAGE_DIR, image["file_name"]))
        ax.imshow(I)

        for sid, sent in enumerate(ref["sentences"]):
            print_log(f"{sid + 1}. {sent['sent']}", logger="current")

        ann_id = ref["ann_id"]
        ann = self.Anns[ann_id]

        if seg_box == "seg":
            self._show_segmentation(ax, ann)
        elif seg_box == "box":
            self._show_bounding_box(ax, ref)

    def _show_segmentation(self, ax, ann):
        if isinstance(ann["segmentation"][0], list):
            polygons = [np.array(seg).reshape((len(seg) // 2, 2)) for seg in ann["segmentation"]]
            p = PatchCollection(
                [Polygon(poly, True, alpha=0.4) for poly in polygons],
                facecolors="none",
                edgecolors=(1, 1, 0, 0),
                linewidths=3,
                alpha=1,
            )
            ax.add_collection(p)
            p = PatchCollection(
                [Polygon(poly, True, alpha=0.4) for poly in polygons],
                facecolors="none",
                edgecolors=(1, 0, 0, 0),
                linewidths=1,
                alpha=1,
            )
            ax.add_collection(p)
        else:
            rle = ann["segmentation"]
            m = mask.decode(rle)
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.array([2.0, 166.0, 101.0]) / 255
            for i in range(3):
                img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack((img, m * 0.5)))

    def _show_bounding_box(self, ax, ref):
        bbox = self.getRefBox(ref["ref_id"])
        box_plot = Rectangle(
            (bbox[0], bbox[1]),
            bbox[2],
            bbox[3],
            fill=False,
            edgecolor="green",
            linewidth=3,
        )
        ax.add_patch(box_plot)

    def getMask(self, ref):
        ann = self.refToAnn[ref["ref_id"]]
        image = self.Imgs[ref["image_id"]]
        if isinstance(ann["segmentation"][0], list):
            rle = mask.frPyObjects(ann["segmentation"], image["height"], image["width"])
        else:
            rle = ann["segmentation"]
        m = mask.decode(rle)
        m = np.sum(m, axis=2).astype(np.uint8)
        area = sum(mask.area(rle))
        return {"mask": m, "area": area}


def print_ref_info(refer, split=None):
    ref_ids = refer.getRefIds(split=split)
    if split:
        print_log(f"There are {len(ref_ids)} {split} referred objects.", logger="current")
    else:
        print_log(f"Total referred objects: {len(ref_ids)}", logger="current")
        print_log(f"Total images: {len(refer.Imgs)}", logger="current")
        print_log(f"Images to references mapping: {len(refer.imgToRefs)}", logger="current")


def display_ref_with_more_than_one_sentence(refer, ref_ids):
    for ref_id in ref_ids:
        ref = refer.loadRefs(ref_id)[0]
        if len(ref["sentences"]) < 2:
            continue

        print_log(ref, logger="current")
        print_log(f"The label is {refer.Cats[ref['category_id']]}.", logger="current")

        plt.figure()
        refer.showRef(ref, seg_box="box")
        plt.show()


if __name__ == "__main__":
    refer = REFER(dataset="refcocog", splitBy="google")

    print_ref_info(refer)
    print_ref_info(refer, split="train")
    train_ref_ids = refer.getRefIds(split="train")
    display_ref_with_more_than_one_sentence(refer, train_ref_ids)
