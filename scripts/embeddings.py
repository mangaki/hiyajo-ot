import os
import numpy as np
from item_encoder import ItemEncoder

import logging

logger = logging.getLogger()


class Embedding:
    def __init__(self, npy_filename, posters_paths=None):
        self.npy_filename = npy_filename
        self.posters_paths = posters_paths or []

        self.npy = None
        self.work_ids = []

        if self.posters_paths:
            self.read_work_ids_from_posters_paths()

    def open_npy(self):
        self.npy = np.load(self.npy_filename)

    def read_posters_paths_from_filename(self, posters_filename, data_directory):
        self.posters_paths = []
        with open(posters_filename, "r") as f:
            for line in f:
                self.posters_paths.append(os.path.join(data_directory, line.strip()))

        self.read_work_ids_from_posters_paths()

    def read_work_ids_from_posters_paths(self):
        for p in self.posters_paths:
            work_id = os.path.basename(p)[: -len(".jpg")]
            try:
                self.work_ids.append(int(work_id))
            except ValueError:
                logger.error("Invalid poster data for {}!".format(work_id))

    @property
    def identifier(self):
        return os.path.basename(self.npy_filename)[len("embeddings-") : -len(".npy")]

    def __repr__(self):
        if self.npy is not None:
            return "<Embedding (opened): {} × {}>".format(*self.npy.shape)
        else:
            return "<Embedding (closed)>"

    @classmethod
    def from_filename(cls, embedding_filename, ignore_posters=False):
        data_directory = os.path.dirname(embedding_filename)
        emb_basename = os.path.basename(embedding_filename)
        poster_basename = (
            "paths-" + emb_basename[len("embeddings-") : -len(".npy")] + ".txt"
        )
        posters_filenames = os.path.join(data_directory, poster_basename)

        if not os.access(posters_filenames, os.R_OK) and not ignore_posters:
            raise ValueError("Posters paths are not available!")

        embedding = Embedding(embedding_filename)
        if not ignore_posters:
            embedding.read_posters_paths_from_filename(
                posters_filenames, data_directory
            )

        return embedding


def generate_mapping(work_directory, ignore_posters: bool = False):
    mapping = []
    files_list = os.listdir(work_directory)

    for filename in files_list:
        if filename.startswith("embeddings-"):
            try:
                embedding = Embedding.from_filename(
                    os.path.join(work_directory, filename),
                    ignore_posters=ignore_posters,
                )
                mapping.append(embedding)
                logger.info("{} has been loaded in the database".format(filename))
            except Exception as e:
                logger.error("{} is a lonely embedding ({})".format(filename, e))

    return mapping


def merge_and_order_embeddings(embeddings):
    flattened = []

    max_work_id = embeddings[0].work_ids[0]
    for embedding in embeddings:
        for index, work_id in enumerate(embedding.work_ids):
            U = embedding.npy[index].reshape((1, 512))
            flattened.append((U, work_id))
            max_work_id = max(max_work_id, work_id)

    ordered_embeddings = sorted(flattened, key=lambda item: item[1])

    nb_works = max_work_id + 1
    C = np.concatenate([x for x, _ in ordered_embeddings])
    work_id_encoder = {y: x for x, (_, y) in enumerate(ordered_embeddings)}
    encoder = ItemEncoder(work_id_encoder)

    return C.astype(np.int16), encoder
