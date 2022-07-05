import tensorflow as tf
from typing import List, Dict, Set, Tuple

Accession = str
Lineage = str
Sequence = str
Devider = int
LineageAccessionsMap = Dict[Lineage, Tuple[List[Accession], List[Accession]]]
LineageLabelMap = Dict[Lineage, tf.Tensor]
