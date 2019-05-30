# This module contains logics of NAS, which includes
# 1. the choice of parent models to expand
# 2. the choice of children/hallu models to train.
# There were plan to make critics to learn to select them
# (However, we currently use FIFO/FILO)
# 3. the formulation of feature selection modules, which are also
# known as candidate or hallucinations.
# (These are currently in LayerInfo/CellNetworkInfo in the logics
# sample/add/contain/select_hallu)
#
# TODO
# 1. possibly separate the hallu related operation from the info module.

from petridish.nas_control.controller import (
    ControllerTypes, PetridishController, RecognitionController,
    MLPController, RNNController, QueueSortMethods,
    Q_PARENT, Q_HALLU, Q_CHILD, PetridishRecover
)

from petridish.nas_control.queue import (
    PetridishQueueEntry, PetridishQueue, PetridishHeapQueue,
    PetridishSortedQueue, IDX_CNT, IDX_PQE, IDX_PV
)