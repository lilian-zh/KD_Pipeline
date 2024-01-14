from .node_off_distill import *
from .node_on_distill import *



KD_options = {
    "KD": {"Trainer": KDTrainer,
           "Extra_Params": {"T": 4}
           },
    "AT": {"Trainer": ATTrainer,
        "Extra_Params": {}
        },
    "Hint": {"Trainer": HintTrainer,
        "Extra_Params": {"hint_layer": 2}
        },
    "RKD": {"Trainer": RKDTrainer,
        "Extra_Params": {"w_d": 25,
                         "w_a": 50}
        },
    "SP": {"Trainer": SPTrainer,
        "Extra_Params": {}
        },
    "DML": {"Trainer": DML_Trainer,
           "Extra_Params": {"T": 4}
           },
    "PKT": {"Trainer": PKTTrainer,
        "Extra_Params": {}
        },
}


