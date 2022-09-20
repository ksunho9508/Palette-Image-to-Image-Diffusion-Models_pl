import argparse
import theconf as C


target = [
    "Hemorrhage",
    "HardExudate",
    "CWP",
    "Drusen",
    "VascularAbnormality",
    "Membrane",
    "ChroioretinalAtrophy",
    "MyelinatedNerveFiber",
    "RNFLDefect",
    "GlaucomatousDiscChange",
    "NonGlaucomatousDiscChange",
    "MacularHole",
]


def parse_argument():
    parser = C.ConfigArgumentParser()
    parser.add_argument("--test_gpu", type=int, default=None)
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
    # conf = C.Config.get().conf
    conf = vars(args)
    conf["target"] = target
    return conf
