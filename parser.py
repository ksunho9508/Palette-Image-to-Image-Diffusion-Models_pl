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
    parser.add_argument("--gpu", type=int, default=None)

    parser.parse_args()

    conf = C.Config.get().conf
    conf["target"] = target
    return conf
