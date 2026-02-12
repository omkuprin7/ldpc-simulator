from enum import Enum


class Result(Enum):
    OK = "eOk"
    INVALID_INPUT = "eInvalidInput"
    INVALID_PATH = "eInvalidPath"
    DATA_TRANSFER_NOT_OK = "eDataTransferNotOk"


class InterleaverType(Enum):
    NONE = "eNone"
    REGULAR = "eRegular"
    RANDOM = "eRandom"
    SRANDOM = "eSRandom"


class LDPCDecoderType(Enum):
    BIT_FLIPPING = "eBitFlipping"
    SUM_PRODUCT = "eSumProduct"


class EncodingMethod(Enum):
    STANDARD = "standard"
    RICHARDSON_URBANKE = "richardson_urbanke"
