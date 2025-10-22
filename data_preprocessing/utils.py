from settings.instruments import INSTRUMENT_LOOKUP_TABLE, INSTRUMENT_NAME

def find_inst_index(inst):
    return INSTRUMENT_LOOKUP_TABLE.get(inst, len(INSTRUMENT_NAME)-1)
