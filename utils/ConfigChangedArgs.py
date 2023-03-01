class ConfigChangedArgs:
    def __init__(self, config_type, selection, par_id, par_val):
        self.type = config_type
        self.selection = selection
        self.par_id = par_id
        self.par_val = par_val
