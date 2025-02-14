import pprint

class ModelParam:
    model_param_dict = {}

    def __init__(self, doc_type, doc_file_path, doc_prompt_header=None, doc_prompt_footer=None):
        self.model_param_dict["doc_type"]      = doc_type
        self.model_param_dict["doc_file_path"] = doc_file_path
        self.model_param_dict["doc_prompt_header"] = doc_prompt_header
        self.model_param_dict["doc_prompt_footer"] = doc_prompt_footer
        pprint.pprint(self.model_param_dict)
        
#x = ModelParam("pdf", "sample_doc\rsu.pdf")
