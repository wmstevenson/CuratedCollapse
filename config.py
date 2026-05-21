class Config:
    def __init__(self, train_dataset_path:str, test_dataset_path:str, target_name:str, audit_alpha: float | None, results_path:str, output_path:str,  plugin_name: str, generations, seed: int):

        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.target_name = target_name
        self.audit_alpha = audit_alpha
        self.results_path = results_path
        self.output_path = output_path

        self.plugin_name = plugin_name
        self.generations = generations

        self.seed = seed