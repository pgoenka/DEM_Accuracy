class PipelineStage:

    def __init__(
        self,
        name,
        fn,
    ):

        self.name = name

        self.fn = fn

    def run(self):

        print(f"\n===== {self.name.upper()} =====")

        self.fn()