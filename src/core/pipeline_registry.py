class PipelineRegistry:

    def __init__(self):

        self.stages = []

    def add(self, stage):

        self.stages.append(stage)

    def run(self):

        for stage in self.stages:

            stage.run()