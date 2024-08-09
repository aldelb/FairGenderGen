import pytorch_lightning as pl
import constants.constants as constants

class TrainModel():

    def __init__(self):
       super().__init__()
    
    def train_model(self, trainer_args, dm): 
        model = constants.model(no_speak_examples=dm.no_speak_examples, speak_examples=dm.speak_examples)
        trainer = pl.Trainer(**trainer_args)
        trainer.fit(model, dm)