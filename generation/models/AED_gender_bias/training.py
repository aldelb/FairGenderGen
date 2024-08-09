import pytorch_lightning as pl
import constants.constants as constants
from pytorch_lightning.callbacks import ModelCheckpoint

class TrainModel():

    def __init__(self):
       super().__init__()
    
    def train_model(self, trainer_args, dm): 
        model = constants.model(no_speak_examples=dm.no_speak_examples, speak_examples=dm.speak_examples)
        if(constants.finetune and not constants.do_resume):
            model.gan = model.gan.load_from_checkpoint(constants.init_model)
            print("the gan model is loaded from", constants.init_model)
        #prendre le modèle de base et ajouter une couche reverse gradient ?? ca peut marcher comme ca ??
        # model.gender_classifier = model.load_from_checkpoint(torch.load(join(constants.dir_path, "generation", "utils", "gender_classifier.pt")))
        # model.gender_classifier.freeze() #je veux pas mettre à jour le gender classifier, je l'ai deja entrainé et il fonctionne comme je veux
        trainer = pl.Trainer(**trainer_args)
        trainer.fit(model, dm)