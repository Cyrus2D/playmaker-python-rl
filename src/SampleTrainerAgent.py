from abc import ABC
from src.rl.trainer import RL_Trainer
from src.IAgent import IAgent
import service_pb2 as pb2


class SampleTrainerAgent(IAgent, ABC):
    def __init__(self):
        super().__init__()
        self.serverParams: pb2.ServerParam = None
        self.playerParams: pb2.PlayerParam = None
        self.playerTypes: dict[pb2.PlayerType] = {}
        self.wm: pb2.WorldModel = None
        self.first_substitution = True
        self.rl_trainer = RL_Trainer()
    
    def get_actions(self, wm:pb2.WorldModel) -> pb2.TrainerActions:
        self.wm = wm
        self.rl_trainer.make_decision(self)
        actions = pb2.TrainerActions()
        actions.actions.extend(self.actions)
        return actions
    
    def set_params(self, params):
        if isinstance(params, pb2.ServerParam):
            self.serverParams = params
        elif isinstance(params, pb2.PlayerParam):
            self.playerParams = params
        elif isinstance(params, pb2.PlayerType):
            self.playerTypes[params.id] = params
        else:
            raise Exception("Unknown params type")