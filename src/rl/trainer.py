from enum import Enum
from multiprocessing.pool import INIT
import random
import re
from src.IAgent import IAgent
import service_pb2 as pb2
from pyrusgeom.vector_2d import Vector2D


class Phase(Enum):
    INIT = 0
    PLAYING = 1

class RL_Trainer:
    def __init__(self) -> None:
        self.phase = Phase.INIT
        self.cycle_counter = 0
    
    def make_decision(self, agent: IAgent):
        wm = agent.wm
        if wm.teammates == None or len(wm.teammates) == 0:
            print('No teammates')
            return
        
        if wm.game_mode_type != pb2.GameModeType.PlayOn:
            print("Changing game mode", wm.cycle)
            self.setGameMode(agent)
            return
        
        if self.phase == Phase.INIT:
            print('Initializing', wm.cycle)
            self.init(agent)
            return
        
        player = wm.teammates[0]
        player_pos = Vector2D(player.position.x, player.position.y)
        ball_pos = Vector2D(wm.ball.position.x, wm.ball.position.y)
        
        agent.add_action(pb2.TrainerAction(
            do_recover=pb2.DoRecover()
        ))
        
        if player_pos.dist(ball_pos) < 1:
            print('Player reached the ball', wm.cycle)
            self.reset(agent)
            return
        
        if self.cycle_counter > 100:
            print('Cycle counter reached 100', wm.cycle)
            self.reset(agent)
            return
        
        self.cycle_counter += 1
        print(f'{self.cycle_counter=}')
        return pb2.TrainerActions()
    
    def setGameMode(self, agent: IAgent):
        agent.add_action(
            pb2.TrainerAction(
                do_change_mode=pb2.DoChangeMode(
                    game_mode_type=pb2.GameModeType.PlayOn,
                    side=pb2.Side.LEFT
                )
            )
        )
    
    def init(self, agent: IAgent) -> pb2.TrainerActions:
        self.reset(agent)
        self.phase = Phase.PLAYING
    
    def reset(self, agent: IAgent) -> pb2.TrainerActions:
        self.cycle_counter = 0
        
        new_ball_pos = Vector2D(random.uniform(-10, 10), random.uniform(-10, 10))
        player_pos = Vector2D(0, 0)
        
        agent.add_action(
            pb2.TrainerAction(
                do_move_ball=pb2.DoMoveBall(
                    position=pb2.Vector2D(
                        x=new_ball_pos.x(),
                        y=new_ball_pos.y()
                    ),
                    velocity=pb2.Vector2D(
                        x=0,
                        y=0
                    )
                )
            )
        )
        
        agent.add_action(
            pb2.TrainerAction(
                do_move_player=pb2.DoMovePlayer(
                    our_side=True,
                    uniform_number=1,
                    position=pb2.Vector2D(
                        x=player_pos.x(),
                        y=player_pos.y()
                    ),
                    body_direction=0,
                )
            )
        )
        