from enum import Enum
from multiprocessing.pool import INIT
import random
import re
import service_pb2 as pb2
from pyrusgeom.vector_2d import Vector2D


class Phase(Enum):
    INIT = 0
    PLAYING = 1

class RL_Trainer:
    def __init__(self) -> None:
        self.phase = Phase.INIT
        self.cycle_counter = 0
    
    def make_decision(self, wm: pb2.WorldModel) -> pb2.TrainerActions:
        if len(wm.teammates) == 0:
            return pb2.TrainerActions()
        
        if wm.game_mode_type != pb2.GameModeType.PlayOn:
            return self.setGameMode()
        
        if self.phase == Phase.INIT:
            return self.init(wm)
        
        player = wm.teammates[0]
        player_pos = Vector2D(player.position.x, player.position.y)
        ball_pos = Vector2D(wm.ball.position.x, wm.ball.position.y)
        
        if player_pos.distance(ball_pos) < 1:
            print('Player reached the ball', wm.cycle)
            return self.reset()
        
        if self.cycle_counter > 100:
            print('Cycle counter reached 100', wm.cycle)
            return self.reset()
        
        self.cycle_counter += 1
        print(f'{self.cycle_counter=}')
        return pb2.TrainerActions()
    
    def setGameMode(self) -> pb2.TrainerActions:
        actions = pb2.TrainerActions()
        actions.actions.append(
            pb2.TrainerAction(
                do_change_mode=pb2.DoChangeMode(
                    game_mode_type=pb2.GameModeType.PlayOn
                )
            )
        )
        
        return actions
    
    def init(self, wm: pb2.WorldModel) -> pb2.TrainerActions:
        actions = self.reset()
        self.phase = Phase.PLAYING
        return actions
    
    def reset(self) -> pb2.TrainerActions:
        self.cycle_counter = 0
        
        new_ball_pos = Vector2D(random.uniform(-10, 10), random.uniform(-10, 10))
        player_pos = Vector2D(0, 0)
        
        actions = pb2.TrainerActions()
        actions.actions.append(
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
        
        actions.actions.append(
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
        
        return actions
        