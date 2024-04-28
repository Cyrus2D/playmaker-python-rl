import numpy as np
from src.rl.rl_model import RL_Model
import service_pb2 as pb2
from src.IDecisionMaker import IDecisionMaker
from src.DM_PlayOn import PlayOnDecisionMaker
from src.DM_SetPlay import SetPlayDecisionMaker
from src.IAgent import IAgent
from pyrusgeom.vector_2d import Vector2D


class DecisionMaker(IDecisionMaker):
    def __init__(self):
        self.playOnDecisionMaker = PlayOnDecisionMaker()
        self.setPlayDecisionMaker = SetPlayDecisionMaker()
        self.rl_model = RL_Model()
        self.last_state = None
        self.last_action = None
        self.last_reward = None
    
    def make_decision(self, agent: IAgent):
        if agent.wm.game_mode_type != pb2.GameModeType.PlayOn:
            return
        player = agent.wm.self
        player_pos = Vector2D(player.position.x, player.position.y)
        player_body = player.body_direction
        player_vel = Vector2D(player.velocity.x, player.velocity.y)
        
        ball_pos = Vector2D(agent.wm.ball.position.x, agent.wm.ball.position.y)
        
        current_state = np.array([
            player_pos.x() / 52.5,
            player_pos.y() / 34.,
            player_body / 180.,
            player_vel.x() / 3.,
            player_vel.y() / 3.,
            ball_pos.x() / 52.5,
            ball_pos.y() / 34.
        ]).reshape(1, -1)
        
        got_to_ball = False
        if ball_pos.dist(player_pos) < 1:
            reward = 1
            self.rl_model.update_memory(
                self.last_state,
                self.last_action,
                reward, # Erm!
                current_state,
            )
            got_to_ball = True
        elif self.last_state is not None:
            reward = self.evaluate(ball_pos, player_pos)
            self.rl_model.update_memory(
                self.last_state,
                self.last_action,
                reward,
                current_state,
            )        
        
        self.rl_model.learn()
        
        rl_action = self.rl_model.policy(current_state)
        print(rl_action)
        pl, dl, pr, dr = rl_action[0][0] * 100, rl_action[0][1] * 180, rl_action[0][2] * 100, rl_action[0][3] * 180
        
        agent.add_action(pb2.PlayerAction(
            bipedal_dash=pb2.BipedalDash(
                power_l = pl*100,
                dir_l = dl*360 - 180,
                power_r = pr*100,
                dir_r = dr*360 - 180
            )
        ))
        
        self.last_action = rl_action
        self.last_state = current_state
        self.last_reward = reward
        if got_to_ball:
            self.last_state = None
    
    def evaluate(self, ball_pos, player_pos):
        return -ball_pos.dist(player_pos) / 100