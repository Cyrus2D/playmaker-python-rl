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
        self.last_ball_pos = Vector2D(-100, -100)
    
    def make_decision(self, agent: IAgent):
        player = agent.wm.self
        
        player_pos = Vector2D(player.position.x, player.position.y)
        player_body = player.body_direction
        player_vel = Vector2D(player.velocity.x, player.velocity.y)
        
        ball_pos = Vector2D(agent.wm.ball.position.x, agent.wm.ball.position.y)
        
        if self.last_ball_pos.dist(ball_pos) > 0.01:
            self.rl_model.update_memory(0, True)
        else:
            self.rl_model.update_memory(self.evaluate(agent.wm), False)
        self.last_ball_pos = ball_pos
        
        state = [
            player_pos.x() / 52.5,
            player_pos.y() / 34,
            player_body / 180,
            player_vel.x() / 3,
            player_vel.y() / 3,
            ball_pos.x() / 52.5,
            ball_pos.y() / 34
        ]
        
        rl_action = self.rl_model.get_action(state)
        pl, dl, pr, dr = rl_action[0] * 100, rl_action[1] * 180, rl_action[2] * 100, rl_action[3] * 180
        
        agent.add_action(pb2.PlayerAction(
            dash=pb2.Dash(
                power_l = pl,
                dir_l = dl,
                power_r = pr,
                dir_r = dr
            )
        ))
        
        