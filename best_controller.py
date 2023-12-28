# Group 29's best implementation of the Kessler Controller. Adapted from Dr. Dick's simple controller
# The boundaries of the fuzzy set have been set to the best values detrimined by the GA in training.py

# This controller is actually made up of 2 fuzzy controllers, a movement controller which dictates thrust, and a firing contoller which
# dictates firing and turn rate. The firing controller is just Dr. Dick's simple controller, while the movement controller is our own design

# By: Jason Kim, Bhagya Patel, Bassam Dakhel

from kesslergame import KesslerController
from typing import Dict, Tuple
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import matplotlib as plt

class BestController(KesslerController):

    # Constructor for the controller. Adapted from Dr. Dick's controller. Builds and defines the 2 fuzzy controllers used using the best
    # parameters for the fuzzy sets detrimined through training.
    def __init__(self):
        self.eval_frames = 0

        # Defining input variables of the firing controller (dictates whether the agent is firing and its turn rate)
        # This controller is the simple controller created by Dr. Dick
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi,math.pi,0.1), 'theta_delta') # Radians due to Python

        # Defining output variables of the firing controller (turn rate and whether the ship is firing)
        ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')

        # Defining the input variables of the movement controller, this dicates the agent's thrust.

        # How fast the ship is moving in m/s (+ in direction of ship's heading, - in direction of ships heading)
        ship_speed = ctrl.Antecedent(np.arange(-240, 240,1), 'ship_speed')
        # How close is the closest asteroid to the ship in m
        asteroid_distance = ctrl.Antecedent(np.arange(0,3000,1), 'asteroid_distance')
        # Where is the asteroid relative to the ship's direction (0 - in front, 180 - behind, 90 - on the sides) in degrees
        asteroid_relative_direction = ctrl.Antecedent(np.arange(0,180,1), 'asteroid_relative_direction')

        # Defining the output variable of the movement controller (thrust)

        # How much thrust to apply to the ship in m/s^2
        thrust = ctrl.Consequent(np.arange(-480,480,1), 'thrust')

        # Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point). Taken from Dr. Dick's controller.
        bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0,0,0.03050900757814064])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0,0.03050900757814064,0.42470127451932677])
        bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.42470127451932677)
        
        # Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle). Taken from Dr. Dick's controller.
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -2.1676546439660713, -0.6048151438722156)
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-2.1676546439660713,-0.6048151438722156,0])
        theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-0.6048151438722156,0,0.6108372909891933])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [0,0.6108372909891933,1.5513545489411302])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe,0.6108372909891933,1.5513545489411302)

        # Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate. Taken from Dr. Dick's controller.
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-111.80323421909745,-111.80323421909745,-10.547437660995378])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-81.4259785860925,-10.547437660995378,0])
        ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-10.547437660995378,0,26.30024730933062])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [0,26.30024730933062,72.24212072607136])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [26.30024730933062,129.74213339349285,129.74213339349285])
        
        # Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
        #   and returned as the boolean 'fire', Taken from Dr. Dick's controller
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1]) 

        # Defining fuzzy sets for the ship_speed
        #   FN - fast negative
        #   SN - slow negative
        #   Z - zero
        #   SP - slow positive
        #   FP - fast positive
        ship_speed['FN'] = fuzz.zmf(ship_speed.universe, -42.059764056766994, -19.181764102696764)
        ship_speed['SN'] = fuzz.trimf(ship_speed.universe, [-42.059764056766994, -19.181764102696764, 0])
        ship_speed['Z'] = fuzz.trimf(ship_speed.universe, [-19.181764102696764, 0, 2.4920100108668075])
        ship_speed['SP'] = fuzz.trimf(ship_speed.universe, [0, 2.4920100108668075, 61.4266777997732])
        ship_speed['FP'] = fuzz.smf(ship_speed.universe, 2.4920100108668075, 61.4266777997732)

        # Defining fuzzy sets for the asteroid_distance
        #   VF - very far
        #   F - far
        #   M - medium
        #   C - close
        #   VC - very close
        asteroid_distance['VF'] = fuzz.smf(asteroid_distance.universe, 232.45150564453647, 769.5274627272959)
        asteroid_distance['F'] = fuzz.trimf(asteroid_distance.universe, [197.5227284769368, 232.45150564453647, 769.5274627272959])
        asteroid_distance['M'] = fuzz.trimf(asteroid_distance.universe, [103.75691436048285, 197.5227284769368, 232.45150564453647])
        asteroid_distance['C'] = fuzz.trimf(asteroid_distance.universe, [1.972087972589226, 103.75691436048285, 197.5227284769368])
        asteroid_distance['VC'] = fuzz.zmf(asteroid_distance.universe, 1.972087972589226, 103.75691436048285)

        # Defining fuzzy sets for the asteroid_relative_direction
        #   F - forward
        #   DF - diagonal forward
        #   S - side
        #   DB - diagonal back
        #   B - back
        asteroid_relative_direction['F'] = fuzz.zmf(asteroid_relative_direction.universe, 22.841661454973135, 67.81278182947733)
        asteroid_relative_direction['DF'] = fuzz.trimf(asteroid_relative_direction.universe, [22.841661454973135, 67.81278182947733, 90])
        asteroid_relative_direction['S'] = fuzz.trimf(asteroid_relative_direction.universe, [67.81278182947733, 90, 127.50579382349153])
        asteroid_relative_direction['DB'] = fuzz.trimf(asteroid_relative_direction.universe, [90, 127.50579382349153, 140.09655231093504])
        asteroid_relative_direction['B'] = fuzz.smf(asteroid_relative_direction.universe, 127.50579382349153, 140.09655231093504)

        # Defining fuzzy sets for the thrust
        #   LS - large stop
        #   SS - small stop
        #   Z - zero
        #   SF - small forward
        #   LF - large forward
        thrust['LS'] = fuzz.zmf(thrust.universe, -542.5608953322832, -376.73613141389137)
        thrust['SS'] = fuzz.trimf(thrust.universe, [-542.5608953322832, -376.73613141389137, 0])
        thrust['Z'] = fuzz.trimf(thrust.universe, [-376.73613141389137, 0, 276.9015302882445])
        thrust['SF'] = fuzz.trimf(thrust.universe, [0, 276.9015302882445, 334.9404869247085])
        thrust['LF'] = fuzz.smf(thrust.universe, 276.9015302882445, 334.9404869247085)
                        
        # Declaring fuzzy rules for the firing controller (which is Dr. Dick's controller)
        # The rules are just the rules defined in his controller
        target_rules = [
            ('L', 'NL', 'NL', 'N'),
            ('L', 'NS', 'NS', 'Y'),
            ('L', 'Z', 'Z', 'Y'),
            ('L', 'PS', 'PS', 'Y'),
            ('L', 'PL', 'PL', 'N'),
            ('M', 'NL', 'NL', 'N'),
            ('M', 'NS', 'NS', 'Y'),
            ('M', 'Z', 'Z', 'Y'),
            ('M', 'PS', 'PS', 'Y'),
            ('M', 'PL', 'PL', 'N'),
            ('S', 'NL', 'NL', 'Y'),
            ('S', 'NS', 'NS', 'Y'),
            ('S', 'Z', 'Z', 'Y'),
            ('S', 'PS', 'PS', 'Y'),
            ('S', 'PL', 'PL', 'Y')
        ]
        
        # Defining asteroid_relative_distance categories
        asteroid_relative_direction_categories = ['F', 'DF', 'S', 'DB', 'B']

        # Defining rules for the movement controller
        thrust_levels = {
            ('FN', 'VF'): 'LF', 
            ('FN', 'F', 'F'): 'LF', ('FN', 'F', 'DF'): 'LF', ('FN', 'F', 'S'): 'LF', 
            ('FN', 'F', 'DB'): 'SF', ('FN', 'F', 'B'): 'LF',
            ('FN', 'M', 'F'): 'LF', ('FN', 'M', 'DF'): 'LF', ('FN', 'M', 'S'): 'SF',
            ('FN', 'M', 'DB'): 'SF', ('FN', 'M', 'B'): 'SF',
            ('FN', 'C', 'F'): 'SF',  ('FN', 'C', 'DF'): 'SF', ('FN', 'C', 'S'): 'SF', 
            ('FN', 'C', 'DB'): 'LF', ('FN', 'C', 'B'): 'LF',  
            ('FN', 'VC'): 'SF',
            ('SN', 'VF'): 'LF', 
            ('SN', 'F', 'F'): 'LF', ('SN', 'F', 'DF'): 'LF', ('SN', 'F', 'S'): 'LF',
            ('SN', 'F', 'DB'): 'SF',  ('SN', 'F', 'B'): 'SF',
            ('SN', 'M', 'F'): 'LF', ('SN', 'M', 'DF'): 'LF', ('SN', 'M', 'S'): 'SF',
            ('SN', 'M', 'DB'): 'SF', ('SN', 'M', 'B'): 'SF',
            ('SN', 'C'): 'SF', 
            ('SN', 'VC', 'F'): 'SS', ('SN', 'VC', 'DF'): 'SS', ('SN', 'VC', 'S'): 'Z',
            ('SN', 'VC', 'DB'): 'Z',  ('SN', 'VC', 'B'): 'Z',
            ('Z', 'VF'): 'LF', 
            ('Z', 'F', 'F'): 'LF', ('Z', 'F', 'DF'): 'LF',
            ('Z', 'F', 'S'): 'LF', ('Z', 'F', 'DB'): 'SF',
            ('Z', 'F', 'B'): 'SF',
            ('Z', 'M', 'F'): 'LF', ('Z', 'M', 'DF'): 'LF', ('Z', 'M', 'S'): 'SF',
            ('Z', 'M', 'DB'): 'SF', ('Z', 'M', 'B'): 'SF',
            ('Z', 'C', 'F'): 'Z', ('Z', 'C', 'DF'): 'Z',  ('Z', 'C', 'S'): 'SF',
            ('Z', 'C', 'DB'): 'SF', ('Z', 'C', 'B'): 'SF',  
            ('Z', 'VC', 'F'): 'LS', ('Z', 'VC', 'DF'): 'LS', ('Z', 'VC', 'S'): 'SS',
            ('Z', 'VC', 'DB'): 'SS', ('Z', 'VC', 'B'): 'SS',
            ('SP', 'VF', 'F'): 'LF',  ('SP', 'VF', 'DF'): 'LF',  
            ('SP', 'VF', 'S'): 'LF',  ('SP', 'VF', 'DB'): 'SF',
            ('SP', 'VF', 'B'): 'SF',
            ('SP', 'F', 'F'): 'LF', ('SP', 'F', 'DF'): 'SF',
            ('SP', 'F', 'S'): 'SF', ('SP', 'F', 'DB'): 'SF',
            ('SP', 'F', 'DB'): 'SF',
            ('SP', 'M', 'F'): 'LF', ('SP', 'M', 'DF'): 'LF', ('SP', 'M', 'S'): 'SF',
            ('SP', 'M', 'DB'): 'SF', ('SP', 'M', 'B'): 'SF',
            ('SP', 'C', 'F'): 'SS', ('SP', 'C', 'DF'): 'Z', ('SP', 'C', 'S'): 'Z', 
            ('SP', 'VC', 'F'): 'LS', ('SP', 'VC', 'DF'): 'LS', ('SP', 'VC', 'S'): 'SS',
            ('SP', 'VC', 'DB'): 'Z', ('SP', 'VC', 'B'): 'Z',
            ('FP', 'VF'): 'SF', ('FP', 'F'): 'SF', 
            ('FP', 'M'): 'SF', 
            ('FP', 'C', 'F'): 'LS', ('FP', 'C', 'DF'): 'LS', ('FP', 'C', 'S'): 'Z',
            ('FP', 'C', 'DB'): 'SS',  ('FP', 'C', 'B'): 'SS',
            ('FP', 'VC', 'F'): 'LS',  ('FP', 'VC', 'DF'): 'LS',  ('FP', 'VC', 'S'): 'LS',
            ('FP', 'VC', 'DB'): 'Z',  ('FP', 'VC', 'B'): 'Z',
        }

        # Defining the movement control system.
        self.movement_control = ctrl.ControlSystem()
        for thrust_inputs in thrust_levels:
            if len(thrust_inputs) == 3:
                condition = (ship_speed[thrust_inputs[0]] & 
                            asteroid_distance[thrust_inputs[1]] & 
                            asteroid_relative_direction[thrust_inputs[2]])
                self.movement_control.addrule(ctrl.Rule(condition, thrust[thrust_levels[thrust_inputs]]))
            else:
                for asteroid_relative_direction_cat in asteroid_relative_direction_categories:
                    condition = (ship_speed[thrust_inputs[0]] & 
                                asteroid_distance[thrust_inputs[1]] & 
                                asteroid_relative_direction[asteroid_relative_direction_cat])
                    self.movement_control.addrule(ctrl.Rule(condition, thrust[thrust_levels[thrust_inputs]]))
    
        # Defining the firing control system. Recall that this is just Dr. Dick's controller
        self.targeting_control = ctrl.ControlSystem()
        for bullet_time_label, theta_delta_label, ship_turn_label, ship_fire_label in target_rules:
            rule = ctrl.Rule(
                bullet_time[bullet_time_label] & theta_delta[theta_delta_label], 
                (ship_turn[ship_turn_label], ship_fire[ship_fire_label])
            )
            self.targeting_control.addrule(rule)

    # Actions method for the controller. Adapted from Dr. Dick's controller.
    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool]:

        # Find the closest asteroid (disregards asteroid velocity)
        ship_pos_x = ship_state["position"][0]     # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]       
        closest_asteroid = None
        
        for a in game_state["asteroids"]:
            #Loop through all asteroids, find minimum Eudlidean distance
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            if closest_asteroid is None :
                # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
                closest_asteroid = dict(aster = a, dist = curr_dist)
                
            else:    
                # closest_asteroid exists, and is thus initialized. 
                if closest_asteroid["dist"] > curr_dist:
                    # New minimum found
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = curr_dist

        # closest_asteroid is now the nearest asteroid object. 
        # Calculate intercept time given ship & asteroid position, asteroid velocity vector, bullet speed (not direction).
        # Based on Law of Cosines calculation, see notes.
        
        # Side D of the triangle is given by closest_asteroid.dist. Need to get the asteroid-ship direction
        #    and the angle of the asteroid's current movement.
        # REMEMBER TRIG FUNCTIONS ARE ALL IN RADAINS!!!
        
        
        asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]
        
        asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)

        # Caluating the asteroid's postion relative to the ship in heading 
        asteroid_ship_heading = (180/math.pi) * math.atan2(-asteroid_ship_x,-asteroid_ship_y)

        # Converting this heading to a range of [-180, 180]
        if ship_state["heading"] > 180:
            adjusted_ship_heading = ship_state["heading"] - 360
        else:
            adjusted_ship_heading = ship_state["heading"]
        
        # Calcuating the difference in this heading with the ship's heading, this is used in the movement controller
        diff_heading = abs(asteroid_ship_heading-adjusted_ship_heading)

        
        asteroid_direction = math.atan2(closest_asteroid["aster"]["velocity"][1], closest_asteroid["aster"]["velocity"][0]) # Velocity is a 2-element array [vx,vy].
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
        asteroid_vel = math.sqrt(closest_asteroid["aster"]["velocity"][0]**2 + closest_asteroid["aster"]["velocity"][1]**2)
        bullet_speed = 800 # Hard-coded bullet speed from bullet.py
        
        # Determinant of the quadratic formula b^2-4ac
        targ_det = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * closest_asteroid["dist"])
        
        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
        intrcpt1 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
        intrcpt2 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))
        
        # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
        if intrcpt1 > intrcpt2:
            if intrcpt2 >= 0:
                bullet_t = intrcpt2
            else:
                bullet_t = intrcpt1
        else:
            if intrcpt1 >= 0:
                bullet_t = intrcpt1
            else:
                bullet_t = intrcpt2
                
        # Calculate the intercept point. The work backwards to find the ship's firing angle my_theta1.
        intrcpt_x = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * bullet_t
        intrcpt_y = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * bullet_t

        my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))
        
        # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
        shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])
        
        # Wrap all angles to (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi
        
        # Pass the inputs into the firing controller
        shooting = ctrl.ControlSystemSimulation(self.targeting_control,flush_after_run=1)

        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta
   
        # Get the defuzzified outputs (ship_fire and turn_rate)
        shooting.compute()
        turn_rate = shooting.output['ship_turn']
        
        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False

        # Pass the inputs into the movement controller
        movement = ctrl.ControlSystemSimulation(self.movement_control,flush_after_run=1)

        movement.input['asteroid_distance'] = closest_asteroid["dist"]
        movement.input['ship_speed'] = ship_state['speed']
        movement.input['asteroid_relative_direction'] = diff_heading

        # Defuzzify the thrust
        movement.compute()
        thrust = movement.output['thrust']

        self.eval_frames +=1

        # Return the 3 output variables
        return thrust, turn_rate, fire


    @property
    def name(self) -> str:
        return "Group 29's best controller!"