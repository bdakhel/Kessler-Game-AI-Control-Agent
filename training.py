from controller import Controller
# from scott_controller import BestController
from best_controller import BestController
import EasyGA
from kesslergame import Scenario, KesslerGame, GraphicsType, TrainerEnvironment
import numpy as np
import random
import math

def test_scenarios():
    map_size_x = 1000
    map_size_y = 800
    
    my_test_scenario = Scenario(name='Test Scenario',
                            num_asteroids=25,
                            ship_states=[
                            {'position': (600, 400), 'angle': 90, 'lives': 30, 'team': 1},
                            {'position': (400, 400), 'angle': 90, 'lives': 30, 'team': 2},
                            ],
                            map_size=(map_size_x, map_size_y),
                            time_limit=60,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)

    return my_test_scenario

def game_env():
    game_settings = {'perf_tracker': True,
                'graphics_type': GraphicsType.Tkinter,
                'realtime_multiplier': 1,
                'graphics_obj': None}

    game = TrainerEnvironment(settings=game_settings)
    # game = KesslerGame(settings=game_settings)
    
    return game

def generate_chromosome():
    initial_theta_delta = random.uniform(-1*math.pi,-1*math.pi/4)
    second_theta_delta = random.uniform(initial_theta_delta+(math.pi/12), 0)
    third_theta_delta = random.uniform(0, math.pi/4)
    fourth_theta_delta = random.uniform(third_theta_delta+(math.pi/12), math.pi)
    theta_deltas = [initial_theta_delta, second_theta_delta, third_theta_delta, fourth_theta_delta]

    initial_relative_direction_angle = random.uniform(0,30)
    second_relative_direction_angle = random.uniform(initial_relative_direction_angle+5, 90)
    third_relative_direction_angle = random.uniform(90, 150)
    fourth_relative_direction_angle =random.uniform(third_relative_direction_angle+5, 180)
    relative_directions = [initial_relative_direction_angle, second_relative_direction_angle, third_relative_direction_angle, fourth_relative_direction_angle]

    small_bullet_time = random.uniform(0,0.05)
    large_bullet_time = random.uniform(small_bullet_time+0.025, 1)
    bullet_times = [small_bullet_time, large_bullet_time]

    initial_ship_speed = random.uniform(-70, -30)
    second_ship_speed = random.uniform(initial_ship_speed+10, 0)
    third_ship_speed = random.uniform(0, 30)
    fourth_ship_speed = random.uniform(third_ship_speed+10, 70)
    ship_speeds = [initial_ship_speed, second_ship_speed, third_ship_speed, fourth_ship_speed]

    intital_distance = random.uniform(0, 75)
    second_distance = random.uniform(intital_distance+25, 200)
    third_distance = random.uniform(second_distance+25, 400)
    fourth_distance = random.uniform(third_distance+25, 650)
    fifth_distance = random.uniform(fourth_distance+25, 850)
    distances = [intital_distance, second_distance, third_distance, fourth_distance, fifth_distance]

    intital_ship_turn = random.uniform(-180, -80)
    second_ship_turn = random.uniform(intital_ship_turn+10, -50)
    third_ship_turn = random.uniform(second_ship_turn+10, 0)
    fourth_ship_turn = random.uniform(0, 40)
    fifth_ship_turn = random.uniform(fourth_ship_turn+10, 100)
    sixth_ship_turn = random.uniform(fifth_ship_turn+10, 180)
    ship_turns = [intital_ship_turn, second_ship_turn, third_ship_turn, fourth_ship_turn, fifth_ship_turn, sixth_ship_turn]

    intital_thrust = random.uniform(-600, -300)
    second_thrust = random.uniform(intital_thrust+50, -80)
    third_thrust = random.uniform(80, 300)
    fourth_thrust = random.uniform(third_thrust+50, 600)
    thrusts = [intital_thrust, second_thrust, third_thrust, fourth_thrust]
    
    return theta_deltas + relative_directions + bullet_times + ship_speeds + distances + ship_turns + thrusts

def fitness_function(chromosome):
    print(chromosome.gene_value_list)
    game = game_env()
    test_game = test_scenarios()
    score, perf_data = game.run(scenario=test_game, controllers=[Controller(chromosome.gene_value_list), BestController()])
    our_deaths = score.teams[0].deaths
    best_deaths = score.teams[1].deaths
    if best_deaths == 0:
        best_deaths = 0.5
    fitness = our_deaths/best_deaths
    print("Our deaths: " + str(our_deaths))
    print("Best deaths: " + str(best_deaths))
    print("Fitness is: " + str(fitness))

    return fitness

ga = EasyGA.GA()
ga.chromosome_impl = lambda: generate_chromosome()
ga.chromosome_length = 29
ga.population_size = 20
ga.target_fitness_type = 'min'
ga.generation_goal = 10
ga.fitness_function_impl = fitness_function
# ga.evolve()
while ga.active():
    # Evolve only a certain number of generations
    ga.evolve()
    # Print the current generation
    ga.print_generation()
    # Print the best chromosome from that generations population
    ga.print_best_chromosome()
    # If you want to show each population
    #ga.print_population()
    # To divide the print to make it easier to look at
    print('-'*75)

ga.print_best_chromosome()