import numpy as np
import copy
from scipy.stats import norm
import random


class Agent:
    """ Class that is an individual agent of a population, and can carry a disease with various parameters"""
    def __init__(self, id) -> None:
        self.state = np.array([1,0,0,0]) # S, I, R, D
        self.id = id
        # self.infected = False
        # self.alive = True
        self.immunity = 0
        self.times_infected = {}
        self.disease = None
        self.disease_history = [] # list containing the past disease ids
        self.time_of_infection = 0
        # self.max_immunity = 1.0 # max_immunity
        # self.immunity_rate = 0.5 # how fast a person becomes immune, the kappa coefficient, 1 means fully immune after first infection -> standard SIR
        
        # self.is_quarantined = False

    def calc_immunity(self, disease=None):
        if (times_infected := self.times_infected.get(disease.id, 0)) == 0:
            return 0.0
        if disease.immunity_rate == 1:
            return disease.max_immunity
        # self.immunity = (self.times_infected + 1)/(self.times_infected + 1) - 1/(self.times_infected + 1)
        # if disease is None: # calculate a immunity from all the times infected
            # times_infected = self.times_infected
        # times_infected = self.times_infected.get(disease.id, 0)
        eps = 1e-8
        immunity = disease.max_immunity * (times_infected)/(times_infected + (1/(disease.immunity_rate+eps) - 1 ))
        return immunity
    
    def get_immunity(self, disease):
        self.immunity = self.calc_immunity(disease)
        return self.immunity

    def infect(self, other_agent, timestep, population_size=1, roll=None):
        #calc infection succes
        # times_infected = other_agent.times_infected.get(self.disease, 0)
        # if other_agent.is_susceptible():
        #     infection_rate = self.disease.alpha
        # else:
        infection_rate = self.disease.alpha/population_size * (1 - other_agent.get_immunity(self.disease))
        if infection_rate == 0.0:
            return

        if roll is None:
            roll = np.random.uniform(0, 1)
        if roll > infection_rate:
            return
        
        # if succes pass infection to other
        other_agent.infect_with_disease(self.disease, timestep)

    def infect_with_disease(self, disease, timestep):
        self.set_state_infected()
        self.disease = copy.copy(disease)
        self.time_of_infection = timestep

    def recover(self, current_time):
        # if ~self.is_infected():
        #     return
        roll = np.random.uniform(0, 1)
        if (current_time - self.time_of_infection) == 0:
            # recover_rate = 0
            return
        else:
            # recover_rate = norm.cdf((current_time - self.time_of_infection), loc=1/self.disease.beta, scale=1)
            recover_rate = self.disease.beta
        # recover_rate = self.disease.beta
        if roll > recover_rate:
            return
        self._recover()


    def _recover(self):
        self.set_state_recovered()
        if not self.disease.id in self.times_infected:
            self.times_infected[self.disease.id] = 1
        else: 
            self.times_infected[self.disease.id] += 1
        self.disease_history.append(self.disease.id)
        setattr(self, "disease", None)

        
    def is_infected(self):
        return self.state[1] == 1

    def is_susceptible(self):
        return self.state[0] == 1

    def get_state_as_string(self):
        pass

    def clear_state(self):
        self.state = np.zeros_like(self.state)


    def set_state_infected(self):
        self.clear_state()
        self.state[1] = 1
        
    def set_state_recovered(self):
        self.clear_state()
        self.state[2] = 1
        

    def set_state_dead(self):
        self.clear_state()
        self.state[3] = 1


class Population:
    """Class that handles the simulated population"""
    def __init__(self, size=0, agent_type=Agent):
        self.agent_type = agent_type
        if size > 0:
            self.init_population(size)
        else:
            self.agents = []
        self.subpopulations =  []

    def init_population(self, size):
        self.agents = [self.agent_type(i) for i in range(size)]
        self.len = len(self)
        self._set_status() # status is the count of each group
        self._set_agent_id_catalog()
        self.get_infected_idx()

        
    def synth_induce_infected(self, idx, disease, timestep=0):
        # if len(self.subpopulations) > 0:
        #     for subpop in self.subpopulations:
        #         subpop.synth_induce_infected(idx, disease, timestep)
        # else:
        if not isinstance(idx, list):
            idx = list(idx)
        for i in idx:
            self[i].infect_with_disease(disease, timestep)
        self._set_status()
        self.get_infected_idx()

    def get_status(self):
        self._set_status()
        return self.status

    def _set_status(self):
        # if len(self.subpopulations) > 0:
        #     for subpop in self.subpopulations:
        #         subpop._set_status()
        # else:
        self.full_status = np.zeros((self.len,4))
        for i, agent in enumerate(self.agents):
            self.full_status[[i],:] = self.full_status[[i],:] + agent.state
            # self.status += agent.state
        self.status = self.full_status.sum(axis=0)
        self.get_infected_idx()
        # print(self.status)

    def recover_infection(self, current_time):
        if len(self.subpopulations) > 0:
            for subpop in self.subpopulations:
                subpop.recover_infection(current_time)
        else:
            for i in self.current_infected_idx:
                self[i].recover(current_time)
        # self.get_infected_idx()

    def spread_infection(self, timestep, population_size=1):   
        if len(self.subpopulations) > 0:
            for subpop in self.subpopulations:
                subpop.spread_infection(timestep)
        else:
            for infected_agent in (self[idx] for idx in self.current_infected_idx):
                rolls = np.random.uniform(0, 1, len(self.current_not_infected_idx))
                for i, roll in zip(self.current_not_infected_idx ,rolls): # iterate over not infected
                    infected_agent.infect(self[i], timestep, self.len, roll)

        # self.get_infected_idx()

    def get_infected_idx(self):
        # rework to get index by the status count instead
        # self.current_infected_idx = []
        # for i in range(self.len): 
        #     if self[i].is_infected():
        #         self.current_infected_idx.append(i)
        # try:
        self.current_infected_idx = set((np.argwhere(self.full_status[:,1])).ravel())
        self.current_not_infected_idx = set((np.argwhere(self.full_status[:,1] == 0)).ravel())

        # except:
        #     pass
        # _, self.current_infected_idx = np.argwhere(self.full_status[:,1])

    def get_infected_iter(self):
        return (self[i] for i in self.current_infected_idx)
        # for i in self.current_infected_idx:
        #     yield self[i]

    def mix_population():
        # reset id_catalog
        pass

    def __len__(self):
        return len(self.agents)

    def __getitem__(self, key):
        # try:
        return self.agents[key]
        # except:
            # return self.agents[key.item()]

    def get_agent_by_id(self, id):
        idx = self.id_catalog[self.id_catalog[:][1] == id][0]
        return self.agents[idx]

    def _set_agent_id_catalog(self):
        self.id_catalog = [0]*self.len
        for i in range(len(self)):
            self.id_catalog[i] = [i, self[i].id]

    def add_subpopulation(self, subpopulation):
        self.subpopulations.append(subpopulation)

    
    def fill_subpopulations(self, agents=None):
        if agents == None:
            agents = self.agents
        shuffled_idx = random.sample(list(range(self.len)), self.len)
        for i, subpop in enumerate(self.subpopulations):
            sub_shuffled_idx = shuffled_idx[subpop.N_max * i : subpop.N_max * (i + 1) ]
            subpop.add_agents([agents[idx] for idx in sub_shuffled_idx])
            # subpop.init_subpopulation()


    def remix_subpopulations(self):
        if len(self.subpopulations) == 0:
            return
        returned_agents = []
        # idx_id = []
        for subpop in self.subpopulations:
            subpop_return = subpop.return_subpopulation()
            # idx_id.extend(subpop_return)
            returned_agents.extend(subpop_return)

        self.fill_subpopulations(returned_agents)
        self.agents = returned_agents




class Subpopulation(Population):
    def __init__(self, N_max):
        self.N_max = N_max
        # self.agents = []
        super().__init__()

    def init_subpopulation(self):
        self.len = len(self)
        self._set_status() # status is the count of each group
        self._set_agent_id_catalog()
        self.get_infected_idx()

    def return_subpopulation(self):
        # for idx_id, agent in zip(self.id_catalog, self.agents): # why make it a iter?
        #     yield idx_id, agent 
        agents = self.agents
        self.agents = []
        # self.init_subpopulation()
        return agents
    
    def add_agents(self, agent):
        self.agents.extend(agent)
        self.init_subpopulation()


    
class Disease():
    def __init__(self, id, alpha=0.2, beta=0.1):
        self.id = id
        self.alpha = alpha
        self.beta = beta
        self.immunity_rate = 1.0 # how fast a person becomes immune, the kappa coefficient, 1 means fully immune after first infection -> standard SIR
        self.max_immunity = 1.0 # max_immunity


class DiseaseSimulator:
    """Simulate a spreading epidemic by a simulated population"""
    def __init__(self) -> None:
        self.population = Population(1000)
        self.size = len(self.population)
        self.disease_a = Disease("A", 0.2 , 0.10)
        self.disease_b = Disease("B", 0.15 , 0.12)

        self.population.synth_induce_infected(list(range(1)), self.disease_a)
        self.population.synth_induce_infected(list(range(1)), self.disease_b)

        if True: # if use_subpopulations:
            self.max_contact_n = 100
            self.N_subpopulations = (self.size-1) // self.max_contact_n + 1 
            for i in range(self.N_subpopulations):
                self.population.add_subpopulation(Subpopulation(self.max_contact_n))

            self.population.fill_subpopulations()

    def __call__(self, *args):
        return self.run_simulation(*args)

    def run_simulation(self, timesteps=365, plot=False):
  
        self.simulation = np.full((timesteps, 4), np.nan)
        self.time = np.arange(timesteps).reshape(-1,1)

        

        # status = self.population.get_status()

        if plot:
            fig, ax = plt.subplots()
            lines = ax.plot(0, np.zeros((1, 4)))
            ax.set_xlim([0,timesteps])
            ax.set_ylim([0,self.size])
        for time in self.time:
            # for infected_agent in self.population.get_infected_iter():
                # limit infection by creating sub populations.
    
            self.population.spread_infection(time)
            self.population.recover_infection(time)

            self.population.remix_subpopulations()
            # population.kill()

            # if time == 100:
            #     self.population.synth_induce_infected([10], self.disease_b, time)



            status = self.population.get_status()
            self.simulation[time,:] = status


            # self.population.get_infected_idx()
            if plot:
                for line, sim_data in zip(lines, self.simulation.T):
                    line.set_data(self.time, sim_data)
                plt.pause(0.000000001)
            print(f"Timestep: {time} Status: {status}\r", end="")
            # print("\n")q

        return self.time, self.simulation

        # for i in range(self.size):


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    disease_simulator = DiseaseSimulator()
    time, simulation = disease_simulator.run_simulation(plot=True)

    np.save("SIRD_sim_data_test", simulation)
    # plt.plot(time, simulation)
    # plt.legend(['S','I','R','D'])
    plt.show()
