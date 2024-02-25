from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import pickle
import numpy as np
from evaluate import evaluate_HIV, evaluate_HIV_population
from sklearn.ensemble import ExtraTreesRegressor
from tqdm import tqdm


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  
class ProjectAgent:
    def __init__(self): 
        self.Q = None
        
    def act(self, observation, use_random=False):
        if use_random:
            return env.action_space.sample()
        else:
            return self.greedy_action(self.Q,observation,env.action_space.n)
    def save(self, path):
        pickle.dump(self.Q, open(path, "wb"))

    def load(self):
      with open("src/model.pkl", "rb") as f:
        self.Q = pickle.load(f)

    def greedy_action(self,Q,s,nb_actions):
        Qsa = []
        for a in range(nb_actions):
            sa = np.append(s,a).reshape(1, -1)
            Qsa.append(Q.predict(sa))
        return np.argmax(Qsa)
    def collect_samples(self, env, horizon, disable_tqdm=False, print_done_states=False):
        s, _ = env.reset()
        S = []
        A = []
        R = []
        S2 = []
        D = []
        for _ in tqdm(range(horizon), disable=disable_tqdm):
            a = env.action_space.sample()
            s2, r, done, trunc, _ = env.step(a)
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done or trunc:
                s, _ = env.reset()
                if done and print_done_states:
                    print("done!")
            else:
                s = s2
        S = np.array(S)
        A = np.array(A).reshape((-1,1))
        R = np.array(R)
        S2= np.array(S2)
        D = np.array(D)
        return S, A, R, S2, D

    
    def rf_fqi(self, S, A, R, S2, D, iterations, nb_actions, gamma, disable_tqdm=False):
        nb_samples = S.shape[0]
        Qfunctions = []
        SA = np.append(S,A,axis=1)
        for iter in tqdm(range(iterations), disable=disable_tqdm):
            if iter==0:
                value=R.copy()
            else:
                Q2 = np.zeros((nb_samples,nb_actions))
                for a2 in range(nb_actions):
                    A2 = a2*np.ones((S.shape[0],1))
                    S2A2 = np.append(S2,A2,axis=1)
                    Q2[:,a2] = Qfunctions[-1].predict(S2A2)
                max_Q2 = np.max(Q2,axis=1)
                value = R + gamma*(1-D)*max_Q2
            
            Q = ExtraTreesRegressor()
            Q.fit(SA,value)
            Qfunctions.append(Q)
        return Qfunctions[-1]
    
    def train(self,gamma,itteration): 
        S, A, R, S2, D = self.collect_samples(env, 5000)
        nb_actions = env.action_space.n
        self.Q = self.rf_fqi(S, A, R, S2, D, itteration, nb_actions, gamma)
        
def evaluate_agent(agent, env, nb_episodes):
    total_rewards = []
    for _ in range(nb_episodes):
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(200):  # Exécute l'agent pendant un nombre fixe d'itérations (200)
            action = agent.act(obs)
            obs, reward, _, _,_ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
    return total_rewards

# Entraînement de l'agent
if __name__ == "__main__":
  gamma=0.95
  iteration=150
  agent = ProjectAgent()
  agent.train(gamma,iteration)
  print ("ok")
  # Évaluation de l'agent
  nb_episodes_evaluation = 2
  scores = evaluate_agent(agent, env, nb_episodes_evaluation)
  average_score = np.mean(scores)
  print (scores)
  print("Score moyen :", average_score//10**8)

  # Sauvegarde de l'agent entraîné
  agent.save("model.pkl")


