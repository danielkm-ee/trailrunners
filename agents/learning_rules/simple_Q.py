import numpy

class simple_Q:

        # Simple_Q defines the value of a state
        # as the amount of food collected as a
        # function of the number of steps.
        # Using an optimal solution to the
        # SF trail, Simple_Q computes the
        # mean squared error of the given
        # agent's path vs the optimal solution.
        
        def __init__(self):

                loader = np.load("simple_Q.npz", allow_pickle = True)

                step = loader['step']
                food = loader['food']

                total_steps = np.shape(step)[0]
                
                
                self.optimal = dict()
                self.optimal_max = food[total_steps-1]

                for i in range(total_steps):

                        self.optimal.update({step[i] : food[i]})



        def get_Q(self, food, steps):
                
                if steps in self.optimal.keys():

                        optimalQ = self.optimal.get(steps)

                else:
                        optimalQ = self.optimal_max


                experimentQ = food

                loss = (optimalQ - experimentQ)**2

                return loss

                
                
