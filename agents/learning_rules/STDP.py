import snntorch as snn

import torch
import torch.nn as nn


class Learner():

        def __init__(self, num_input, num_hidden, num_output, window, w_inc_hid, w_inc_out, w_s_max, device):

                self.device = device
                self.window = window

                # Eligibility traces for synapses: 1 if eligible, 0 otherwise
                self.elig_hidden = torch.zeros((num_hidden, num_input)).to(device)
                self.elig_output = torch.zeros((num_output, num_hidden)).to(device)

                # eligibility timers
                self.last_in = torch.zeros(num_input)
                self.last_hid = torch.zeros(num_hidden)
                self.last_out = torch.zeros(num_output)

                self.num_input = num_input
                self.num_hidden = num_hidden
                self.num_output = num_output


                # weight_modifiers
                self.weight_mod_hidden = torch.ones_like(self.elig_hidden).to(device)
                self.weight_mod_output = torch.ones_like(self.elig_output).to(device)


                self.w_s_max = w_s_max
                self.w_inc_hid = w_inc_hid
                self.w_inc_out = w_inc_out
                

        # Calculates eligibility traces
        def update(self, in_spk, hid_spk, out_spk):
                """
                # for input-hidden trace, increase eligibility by
                # [self.window] for indices [i,j] where
                # in_spk[i] and hid_spk[j] are both 1
                new_elig_hidden = torch.mul(torch.reshape(in_spk, (1, self.num_input)), torch.reshape(hid_spk, (self.num_hidden, 1))) * self.window


                # Do the same for hidden-output trace
                new_elig_output = torch.mul(torch.reshape(hid_spk, (1, self.num_hidden)), torch.reshape(out_spk, (self.num_output, 1))) * self.window


                # Then traces need to be decayed and regularized;
                # Traces decay by 1 at each step (each call of update())
                # Only decay positive-valued indices
                decay_hidden = (self.eh_timer > 0).float()
                decay_output = (self.eo_timer > 0).float()

                self.eh_timer.sub_(decay_hidden)
                self.eo_timer.sub_(decay_output)


                self.eh_timer.add_(new_elig_hidden)
                self.eo_timer.add_(new_elig_output)


                # Trace values also need to stay between 0 and self.window
                regularization_hidden = torch.zeros(0).to(self.device)
                regularization_output = torch.zeros(0).to(self.device)

                torch.remainder(new_elig_hidden, self.window, out=regularization_hidden)
                torch.remainder(new_elig_output, self.window, out=regularization_output)

                self.eh_timer.sub_(regularization_hidden)
                self.eo_timer.sub_(regularization_output)

                # eligible synapses are those with timers > 0
                self.elig_hidden = (self.eh_timer > 0).float()
                self.elig_output = (self.eo_timer > 0).float()
                """
                decay_in = (self.last_in > 0).float()
                decay_hid = (self.last_hid > 0).float()
                decay_out = (self.last_out > 0).float()

                self.last_in.sub_(decay_in)
                self.last_hid.sub_(decay_hid)
                self.last_out.sub_(decay_out)
                
                self.last_in.add_(torch.mul(in_spk, self.window))
                self.last_hid.add_(torch.mul(hid_spk, self.window))
                self.last_out.add_(torch.mul(out_spk, self.window))


                reg_in = torch.zeros(0).to(self.device)
                reg_hid = torch.zeros(0).to(self.device)
                reg_out = torch.zeros(0).to(self.device)


                torch.remainder(self.last_in, self.window, out=reg_in)
                torch.remainder(self.last_hid, self.window, out=reg_hid)
                torch.remainder(self.last_out, self.window, out=reg_out)

                self.last_in.sub_(reg_in)
                self.last_hid.sub_(reg_hid)
                self.last_out.sub_(reg_out)

                self.elig_hidden = torch.mul(torch.reshape(self.last_in, (1, self.num_input)), torch.reshape(self.last_hid, (self.num_hidden, 1))) * self.window
                self.elig_output = torch.mul(torch.reshape(self.last_hid, (1, self.num_hidden)), torch.reshape(self.last_out, (self.num_output, 1))) * self.window

                self.elig_hidden = (self.elig_hidden > 0).float()
                self.elig_output = (self.elig_output > 0).float()

        
        def weight_change(self, criticism):


                # If the critic returns a negative, we've performed badly.
                if criticism < 0:
                        
                        delta_w_s_hidden = torch.mul(self.weight_mod_hidden, criticism*self.w_inc_hid)

                        delta_w_s_output = torch.mul(self.weight_mod_output, criticism*self.w_inc_out)
                
                else:
                        inverse = torch.sub(1, self.weight_mod_hidden)
                        delta_w_s_hidden = torch.mul(inverse, criticism*self.w_inc_hid)

                        inverse = torch.sub(1, self.weight_mod_output)
                        delta_w_s_output = torch.mul(inverse, criticism*self.w_inc_out)

                delta_w_s_hidden.div_(self.w_s_max)
                delta_w_s_output.div_(self.w_s_max)
                        
                self.weight_mod_hidden.add_(delta_w_s_hidden.mul(self.elig_hidden))
                self.weight_mod_output.add_(delta_w_s_output.mul(self.elig_output))


                weight_hid = self.weight_mod_hidden
                weight_out = self.weight_mod_output
                

                # synapses should be multiplied by these values
                return weight_hid, weight_out
