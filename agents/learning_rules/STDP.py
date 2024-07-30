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

                # eligibility timers for elig. trace values
                self.eh_timer = torch.zeros_like(self.elig_hidden)
                self.eo_timer = torch.zeros_like(self.elig_output)

                self.num_input = num_input
                self.num_hidden = num_hidden
                self.num_output = num_output


                # weight_modifiers
                self.weight_mod_hidden = torch.ones_like(self.elig_trace_hidden).to(device)
                self.weight_mod_output = torch.ones_like(self.elig_trace_output).to(device)


                self.w_s_max = w_s_max
                self.w_inc_hid = w_inc_hid
                self.w_inc_out = w_inc_out


        # Calculates eligibility traces
        def update(self, in_spk, hid_spk, out_spk):

                # for input-hidden trace, increase eligibility by
                # [self.window] for indices [i,j] where
                # in_spk[i] and hid_spk[j] are both 1
                new_elig_hidden = torch.mul(in_spk, torch.reshape(hid_spk, (self.num_hidden, self.num_input))) * window


                # Do the same for hidden-output trace
                new_elig_output = torch.mul(hid_spk, torch.reshape(out_spk, (self.num_output, self.num_hidden))) * window


                # Then traces need to be decayed and regularized;
                # Traces decay by 1 at each step (each call of update())
                # Only decay positive-valued indices
                decay_hidden = (self.eh_timer > 0)
                decay_output = (self.eo_timer > 0)

                self.eh_timer.sub_(decay_hidden)
                self.eo_timer.sub_(decay_output)


                self.eh_timer.add_(new_elig_hidden)
                self.eo_timer.add_(new_elig_output)


                # Trace values also need to stay between 0 and self.window
                regularization_hidden = torch.zeros(0)
                regularization_output = torch.zeros(0)

                torch.remainder(self.delta_elig_hidden, self.window, out=regularization_hidden)
                torch.remainder(self.delta_elig_output, self.window, out=regularization_output)

                self.eh_timer.sub_(regularization_hidden)
                self.eo_timer.sub_(regularization_output)

                # eligible synapses are those with timers > 0
                self.elig_hidden = (self.eh_timer > 0)
                self.elig_output = (self.eo_timer > 0)


        
        def weight_change(self, criticism):

                if criticism > 0:
                        
                        delta_w_s_hidden = torch.mul(self.weight_mod_hidden, criticism*self.w_inc_hid)

                        delta_w_s_output = torch.mul(self.weight_mod_output, criticism*self.w_inc_out)
                        
                else:
                        inverse = torch.sub(1, self.weight_mod_hidden)
                        delta_w_s_hidden = torch.mul(inverse, criticism*self.w_inc_hid)

                        inverse = torch.sub(1, self.weight_mod_output)
                        delta_w_s_output = torch.mul(inverse, criticism*self.w_inc_out)

                        
                self.weight_mod_hidden.add_(delta_w_s_hidden.mul(self.elig_hidden))
                self.weight_mod_output.add_(delta_w_s_output.mul(self.elig_output))


                # synapses should be multiplied by these values
                return self.weight_mod_hidden, self.weight_mod_output
